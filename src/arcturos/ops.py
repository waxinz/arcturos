"""Dispatch layer — kick off benchmark / eval runs against a live server.

Called from API endpoints (thin wrappers here), so the dashboard UI can
CREATE entities and START real work, not only view stored rows. Every
dispatch is synchronous one-shot work: a bench point, a full bench sweep,
or one eval suite replay. Long jobs are still bounded by timeouts; the UI
gets progress via on_point callbacks -> stored rows appear as they land.

Append-only is preserved: dispatch only ever INSERTs (runs, benchmarks,
eval_results). There is no UPDATE/DELETE path anywhere.
"""

from __future__ import annotations

import threading
import time
import uuid

from typing import Any

from arcturos import bench
from arcturos import bench_openai
from arcturos import compare
from arcturos import power
from arcturos import multiturn as mt
from arcturos import preflight

import httpx

# Bench transports: 'native' (llama.cpp /tokenize + /completion) and
# 'openai' (any /v1/chat/completions server: tabbyAPI, vLLM, litellm).
_BENCH_TRANSPORTS = ("native", "openai")

# Client-side token estimate of bench.DEFAULT_UNIT_TEXT per section.
# OpenAI-compatible servers expose no /tokenize endpoint (tabbyAPI returns
# 404), so prompt sizing cannot come from the server's own tokenizer; this
# estimate is measured against real tokenizers (Qwen2.5: 64, DeepSeek-V3: 65)
# and feeds the same margin logic as the native path. The STORED prompt_tokens
# always comes from the server's authoritative usage, not this constant.
_UNIT_TEXT_TOKENS_EST = 65


class DispatchError(ValueError):
    """Bad input / preflight failure — the caller's form needs fixing (422)."""


class DispatchFailure(RuntimeError):
    """Runtime failure against the target server — 502 per the API contract.

    main.py maps DispatchError -> 422 ("fix the form") and DispatchFailure
    -> 502 ("check the server"), so the UI error message points at the right
    next step instead of blaming the input for a dead target.
    """


def default_run_name(server_url: str, model_fingerprint: str,
                     targets: list[int]) -> str:
    """Humanized default run name (2026-09-22 naming feature).

    'host · model · 64k/128k' — host from the URL, model as given
    (registry alias rendering happens at display time), context targets
    humanized (32768 -> 64k, 1500 -> 1500). Used when the operator does
    not name a run explicitly.
    """
    host = compare.host_label(server_url)
    humanized = "/".join(
        f"{t // 1024}k" if t >= 1024 and t % 1024 == 0 else str(t)
        for t in targets)
    return f"{host} · {model_fingerprint} · {humanized}"


def _point_payload(p) -> dict:
    """One benchmark point as a JSON-ready dict (dispatch return payload)."""
    return {
        "context_tokens": p.target_tokens,
        "prefill_tps": p.prefill_tps,
        "decode_tps": p.decode_tps,
        "ttft_ms": p.ttft_ms,
        "wall_s": p.wall_s,
        "output_tokens": p.output_tokens,
        "mtp_draft_n": p.mtp_draft_n,
        "mtp_accepted": p.mtp_accepted,
        "prompt_tokens": p.prompt_tokens,
        "stop_reason": p.stop_reason,
        "power_watts": p.power_watts,
        "power_host": p.power_host,
        "power_gpu_index": p.power_gpu_index,
        "streams": p.streams,
        "decode_tps_combined": p.decode_tps_combined,
        "prefill_tps_combined": p.prefill_tps_combined,
    }


def dispatch_bench(
    db_path,
    server_url: str,
    targets: list[int],
    n_predict: int,
    timeout: float = 3600.0,
    transport=None,
    on_point=None,
    api_key: str | None = None,
    transport_name: str | None = None,
    model: str | None = None,
    streams: int = 1,
    power_host: str | None = None,
    power_gpu_index: int = 0,
    name: str | None = None,
) -> dict:
    """Run a cold-cache bench sweep; store run + points; return run payload.

    ``api_key`` authenticates every request (tokenize, completion, props)
    for key-protected llama.cpp servers; None is fine for open ones.

    ``transport_name`` selects the wire protocol:

    * ``'native'`` (default) — llama.cpp: preflight reachable + tokenize,
      points via the native ``/completion`` endpoint (server-authoritative
      timings, MTP acceptance).
    * ``'openai'`` — any OpenAI-compatible ``/v1/chat/completions`` server
      (tabbyAPI, vLLM, litellm). Preflight runs kind='eval' (reachable +
      chat, since these servers have no ``/tokenize`` — 404); prompts are
      cold UUID-prefixed and sized client-side (no server tokenizer);
      each point streams via ``bench_openai.run_openai_stream_point`` and
      reports client-side TTFT + decode with the prefill estimate.
      ``model`` (the chat model name) is required for this transport.

      Power is opt-in for every transport: pass ``power_host`` (a
      dedicated inference host per ADR 002) to run an SSH nvidia-smi
      sampler alongside each point; without it watts stay None. An
      OpenAI endpoint may sit behind a proxy with no knowable serving
      host, so the host must be named explicitly — never inferred.

    ``transport`` (the ``transport=None`` kwarg) is the injectable httpx
    transport used by the native path's preflight/completion/props and by
    tests; the openai path's SSE call is mocked via
    ``bench_openai.run_openai_stream_point`` in tests.
    """
    if transport_name is None:
        transport_name = "native"
    if transport_name not in _BENCH_TRANSPORTS:
        raise DispatchError(
            f"unknown transport {transport_name!r} — expected one of "
            f"{', '.join(_BENCH_TRANSPORTS)}")
    if transport_name == "openai" and not (isinstance(model, str) and model):
        raise DispatchError(
            "transport 'openai' requires 'model' (the chat model name the "
            "target serves)")
    if not targets:
        raise DispatchError("targets list is empty")
    if any(t < 1 for t in targets):
        raise DispatchError("every target must be >= 1 token")
    if n_predict < 1:
        raise DispatchError("n_predict must be >= 1")
    if not isinstance(streams, int) or isinstance(streams, bool) \
            or not (1 <= streams <= 16):
        raise DispatchError("streams must be an int between 1 and 16")
    if power_host is not None and (not isinstance(power_host, str)
                                   or not power_host.strip()):
        raise DispatchError("power_host must be a non-empty string when set")
    pf = preflight.preflight(
        server_url,
        kind="bench" if transport_name == "native" else "eval",
        model=model, transport=transport, api_key=api_key)
    if not pf.ok():
        raise DispatchError(
            _preflight_detail(server_url, pf))
    # Partial-run preservation (2026-09-22): points completed before a
    # mid-sweep failure are still stored — a failed sweep discards nothing
    # it already measured. on_point appends as points finish, so the
    # on_point callback's list IS the partial record.
    partial_points: list = []

    def _collecting_on_point(point) -> None:
        partial_points.append(point)
        if on_point is not None:
            on_point(point)

    if transport_name == "native":
        try:
            points = bench.run_benchmark(
                server_url, "default", targets, n_predict,
                timeout=timeout, transport=transport,
                on_point=_collecting_on_point,
                api_key=api_key, streams=streams,
                power_host=power_host, power_gpu_index=power_gpu_index)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            # httpx.HTTPError: connection-level failure (server died
            # mid-sweep) — previously escaped as an unhandled 500.
            if not partial_points:
                raise DispatchFailure(
                    f"bench failed against {server_url}: {exc} "
                    "— check the server is still up") from exc
            engine_meta = bench.capture_engine_metadata(
                server_url, transport=transport, api_key=api_key)
            run_id = bench.store_benchmark_run(
                db_path, server_url, engine_meta, partial_points,
                engine=None, status="partial", name=name)
            return {
                "run_id": run_id,
                "status": "partial",
                "error": str(exc),
                "engine_metadata": engine_meta,
                "points": [_point_payload(p) for p in partial_points],
            }
        engine_meta = bench.capture_engine_metadata(server_url,
                                                    transport=transport,
                                                    api_key=api_key)
        if name is None:
            name = default_run_name(server_url,
                                    engine_meta.get("model_fingerprint", "?"),
                                    targets)
        run_id = bench.store_benchmark_run(db_path, server_url, engine_meta, points,
                                           engine=None, name=name)
    else:
        try:
            points = _run_openai_bench_points(
                server_url, model, targets, n_predict,
                timeout=timeout, on_point=_collecting_on_point,
                api_key=api_key, streams=streams, power_host=power_host,
                power_gpu_index=power_gpu_index)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            if not partial_points:
                raise DispatchFailure(
                    f"bench failed against {server_url}: {exc} "
                    "— check the server is still up") from exc
            engine_meta = _openai_engine_meta(model)
            run_id = bench.store_benchmark_run(
                db_path, server_url, engine_meta, partial_points,
                engine="openai", status="partial", name=name)
            return {
                "run_id": run_id,
                "status": "partial",
                "error": str(exc),
                "engine_metadata": engine_meta,
                "points": [_point_payload(p) for p in partial_points],
            }
        engine_meta = _openai_engine_meta(model)
        if name is None:
            name = default_run_name(server_url,
                                    engine_meta.get("model_fingerprint", "?"),
                                    targets)
        run_id = bench.store_benchmark_run(db_path, server_url, engine_meta, points,
                                       engine="openai"
                                       if transport_name == "openai" else None,
                                       name=name)
    return {
        "run_id": run_id,
        "status": "complete",
        "engine_metadata": engine_meta,
        "points": [_point_payload(p) for p in points],
    }


def _plan_openai_points(targets: list[int],
                        unit_text: str = bench.DEFAULT_UNIT_TEXT,
                        margin: int = bench.DEFAULT_MARGIN) -> list[bench.PromptPlan]:
    """Cold prompt plans for an OpenAI-compatible target (no /tokenize).

    Mirrors the sizing math of ``bench.plan_prompt_sizes`` — same margin,
    same cold-UUID construction — but derives ``tokens_per_section`` from
    ``_UNIT_TEXT_TOKENS_EST`` instead of the server tokenizer, which these
    servers do not expose (tabbyAPI returns 404 on /tokenize).
    """
    plans: list[bench.PromptPlan] = []
    for target in targets:
        sections = max(1, int((target - margin) / _UNIT_TEXT_TOKENS_EST))
        plans.append(bench.PromptPlan(
            target_tokens=target, sections=sections, unit_text=unit_text))
    return plans


def _openai_point_to_bench_point(
    op: "bench_openai.OpenAIBenchPoint", target_tokens: int,
    power: dict | None = None,
) -> bench.BenchPoint:
    """Map one OpenAI stream point into the native BenchPoint shape.

    Semantics preserved from bench_openai: TTFT = first stream chunk,
    decode = resolved rate (server-authoritative usage timing when the
    final chunk carries it, client chunk-rate fallback otherwise — the
    fallback rejects burst-flushed spans), prefill = prompt_tokens/ttft
    (an estimate, not server-authoritative). MTP fields map from the
    server's speculative-decoding counters when the final usage chunk
    carries them (tabbyAPI: accepted/rejected_prediction_tokens);
    absent counters stay None — honest nulls. Power comes only from an
    explicit opt-in sampler (ADR 002: no watts without a dedicated-host
    sampler); None otherwise.
    """
    power = power or {}
    return bench.BenchPoint(
        target_tokens=target_tokens,
        prefill_tps=op.prefill_tps_estimate,
        decode_tps=op.decode_tps,
        ttft_ms=op.ttft_ms,
        wall_s=op.wall_s,
        output_tokens=op.completion_tokens,
        mtp_draft_n=op.mtp_draft_n,
        mtp_accepted=op.mtp_accepted,
        stop_reason=op.stop_reason,
        prompt_tokens=op.prompt_tokens,
        power_watts=power.get("watts"),
        power_host=power.get("host"),
        power_gpu_index=power.get("gpu_index"),
    )


def _run_openai_bench_points(
    server_url: str,
    model: str,
    targets: list[int],
    n_predict: int,
    timeout: float = 3600.0,
    on_point=None,
    api_key: str | None = None,
    streams: int = 1,
    power_host: str | None = None,
    power_gpu_index: int = 0,
) -> list[bench.BenchPoint]:
    """Sweep one OpenAI-compatible target over the given context lengths.

    Per target: build the same cold UUID-prefixed prompt the native bench
    builds, run ``streams`` identical concurrent SSE workstreams via
    ``bench_openai.run_openai_stream_point`` (client-side metrics), and
    aggregate them into one BenchPoint (mean rates/TTFT, max wall, combined
    = sum across streams). A stream point that reports an error (server
    dropped the connection, HTTP failure inside the stream) fails the
    sweep — dispatch surfaces it as DispatchFailure, same as the native
    path.
    """
    points: list[bench.BenchPoint] = []
    for plan in _plan_openai_points(targets):
        prompt = bench.build_cold_prompt(plan)

        def _one_stream():
            op = bench_openai.run_openai_stream_point(
                server_url, model,
                [{"role": "user", "content": prompt}],
                api_key=api_key, max_tokens=n_predict,
                timeout_s=timeout,
            )
            if op.error:
                raise RuntimeError(f"openai bench point failed at "
                                   f"{plan.target_tokens} tokens: {op.error}")
            return op

        def _one_mapped_stream():
            return _openai_point_to_bench_point(_one_stream(),
                                                plan.target_tokens)

        # Opt-in power sampling (ADR 002): when the dispatcher names a
        # dedicated host, one SSH sampler runs alongside the point and its
        # mean W annotates every stream in the aggregate. (The local is
        # power_info, NOT power — that name is the module import here.)
        power_info = None
        sampler = None
        if power_host:
            sampler = power.PowerSampler(power_host, gpu_index=power_gpu_index)
            sampler.start()
        try:
            point = bench.run_point_streams(_one_mapped_stream, streams)
        finally:
            if sampler is not None:
                watts = sampler.stop()
                power_info = {"watts": watts, "host": power_host,
                              "gpu_index": power_gpu_index}
        if power_info:
            point.power_watts = power_info["watts"]
            point.power_host = power_info["host"]
            point.power_gpu_index = power_info["gpu_index"]
        points.append(point)
        if on_point is not None:
            on_point(point)
    return points


def _openai_engine_meta(model: str) -> dict:
    """Engine metadata for an OpenAI-compatible run.

    These servers have no /props endpoint, so n_ctx/build/system_info stay
    None (the storage layer stores the 0 sentinel for n_ctx). The chat
    model name is the best available fingerprint.
    """
    return {
        "n_ctx": None,
        "model_path": model,
        "build": None,
        "engine": "openai",
        "system_info": None,
        "model_fingerprint": model,
    }


# ----------------------------------------------------- async bench jobs -----
# In-process job registry for the async bench API. Jobs are metadata only
# (status/progress/ETA); every measured value still lands in the append-only
# store via the normal dispatch path, so a dashboard restart loses at most
# the live progress view — never data. Bounded: finished jobs older than
# _JOB_TTL are dropped on each new submission.
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_S = 3600.0


def _prune_jobs() -> None:
    now = time.monotonic()
    for job_id in [j for j, rec in _JOBS.items()
                   if rec["status"] in ("done", "error")
                   and now - rec["finished_at"] > _JOB_TTL_S]:
        _JOBS.pop(job_id, None)


def bench_job_status(job_id: str) -> dict[str, Any] | None:
    """Snapshot of one job record (thread-safe copy).

    ``points`` is a list the worker thread appends to while a poll may be
    serializing it, so the snapshot copies the list and each row dict under
    the lock — a poll can never observe a torn row, and mutating the
    returned snapshot cannot corrupt the live job record.
    """
    with _JOBS_LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return None
        snapshot = dict(rec)
        snapshot["points"] = [dict(p) for p in rec["points"]]
        return snapshot


def _track_point(job: dict[str, Any], point) -> None:
    """on_point hook: record the finished point and recompute the ETA.

    ETA is self-correcting: remaining = elapsed/points_done * points_left,
    recomputed from actuals after every point (no wall-clock guessing).
    """
    job["points_done"] += 1
    job["points"].append({
        "context_tokens": point.target_tokens,
        "prefill_tps": point.prefill_tps,
        "decode_tps": point.decode_tps,
        "prefill_tps_combined": getattr(point, "prefill_tps_combined", None),
        "decode_tps_combined": getattr(point, "decode_tps_combined", None),
        "ttft_ms": point.ttft_ms,
        "wall_s": point.wall_s,
        "output_tokens": point.output_tokens,
        "streams": getattr(point, "streams", 1),
    })
    done = job["points_done"]
    elapsed = time.monotonic() - job["started_at"]
    remaining = job["targets_total"] - done
    job["eta_s"] = round(elapsed / done * remaining, 1) if remaining and done else None
    job["status_line"] = (f"point {done}/{job['targets_total']} complete "
                          f"({point.target_tokens} tokens, {point.wall_s:.1f}s)")


def run_bench_job(
    db_path,
    server_url: str,
    targets: list[int],
    n_predict: int,
    api_key: str | None,
    transport_name: str,
    model: str | None,
    streams: int = 1,
    power_host: str | None = None,
    power_gpu_index: int = 0,
    name: str | None = None,
) -> str:
    """Validate + snapshot a job, spawn the sweep in a daemon thread.

    Returns the job id immediately. Validation errors raise DispatchError
    synchronously (the HTTP layer maps them to 422 before any thread
    starts); everything that happens inside the thread is reported via
    the job record, never raised.
    """
    if transport_name is None:
        transport_name = "native"
    if transport_name not in _BENCH_TRANSPORTS:
        raise DispatchError(
            f"unknown transport {transport_name!r} — expected one of "
            f"{', '.join(_BENCH_TRANSPORTS)}")
    if transport_name == "openai" and not (isinstance(model, str) and model):
        raise DispatchError(
            "transport 'openai' requires 'model' (the chat model name the "
            "target serves)")
    if not targets:
        raise DispatchError("targets list is empty")
    if any(t < 1 for t in targets):
        raise DispatchError("every target must be >= 1 token")
    if n_predict < 1:
        raise DispatchError("n_predict must be >= 1")

    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "preflight",          # preflight | running | done | error
        "server_url": server_url,
        "transport": transport_name,
        "model": model,
        "streams": streams,
        "targets_total": len(targets),
        "points_done": 0,
        "points": [],
        "status_line": "validating + preflight…",
        "eta_s": None,
        "started_at": time.monotonic(),
        "finished_at": None,
        "run_id": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def _worker() -> None:
        try:
            def on_point(point) -> None:
                with _JOBS_LOCK:
                    job["status"] = "running"
                    job["status_line"] = (f"point {job['points_done'] + 1}/"
                                          f"{job['targets_total']} "
                                          f"({point.target_tokens} tokens) streaming…")
                _track_point(job, point)

            result = dispatch_bench(
                db_path=db_path, server_url=server_url, targets=targets,
                n_predict=n_predict, api_key=api_key,
                transport_name=transport_name, model=model,
                on_point=on_point, streams=streams,
                power_host=power_host, power_gpu_index=power_gpu_index,
                name=name)
            with _JOBS_LOCK:
                if result.get("status") == "partial":
                    # Partial-run preservation (2026-09-22): the sweep failed
                    # mid-way but the completed points were stored — surface
                    # the partial run so the UI can link to the data that
                    # DID land.
                    job["status"] = "partial"
                    job["error"] = result.get("error")
                    job["run_id"] = result["run_id"]
                    job["status_line"] = (
                        f"failed after {job['points_done']} point(s) — "
                        f"partial run #{result['run_id']} stored: "
                        f"{result.get('error')}")
                else:
                    job["status"] = "done"
                    job["run_id"] = result["run_id"]
                    job["status_line"] = (f"run #{result['run_id']} stored "
                                          f"({job['targets_total']} points)")
                job["finished_at"] = time.monotonic()
                job["eta_s"] = 0.0 if result.get("status") != "partial" else None
        except Exception as exc:  # noqa: BLE001 — the thread IS the report
            with _JOBS_LOCK:
                job["status"] = "error"
                job["error"] = str(exc)
                job["status_line"] = f"failed: {exc}"
                job["finished_at"] = time.monotonic()
                job["eta_s"] = None

    threading.Thread(target=_worker, name=f"bench-job-{job_id}",
                     daemon=True).start()
    return job_id


def _preflight_detail(target: str, pf) -> str:
    """Actionable preflight failure message: names failing checks, states
    the target, and points at the next step (per UX review finding 3)."""
    failed = [c["name"] for c in pf["checks"] if not c["ok"]]
    details = [c["detail"] for c in pf["checks"] if not c["ok"]]
    return ("preflight failed (" + ", ".join(failed) + ") at " + target + ": " +
            "; ".join(details) +
            " — verify the server is running or correct the URL; "
            "use ✔ Check target for per-check detail")


def dispatch_eval(
    db_path,
    suite_id: int,
    target: str,
    model_fingerprint: str,
    suite: dict[str, Any],
    api_key: str | None = None,
    name: str | None = None,
) -> dict:
    """Replay one eval suite against a target; store eval_results per item.

    Single-turn items replay as one POST; multi-turn items replay via
    multiturn.replay_multiturn (one stored row per assistant turn).
    ``model_fingerprint`` is passed to the target as the chat model name —
    proxy-fronted endpoints (litellm/vLLM) need the real name; native
    llama.cpp servers ignore it.

    ``name`` (optional run label, 2026-09-22): names the eval run shown
    on /evals and reports; None stores NULL and the UI falls back to the
    default 'suite · model · N items' rendering.
    """
    items = suite.get("items")
    if not isinstance(items, list) or not items:
        raise DispatchError("suite has no items to replay")
    # cheap shape validation BEFORE preflight: bad input needs no network
    for item in items:
        if item.get("type") == "multi-turn":
            mt.validate_multiturn_item(item)
        else:
            item_id = item.get("id") or item.get("item_id")
            if not item_id:
                raise DispatchError(f"item missing id: {item!r}")
            if not (item.get("prompt") or item.get("content")):
                raise DispatchError(f"item {item_id!r} missing prompt")

    from arcturos import db as dbmod
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    if conn.execute("SELECT 1 FROM eval_suites WHERE id = ?", (suite_id,)).fetchone() is None:
        conn.close()
        raise DispatchError(f"eval suite {suite_id} not found (create it first)")
    conn.close()
    pf = preflight.preflight(target, kind="eval", model=model_fingerprint,
                             api_key=api_key)
    if not pf.ok():
        raise DispatchError(
            _preflight_detail(target, pf))
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    stored: list[dict] = []
    try:
        # Eval-run entity (2026-09-22): one replay = one eval_runs row; the
        # items stamp eval_run_id so results group per run. Insert happens
        # before replay: a mid-replay failure still leaves the (partial)
        # run row + whatever items completed — consistent with bench's
        # partial-run preservation.
        now = bench._utcnow()
        cur = conn.execute(
            "INSERT INTO eval_runs (suite_id, model_fingerprint, name, created_at)"
            " VALUES (?, ?, ?, ?)",
            (suite_id, model_fingerprint, name, now))
        eval_run_id = int(cur.lastrowid)
        for item in items:
            if item.get("type") == "multi-turn":
                records = mt.replay_multiturn(target, item, api_key=api_key,
                                              model=model_fingerprint)
                for rec in records:
                    row = _store_eval_result(
                        conn, suite_id, model_fingerprint, rec["item_id"],
                        rec["output"], rec["prompt_tokens"],
                        rec["completion_tokens"], rec["latency_ms"],
                        eval_run_id=eval_run_id)
                    stored.append(row)
            else:
                item_id = item.get("id") or item.get("item_id")
                if not item_id:
                    raise DispatchError(f"item missing id: {item!r}")
                prompt = item.get("prompt") or item.get("content")
                if not prompt:
                    raise DispatchError(f"item {item_id!r} missing prompt")
                t0 = time.monotonic()
                resp = mt.post_openai_chat(
                    target, [{"role": "user", "content": prompt}],
                    api_key=api_key, model=model_fingerprint)
                latency_ms = round((time.monotonic() - t0) * 1000, 1)
                content, ptoks, ctoks = mt.extract_output(resp)
                stored.append(_store_eval_result(
                    conn, suite_id, model_fingerprint, item_id,
                    content, ptoks, ctoks, latency_ms,
                    eval_run_id=eval_run_id))
        conn.commit()
    except (RuntimeError, ValueError) as exc:
        conn.rollback()
        raise DispatchFailure(
            f"eval replay failed against {target}: {exc} — check the target "
            "is still up / returns valid chat responses") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"suite_id": suite_id, "eval_run_id": eval_run_id,
            "stored": stored}


def _store_eval_result(conn, suite_id: int, model_fingerprint: str,
                       item_id: str, output: str, prompt_tokens: int,
                       completion_tokens: int, latency_ms: float,
                       eval_run_id: int | None = None) -> dict:
    # Registry auto-registration on first sighting (§4.6): eval replays
    # register their model even without a bench ever touching it.
    now = bench._utcnow()  # one clock read: registry provenance == row timestamp
    conn.execute(
        "INSERT INTO models (model_fingerprint, alias, engine, "
        "first_seen, last_seen) VALUES (?, NULL, NULL, ?, ?) "
        "ON CONFLICT(model_fingerprint) DO UPDATE SET last_seen = ?",
        (model_fingerprint, now, now, now),
    )
    cur = conn.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, output,"
        " prompt_tokens, completion_tokens, latency_ms, eval_run_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (suite_id, model_fingerprint, item_id, output,
         prompt_tokens, completion_tokens, latency_ms, eval_run_id, now))
    row = dict(conn.execute(
        "SELECT * FROM eval_results WHERE id = ?", (cur.lastrowid,)).fetchone())
    return row
