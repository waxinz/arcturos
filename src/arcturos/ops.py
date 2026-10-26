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
from arcturos import evals
from arcturos import power
from arcturos import multiturn as mt
from arcturos import preflight

import httpx
from urllib.error import URLError

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


class SweepAborted(RuntimeError):
    """Raised between points when the operator stops a faulty sweep.

    Subclasses RuntimeError so every existing failure net (dispatch's
    partial-run preservation, the async job's error path) handles it
    unchanged: points completed before the abort are kept as a partial
    run, points after it never start.
    """


class AbortFlag:
    """Thread-safe cooperative-abort switch for a bench sweep.

    The async job owns one; the UI's Stop button flips it via
    ``ops.abort_bench_job``. ``check()`` raises SweepAborted — cheap to
    call between points and inside streaming loops, never swallowed.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise SweepAborted("sweep aborted by operator (stop requested)")


def default_run_name(server_url: str, model_fingerprint: str,
                     targets: list[int], engine: str | None = None) -> str:
    """Humanized default run name (2026-09-22 naming feature).

    Owner chain (2026-10-11): when the operator leaves the run name
    blank AND an engine label was given, the name is just
    'Engine / Model' (owner: drop the targets part); without an engine
    (or with an unknown fingerprint) it falls back to the original
    'host · model · 64k/128k'. Model is rendered short (registry alias
    rendering happens at display time); the host fallback keeps
    humanized targets (32768 -> 64k, 1500 -> 1500).
    """
    host = compare.host_label(server_url)
    humanized = "/".join(
        f"{t // 1024}k" if t >= 1024 and t % 1024 == 0 else str(t)
        for t in targets)
    short_model = compare.short_fingerprint(model_fingerprint)
    if engine and short_model and short_model != "?":
        return f"{engine} / {short_model}"
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
    engine: str | None = None,
    streams: int = 1,
    power_host: str | None = None,
    power_gpu_index: int = 0,
    name: str | None = None,
    abort: "AbortFlag | None" = None,
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
    if transport_name == "auto":
        # UI autodetect (2026-10-11): the operator no longer picks a
        # transport; probe the target (openai first — the usual case —
        # native fallback) and dispatch on what answers.
        transport_name = preflight.detect_transport(
            server_url, model=model, transport=transport, api_key=api_key)
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
                power_host=power_host, power_gpu_index=power_gpu_index,
                abort=abort)
        except SweepAborted as exc:
            return _aborted_result(db_path, server_url, targets,
                                   partial_points, exc, engine="native",
                                   name=name, transport=transport,
                                   api_key=api_key, engine_label=engine)
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
            # Name chain (owner 2026-10-11): operator engine given ->
            # 'Engine / Model / targets'; engine omitted (bare API call)
            # keeps the classic 'host · model · targets'. Storage still
            # records the llama.cpp default engine for the run row.
            name = default_run_name(
                server_url, engine_meta.get("model_fingerprint", "?"),
                targets, engine=engine)
        run_id = bench.store_benchmark_run(db_path, server_url, engine_meta, points,
                                           engine=engine or "llama.cpp",
                                           name=name)
    else:
        try:
            points = _run_openai_bench_points(
                server_url, model, targets, n_predict,
                timeout=timeout, on_point=_collecting_on_point,
                api_key=api_key, streams=streams, power_host=power_host,
                power_gpu_index=power_gpu_index, abort=abort)
        except SweepAborted as exc:
            return _aborted_result(db_path, server_url, targets,
                                   partial_points, exc, engine="openai",
                                   name=name, model=model,
                                   engine_label=engine)
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
            # Same chain as the native branch: engine given -> 'Engine /
            # Model / targets', else classic host fallback. Storage
            # records the openai default engine.
            name = default_run_name(
                server_url, engine_meta.get("model_fingerprint", "?"),
                targets, engine=engine)
        run_id = bench.store_benchmark_run(db_path, server_url, engine_meta, points,
                                       engine=engine or "openai",
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
    abort: "AbortFlag | None" = None,
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
        # Cooperative abort (ADR 006): a stop request raised between points
        # leaves the sweep via SweepAborted — the dispatch failure net keeps
        # the points already stored as a partial run.
        if abort is not None:
            abort.check()

        def _one_stream():
            # Fresh prompt PER STREAM (ADR 004). The OpenAI API exposes no
            # cache_prompt:false, so the UUID prefix is the only defence
            # against prefix caching; building it once per point and
            # sharing it made streams 2..N 100% cache hits whose TTFT
            # was queue time, not prefill — the combined-throughput row
            # then credited cached tokens as if prefilled (run #103:
            # Σ prefill 1222 T/s on a server that really did ~640).
            prompt = bench.build_cold_prompt(plan)
            op = bench_openai.run_openai_stream_point(
                server_url, model,
                [{"role": "user", "content": prompt}],
                api_key=api_key, max_tokens=n_predict,
                timeout_s=timeout, abort=abort)
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


def _aborted_result(db_path, server_url: str, targets: list[int],
                    partial_points: list, exc: "SweepAborted",
                    engine: str, name: str | None,
                    model: str | None = None,
                    transport=None, api_key: str | None = None,
                    engine_label: str | None = None) -> dict:
    """Store an operator-stopped sweep as a partial run (ADR 006).

    Same shape as the failure partial path, with an honest status line:
    the stop was commanded, not a server fault. Zero completed points ->
    nothing is stored and DispatchFailure propagates (contract: no empty
    run rows). ``engine_label`` carries the operator's freeform engine
    tag (2026-10-11) so a stopped run records the same engine as a
    complete one would.
    """
    if not partial_points:
        raise DispatchFailure(
            f"sweep aborted before any point completed: {exc}") from exc
    if engine == "native":
        engine_meta = bench.capture_engine_metadata(
            server_url, transport=transport, api_key=api_key)
    else:
        engine_meta = _openai_engine_meta(model)
    run_id = bench.store_benchmark_run(
        db_path, server_url, engine_meta, partial_points,
        engine=engine_label or (None if engine == "native" else "openai"),
        status="partial", name=name)
    return {
        "run_id": run_id,
        "status": "aborted",
        "error": str(exc),
        "engine_metadata": engine_meta,
        "points": [_point_payload(p) for p in partial_points],
    }


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
# Abort switches for RUNNING jobs (ADR 006): job_id -> AbortFlag. Kept out
# of the job record (a live threading.Event would leak into the JSON
# snapshot); popped when the abort fires or the job is pruned.
_ABORT_FLAGS: dict[str, "AbortFlag"] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_S = 3600.0


def abort_bench_job(job_id: str) -> dict[str, Any] | None:
    """Request a cooperative abort of a running bench job (faulty-run stop).

    Sets the flag the sweep checks between points (and inside streaming
    decode loops): the CURRENT point is abandoned, points after it never
    start, and the partial-run preservation path stores whatever completed
    — an aborted run keeps its data, like any other mid-sweep failure.
    Returns a snapshot for the API response, or None for an unknown job.
    Aborting an already-finished job is a no-op (the flag just stays set).
    """
    with _JOBS_LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return None
        rec["abort_requested"] = True
        flag = _ABORT_FLAGS.pop(job_id, None)
    if flag is not None:
        flag.request()
    return {
        "job_id": job_id,
        "status": rec["status"],
        "abort_requested": True,
    }


def _prune_jobs() -> None:
    now = time.monotonic()
    for job_id in [j for j, rec in _JOBS.items()
                   if rec["status"] in ("done", "error", "partial", "aborted")
                   and now - rec["finished_at"] > _JOB_TTL_S]:
        _JOBS.pop(job_id, None)
        _ABORT_FLAGS.pop(job_id, None)


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
    engine: str | None = None,
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
    if transport_name not in _BENCH_TRANSPORTS + ("auto",):
        raise DispatchError(
            f"unknown transport {transport_name!r} — expected one of "
            f"{', '.join(_BENCH_TRANSPORTS)}, auto")
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
    abort_flag = AbortFlag()
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
        "abort_requested": False,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _ABORT_FLAGS[job_id] = abort_flag

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
                transport_name=transport_name, model=model, engine=engine,
                on_point=on_point, streams=streams,
                power_host=power_host, power_gpu_index=power_gpu_index,
                name=name, abort=abort_flag)
            with _JOBS_LOCK:
                if result.get("status") == "aborted":
                    # Operator stop (ADR 006): completed points were kept
                    # as a partial run — same reporting as a failure, with
                    # an honest "stopped" line instead of a server error.
                    job["status"] = "aborted"
                    job["run_id"] = result["run_id"]
                    job["status_line"] = (
                        f"stopped by operator after "
                        f"{job['points_done']} point(s) — partial run "
                        f"#{result['run_id']} stored")
                elif result.get("status") == "partial":
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
    max_tokens: int | None = None,
    temperature: float = 0.0,
    retries: int = 1,
    concurrency: int = 1,
    on_result=None,
    abort: "AbortFlag | None" = None,
    reasoning_effort: str | None = None,
    reasoning_budget: int | None = None,
) -> dict:
    """Replay one eval suite against a target; store eval_results per item.

    Single-turn items replay as one POST (honoring the item's ``system``
    prompt and ``image`` — v0.2); multi-turn items replay via
    multiturn.replay_multiturn (one stored row per assistant turn).
    ``model_fingerprint`` is passed to the target as the chat model name —
    proxy-fronted endpoints (litellm/vLLM) need the real name; native
    llama.cpp servers ignore it.

    v0.2 parameters (ADR 007 §5): ``temperature`` default 0.0,
    ``concurrency`` across items (default 1, max 8). Items are committed
    PER ITEM — a mid-replay failure keeps completed items and marks the
    eval_run partial (partial preservation, the bench pattern).
    ``abort`` (ops.AbortFlag) stops between items.

    ``retries`` per item (2026-10-26, default 1 — was 0): a transient
    request failure (a shared inference server stalls under co-tenant
    load; run #17 lost 4 of 118 items to four >300s read-timeouts and
    finished partial) retries the WHOLE item from its first turn before
    the failure is recorded. Judge jobs already defaulted eval_retries
    to 1; explicit 0 keeps the strict one-shot behavior.

    ``max_tokens`` (2026-10-25): None — the default — omits the field from
    every item's wire body so the SERVER applies its own budget. The old
    1024 default shipped degraded results on the Maya/DSF reasoning server
    (max_tokens <= 1024 strangles its thinking phase: un-thought wrong
    answers + think-tail leaking into content — runs 14/15). A caller-set
    value (1..32768) is passed through for servers that need a cap.

    ``name`` (optional run label, 2026-09-22): names the eval run shown
    on /evals and reports; None stores NULL and the UI falls back to the
    the default 'suite · model · N items' rendering.

    ``reasoning_effort``/``reasoning_budget`` (2026-10-15, optional):
    per-MODEL reasoning depth sent with every item's chat call —
    'minimal'|'low'|'medium'|'high'|'xhigh'|'max' and a max-thinking-tokens
    int.
    Optional: omitted from the wire body when unset.
    """
    try:
        items = evals.validate_suite(suite)
    except evals.ItemValidationError as exc:
        # cheap validation keeps the historical DispatchError contract
        # (main.py + callers match on it); no network has been touched.
        raise DispatchError(str(exc)) from exc
    if not items:
        raise DispatchError("suite has no items to replay")
    if max_tokens is not None and (not isinstance(max_tokens, int)
                                   or isinstance(max_tokens, bool)
                                   or not (1 <= max_tokens <= 32768)):
        raise DispatchError("max_tokens must be an int between 1 and 32768")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not (0.0 <= temperature <= 2.0):
        raise DispatchError("temperature must be a number between 0 and 2")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0 \
            or retries > 3:
        raise DispatchError("retries must be an int between 0 and 3")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) \
            or not (1 <= concurrency <= 8):
        raise DispatchError("concurrency must be an int between 1 and 8")
    try:
        mt.validate_reasoning(reasoning_effort, reasoning_budget)
    except ValueError as exc:
        raise DispatchError(str(exc)) from exc

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
    item_errors: list[str] = []
    eval_run_id: int | None = None
    try:
        # Eval-run entity (2026-09-22): one replay = one eval_runs row; the
        # items stamp eval_run_id so results group per run. Insert happens
        # before replay; items commit PER ITEM (v0.2) — a mid-replay failure
        # still leaves the run row + whatever items completed (partial
        # preservation), and the run is marked partial in the except net.
        now = bench._utcnow()
        cur = conn.execute(
            "INSERT INTO eval_runs (suite_id, model_fingerprint, name, created_at)"
            " VALUES (?, ?, ?, ?)",
            (suite_id, model_fingerprint, name, now))
        eval_run_id = int(cur.lastrowid)
        conn.commit()

        def _replay_one(item: dict[str, Any]) -> list[dict[str, Any]]:
            """Replay one item with the retry policy; raises after budget."""
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    if item.get("type") == "multi-turn":
                        return list(mt.replay_multiturn(
                            target, item, api_key=api_key,
                            model=model_fingerprint,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            reasoning_budget=reasoning_budget))
                    rec = evals.replay_single_turn(
                        target, item, api_key=api_key,
                        model=model_fingerprint, max_tokens=max_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        reasoning_budget=reasoning_budget)
                    return [rec]
                except (httpx.HTTPError, URLError, TimeoutError,
                        RuntimeError, ValueError, KeyError) as exc:
                    last_exc = exc
            raise last_exc  # type: ignore[misc]

        def _store_and_track(rec: dict[str, Any]) -> dict:
            row = _store_eval_result(
                conn, suite_id, model_fingerprint, rec["item_id"],
                rec["output"], rec["prompt_tokens"],
                rec["completion_tokens"], rec["latency_ms"],
                eval_run_id=eval_run_id)
            conn.commit()  # per-item commit: partial preservation (v0.2)
            stored.append(row)
            if on_result is not None:
                on_result(rec["item_id"], row)
            return row

        def _work_items() -> list[tuple[dict[str, Any], Any]]:
            """(item, done-marker) pairs in stable suite order."""
            return [(item, None) for item in items]

        done_count = 0

        def _track(done_label: str) -> None:
            nonlocal done_count
            done_count += 1
            if abort is not None:
                abort.check()

        if concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_replay_one, item): item
                           for item, _ in _work_items()}
                for future, item in futures.items():
                    if abort is not None:
                        abort.check()
                    try:
                        records = future.result()
                    except SweepAborted:
                        raise
                    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                        item_errors.append(
                            f"{item.get('id', '?')}: {exc}")
                        continue
                    for rec in records:
                        _store_and_track(rec)
                    _track(item.get("id", "?"))
        else:
            for item, _ in _work_items():
                if abort is not None:
                    abort.check()
                try:
                    records = _replay_one(item)
                except SweepAborted:
                    raise
                except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                    item_errors.append(f"{item.get('id', '?')}: {exc}")
                    continue
                for rec in records:
                    _store_and_track(rec)
                _track(item.get("id", "?"))
        # eval_run status: partial when any item failed or an abort landed;
        # complete when every item stored. (runs.status doctrine — ADR 007 §5)
        if item_errors or (abort is not None and abort.requested):
            conn.execute("UPDATE eval_runs SET status = 'partial' WHERE id = ?",
                         (eval_run_id,))
        conn.commit()
    except SweepAborted as exc:
        _mark_partial(conn, eval_run_id)
        conn.commit()
        if not stored:
            conn.close()
            raise DispatchFailure(
                f"eval replay aborted before any item completed: {exc}") from exc
        conn.close()
        return {"suite_id": suite_id, "eval_run_id": eval_run_id,
                "status": "aborted", "stored": stored,
                "item_errors": item_errors, "error": str(exc)}
    except (RuntimeError, ValueError) as exc:
        _mark_partial(conn, eval_run_id)
        conn.commit()
        conn.close()
        if not stored:
            raise DispatchFailure(
                f"eval replay failed against {target}: {exc} — check the target "
                "is still up / returns valid chat responses") from exc
        return {"suite_id": suite_id, "eval_run_id": eval_run_id,
                "status": "partial", "stored": stored,
                "item_errors": item_errors, "error": str(exc)}
    except Exception:
        _mark_partial(conn, eval_run_id)
        conn.commit()
        conn.close()
        raise
    conn.close()
    status = "partial" if item_errors else "complete"
    return {"suite_id": suite_id, "eval_run_id": eval_run_id,
            "status": status, "stored": stored, "item_errors": item_errors}


def _mark_partial(conn, eval_run_id: int | None) -> None:
    """Stamp an eval_run partial (the trigger whitelists name + status)."""
    if not eval_run_id:
        return
    try:
        conn.execute("UPDATE eval_runs SET status = 'partial' WHERE id = ?",
                     (eval_run_id,))
    except Exception:
        pass  # pre-migration DB without the status column — best effort


# ----------------------------------------------------- async eval jobs -----
# Same in-process job pattern as bench jobs (ADR 007 §5): metadata + live
# progress only; every measurement lands in the append-only store through
# the normal dispatch path. A dashboard restart loses the live progress
# view, never data.
_EVAL_JOBS: dict[str, dict[str, Any]] = {}
_EVAL_ABORT_FLAGS: dict[str, "AbortFlag"] = {}
_EVAL_JOBS_LOCK = threading.Lock()


def abort_eval_job(job_id: str) -> dict[str, Any] | None:
    """Request a cooperative stop of a running eval replay (ADR 006 pattern).

    The current item finishes or is abandoned per its retry budget; items
    after the flag never start; completed items stay stored (partial
    preservation). Aborting a finished job is a no-op.
    """
    with _EVAL_JOBS_LOCK:
        rec = _EVAL_JOBS.get(job_id)
        if rec is None:
            return None
        rec["abort_requested"] = True
        flag = _EVAL_ABORT_FLAGS.pop(job_id, None)
    if flag is not None:
        flag.request()
    return {"job_id": job_id, "status": rec["status"], "abort_requested": True}


def eval_job_status(job_id: str) -> dict[str, Any] | None:
    """Thread-safe snapshot of one eval job record."""
    with _EVAL_JOBS_LOCK:
        rec = _EVAL_JOBS.get(job_id)
        if rec is None:
            return None
        snapshot = dict(rec)
        snapshot["items"] = [dict(p) for p in rec["items"]]
    # computed at read time (like bench jobs) — never stored on the record
    if snapshot.get("finished_at") is not None:
        snapshot["elapsed_s"] = round(
            snapshot["finished_at"] - snapshot["started_at"], 1)
    else:
        snapshot["elapsed_s"] = round(
            time.monotonic() - snapshot["started_at"], 1)
    return snapshot


def _prune_eval_jobs() -> None:
    now = time.monotonic()
    for job_id in [j for j, rec in _EVAL_JOBS.items()
                   if rec["status"] in ("done", "error", "partial", "aborted")
                   and now - rec["finished_at"] > _JOB_TTL_S]:
        _EVAL_JOBS.pop(job_id, None)
        _EVAL_ABORT_FLAGS.pop(job_id, None)


def _expected_rows(items: list[dict[str, Any]]) -> int:
    """Stored-row count a full replay will produce.

    Single-turn items store one eval_result; multi-turn items store one
    row per assistant turn (v0.2 multiturn contract). Job progress counts
    stored rows, so the progress denominator must too — ``len(items)``
    undercounts multi-turn suites (smart-agent-101: 480 definition items
    but 376 + 211 assistant turns = 587 stored rows, which rendered as
    "item 553/480" in the live progress view).
    """
    n = 0
    for it in items:
        turns = it.get("turns")
        if it.get("type") == "multi-turn" and isinstance(turns, list) \
                and turns:
            n += sum(1 for t in turns if isinstance(t, dict)
                     and t.get("role") == "assistant")
        else:
            n += 1
    return n


def run_eval_job(
    db_path,
    suite_id: int,
    target: str,
    model_fingerprint: str,
    suite: dict[str, Any],
    api_key: str | None = None,
    name: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    retries: int = 1,
    concurrency: int = 1,
    reasoning_effort: str | None = None,
    reasoning_budget: int | None = None,
) -> str:
    """Validate + snapshot an eval job, replay in a daemon thread.

    Returns the job id immediately. Validation errors raise DispatchError
    synchronously (422 before any thread starts); everything inside the
    thread is reported via the job record.

    ``max_tokens`` (2026-10-25): None — the default — forwards omission to
    dispatch_eval (the field is left OFF the wire body; server budget).
    ``retries`` (2026-10-26): default 1 — forwarded to dispatch_eval
    (transient-stall resilience on shared inference servers; see there).
    ``reasoning_effort``/``reasoning_budget`` (2026-10-15, optional):
    per-model reasoning depth forwarded to dispatch_eval — validated here
    too so a bad value 422s before the thread starts.
    """
    try:
        items = evals.validate_suite(suite)
    except evals.ItemValidationError as exc:
        raise DispatchError(str(exc)) from exc
    if not items:
        raise DispatchError("suite has no items to replay")
    mt.validate_reasoning(reasoning_effort, reasoning_budget)
    _prune_eval_jobs()
    job_id = uuid.uuid4().hex[:12]
    flag = AbortFlag()
    job: dict[str, Any] = {
        "job_id": job_id,
        "kind": "eval",
        "status": "preflight",         # preflight | running | done | error | partial | aborted
        "suite_id": suite_id,
        "target": target,
        "model": model_fingerprint,
        "items_total": _expected_rows(items),
        "items_done": 0,
        "results_stored": 0,
        "items": [],                   # per-item progress rows
        "status_line": "validating + preflight…",
        "eta_s": None,
        "started_at": time.monotonic(),
        "finished_at": None,
        "eval_run_id": None,
        "error": None,
        "item_errors": [],
        "abort_requested": False,
    }
    with _EVAL_JOBS_LOCK:
        _EVAL_JOBS[job_id] = job
        _EVAL_ABORT_FLAGS[job_id] = flag

    def _worker() -> None:
        try:
            def on_result(item_id: str, row: dict) -> None:
                with _EVAL_JOBS_LOCK:
                    job["status"] = "running"
                    job["items_done"] += 1
                    job["results_stored"] += 1
                    job["items"].append({
                        "item_id": item_id, "result_id": row["id"],
                        "latency_ms": row["latency_ms"],
                        "completion_tokens": row["completion_tokens"],
                    })
                    elapsed = time.monotonic() - job["started_at"]
                    remaining = job["items_total"] - job["items_done"]
                    job["eta_s"] = (
                        round(elapsed / job["items_done"] * remaining, 1)
                        if job["items_done"] and remaining > 0 else None)
                    job["status_line"] = (
                        f"stored {job['items_done']}/{job['items_total']}"
                        f" — {item_id}")

            result = dispatch_eval(
                db_path=db_path, suite_id=suite_id, target=target,
                model_fingerprint=model_fingerprint, suite=suite,
                api_key=api_key, name=name, max_tokens=max_tokens,
                temperature=temperature, retries=retries,
                concurrency=concurrency, on_result=on_result, abort=flag,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget)
            with _EVAL_JOBS_LOCK:
                job["eval_run_id"] = result.get("eval_run_id")
                job["item_errors"] = result.get("item_errors", [])
                status = result.get("status", "complete")
                if status == "aborted":
                    job["status"] = "aborted"
                    job["status_line"] = (
                        f"stopped by operator after {job['items_done']} result(s)"
                        f" — eval run #{result.get('eval_run_id')} (partial) stored")
                elif status == "partial":
                    job["status"] = "partial"
                    job["error"] = result.get("error")
                    job["status_line"] = (
                        f"failed after {job['items_done']} result(s) — eval run "
                        f"#{result.get('eval_run_id')} (partial) stored")
                else:
                    job["status"] = "done"
                    job["status_line"] = (
                        f"eval run #{result.get('eval_run_id')} stored "
                        f"({job['items_total']} results from {len(items)} items)")
                job["finished_at"] = time.monotonic()
                job["eta_s"] = 0.0 if status == "complete" else None
        except DispatchFailure as exc:
            with _EVAL_JOBS_LOCK:
                job["status"] = "error"
                job["error"] = str(exc)
                job["status_line"] = f"failed: {exc}"
                job["finished_at"] = time.monotonic()
                job["eta_s"] = None
        except Exception as exc:  # noqa: BLE001 — the thread IS the report
            with _EVAL_JOBS_LOCK:
                job["status"] = "error"
                job["error"] = str(exc)
                job["status_line"] = f"failed: {exc}"
                job["finished_at"] = time.monotonic()
                job["eta_s"] = None

    threading.Thread(target=_worker, name=f"eval-job-{job_id}",
                     daemon=True).start()
    return job_id


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
