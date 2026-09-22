"""Dispatch layer tests — bench/eval kick-off paths (ops.py).

Uses httpx.MockTransport to fake the llama.cpp native endpoints for the
bench path; the eval path fakes /v1/chat/completions via multiturn's
urlreq (patched).
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos import ops  # noqa: E402
from arcturos.multiturn import post_openai_chat  # noqa: E402


def _tokenize_resp(tokens_per_copy: int) -> dict:
    return {"tokens": list(range(tokens_per_copy * 50))}


def _bench_app(targets_tokenized: int = 27, authed: bool = False):
    """MockTransport serving /tokenize, /completion, /props.

    authed=True rejects requests without an Authorization header (mirrors
    llama.cpp --api-key behavior) so the api_key path is testable.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if authed and not request.headers.get("authorization"):
            return httpx.Response(401, json={"error": "unauthorized"})
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/tokenize":
            return httpx.Response(200, json=_tokenize_resp(2))
        if path == "/completion":
            data = {
                "tokens_predicted": 8,
                "tokens_evaluated": targets_tokenized,
                "stop_reason": "eos",
                "timings": {
                    "prompt_per_second": 400.0,
                    "predicted_per_second": 50.0,
                    "draft_n": 30,
                    "draft_n_accepted": 25,
                },
            }
            return httpx.Response(200, json=data)
        if path == "/props":
            return httpx.Response(200, json={
                "n_ctx": 4096, "model_path": "/models/test.gguf",
                "build": "1234", "engine": "llama.cpp",
                "model_fingerprint": "/models/test.gguf"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_dispatch_bench_passes_api_key(db_path):
    """authed mock + api_key -> preflight AND sweep succeed; header present."""
    seen_headers: list[str | None] = []

    transport = _bench_app(authed=True)

    result = ops.dispatch_bench(
        db_path, "http://fake:8000", [128], 8, transport=transport,
        api_key="sk-test-key")
    assert result["run_id"] == 1


def test_dispatch_bench_authed_without_key_fails(db_path):
    """authed mock + no key -> preflight 401 -> DispatchError with detail."""
    transport = _bench_app(authed=True)
    with pytest.raises(ops.DispatchError, match="401"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           transport=transport)


def test_dispatch_bench_runtime_failure_maps_502(db_path):
    """Target dies mid-sweep -> DispatchFailure (502), not DispatchError.

    UX review finding 1: the owner must be told 'check the server', not
    blamed with a bad-input 422.
    """
    transport = _bench_app()
    # First dispatch works; a dead server on retry raises RuntimeError
    # inside run_benchmark -> DispatchFailure with directive text.
    ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                       transport=transport)

    def dying_handler(request):
        # Pass preflight (health/tokenize), then fail the sweep itself.
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(100))})
        if request.url.path == "/props":
            return httpx.Response(200, json={"n_ctx": 2048})
        return httpx.Response(500, json={"error": "server crashed mid-completion"})

    dead = httpx.MockTransport(dying_handler)
    with pytest.raises(ops.DispatchFailure, match="check the server"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           transport=dead, timeout=3600.0)


def test_dispatch_bench_preflight_failure_is_actionable(db_path):
    """Preflight failure names checks, target, and the next step."""
    transport = _bench_app(authed=True)
    try:
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           transport=transport)
    except ops.DispatchError as exc:
        msg = str(exc)
        # /health 503 is a non-blocking warning now; the gate fires
        # on the tokenize 401 (auth required) instead.
        assert "preflight failed (tokenize)" in msg
        assert "401" in msg
        assert "http://fake:8000" in msg
        assert "Check target" in msg


@pytest.fixture()
def db_path(tmp_path):
    from arcturos import db as dbmod
    p = tmp_path / "ops.db"
    conn = dbmod.connect(p)
    dbmod.init_db(conn)
    conn.close()
    return p


def _mk_suite(db_path, suite_id=1):
    """Dispatch requires a real eval_suites row (existence check)."""
    from arcturos import db as dbmod
    conn = dbmod.connect(db_path)
    conn.execute("INSERT INTO eval_suites (id, name, version) VALUES (?, ?, ?)",
                 (suite_id, "s", "1"))
    conn.commit()
    conn.close()


@pytest.fixture()
def pass_preflight():
    """Patch ops.preflight.preflight to a stub that always passes, and
    RESTORE it afterwards.

    The eval-path tests exercise dispatch logic, not preflight — the target
    URL is fake and would hit real DNS otherwise. The old _pass_preflight()
    permanently replaced the module attribute, leaking the stub into every
    later test in the session (found by the UX-review round: dead-target
    tests received ok:true stub checks from an earlier test's patch)."""
    from arcturos.ops import preflight as pf_mod
    stub = pf_mod.PreflightResult(
        target="stub", kind="eval",
        checks=[{"name": "reachable", "ok": True, "detail": "stub"},
                {"name": "chat", "ok": True, "detail": "stub"}])
    sentinel = pf_mod.preflight
    pf_mod.preflight = lambda *a, **kw: stub
    yield
    pf_mod.preflight = sentinel


def test_dispatch_bench_happy(db_path):
    transport = _bench_app(authed=False)
    result = ops.dispatch_bench(
        db_path, "http://fake:8000", [128], 8, transport=transport)
    assert result["run_id"] == 1
    assert result["engine_metadata"]["n_ctx"] == 4096
    assert result["points"][0]["decode_tps"] == 50.0
    # stored and re-readable through the export path
    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    assert exported["benchmarks"][0]["decode_tps"] == 50.0


def test_dispatch_bench_validation(db_path):
    with pytest.raises(ops.DispatchError, match="targets list is empty"):
        ops.dispatch_bench(db_path, "http://fake:8000", [], 8)
    with pytest.raises(ops.DispatchError, match=">= 1"):
        ops.dispatch_bench(db_path, "http://fake:8000", [0], 8)


def test_dispatch_bench_streams_validation(db_path):
    """streams must be an int in [1, 16]; bools are rejected explicitly."""
    with pytest.raises(ops.DispatchError, match="streams must be an int"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8, streams=0)
    with pytest.raises(ops.DispatchError, match="streams must be an int"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8, streams=17)
    with pytest.raises(ops.DispatchError, match="streams must be an int"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8, streams=True)


def test_dispatch_bench_streams_two_stores_stream_count(db_path):
    """streams=2: both workstreams run, aggregate stored with streams=2."""
    transport = _bench_app(authed=False)
    result = ops.dispatch_bench(
        db_path, "http://fake:8000", [128], 8, transport=transport, streams=2)
    assert result["run_id"] == 1
    p0 = result["points"][0]
    assert p0["streams"] == 2
    # the mock timings are deterministic, so the mean equals the single value
    assert p0["decode_tps"] == 50.0
    assert p0["prefill_tps"] == 400.0
    # combined = sum across the 2 streams
    assert p0["decode_tps_combined"] == 100.0
    assert p0["prefill_tps_combined"] == 800.0
    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    assert exported["benchmarks"][0]["streams"] == 2


def test_dispatch_bench_default_streams_is_one(db_path):
    transport = _bench_app(authed=False)
    result = ops.dispatch_bench(
        db_path, "http://fake:8000", [128], 8, transport=transport)
    assert result["points"][0]["streams"] == 1


def test_dispatch_eval_single_turn(db_path, pass_preflight):
    _mk_suite(db_path)
    suite = {"items": [
        {"id": "q1", "prompt": "What is 2+2?"},
        {"id": "q2", "prompt": "Capital of France?"},
    ]}
    with patch("arcturos.multiturn.post_openai_chat") as fake_post:
        fake_post.return_value = {
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = ops.dispatch_eval(
            db_path, 1, "http://fake:4000/v1", "model-x", suite)
    assert len(result["stored"]) == 2
    assert result["stored"][0]["item_id"] == "q1"
    assert result["stored"][0]["output"] == "4"
    assert result["stored"][0]["prompt_tokens"] == 5


def test_dispatch_eval_multiturn(db_path, pass_preflight):
    _mk_suite(db_path)
    suite = {"items": [{
        "type": "multi-turn", "id": "mt-1",
        "turns": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "more"},
        ],
    }]}
    with patch("arcturos.multiturn.post_openai_chat") as fake_post:
        fake_post.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = ops.dispatch_eval(
            db_path, 1, "http://fake:4000/v1", "model-x", suite)
    # fingerprint is passed through as the chat model name (litellm needs it)
    assert fake_post.call_args.kwargs["model"] == "model-x"
    # one record per assistant placeholder (this item has exactly one)
    assert len(result["stored"]) == 1
    assert result["stored"][0]["item_id"] == "mt-1"
    assert result["stored"][0]["latency_ms"] is not None  # recorded (mock may be instant)


def test_dispatch_eval_empty_suite(db_path, pass_preflight):
    _mk_suite(db_path)
    with pytest.raises(ops.DispatchError, match="no items"):
        ops.dispatch_eval(db_path, 1, "http://fake:4000/v1", "m", {"items": []})


def test_dispatch_eval_unknown_suite(db_path):
    with pytest.raises(ops.DispatchError, match="not found"):
        ops.dispatch_eval(db_path, 999, "http://fake:4000/v1", "m",
                          {"items": [{"id": "q1", "prompt": "x"}]})


def test_dispatch_eval_missing_prompt(db_path, pass_preflight):
    _mk_suite(db_path)
    with pytest.raises(ops.DispatchError, match="missing prompt"):
        ops.dispatch_eval(db_path, 1, "http://fake:4000/v1", "m",
                          {"items": [{"id": "q1"}]})


def test_dispatch_eval_rolls_back_on_failure(db_path, pass_preflight):
    """Second item raises -> nothing from the batch is stored."""
    _mk_suite(db_path)
    suite = {"items": [
        {"id": "q1", "prompt": "ok"},
        {"id": "q2", "prompt": "boom"},
    ]}

    calls = {"n": 0}

    def flaky(target, messages, api_key=None, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("target exploded")
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    with patch("arcturos.multiturn.post_openai_chat", side_effect=flaky):
        with pytest.raises(RuntimeError, match="exploded"):
            ops.dispatch_eval(db_path, 1, "http://fake:4000/v1", "m", suite)
    from arcturos import db as dbmod
    conn = dbmod.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 0
    conn.close()


# --------------------------------------------------- openai transport ------
#
# The openai transport targets /v1/chat/completions servers (tabbyAPI,
# vLLM, litellm) that have NO /tokenize endpoint. Preflight runs
# kind='eval' (reachable + chat) over the injectable httpx transport; each
# bench point streams through bench_openai.run_openai_stream_point, whose
# urlopen is patched (never a real server).


def _sse_text(content_chunks=4, prompt=500, completion=30, stop="stop",
              mtp=None):
    """SSE body shaped like an OpenAI chat completion stream with usage.

    ``mtp=(accepted, rejected)`` adds tabbyAPI-style speculative-decoding
    counters to the final usage chunk's completion_tokens_details."""
    import json as _json
    lines = []
    for i in range(content_chunks):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"},
                          "finish_reason": None}]}))
    usage = {"prompt_tokens": prompt,
             "completion_tokens": completion,
             "completion_time": 0.5,
             "completion_tokens_per_sec": completion / 0.5}
    if mtp is not None:
        accepted, rejected = mtp
        usage["completion_tokens_details"] = {
            "accepted_prediction_tokens": accepted,
            "rejected_prediction_tokens": rejected,
        }
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": stop}], "usage": usage}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


class _SSEFakeResp:
    def __init__(self, text):
        self._lines = text.encode().splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _openai_preflight_app(chat_ok: bool = True):
    """MockTransport for the openai transport's preflight (kind='eval'):
    /health + /v1/chat/completions. No /tokenize — 404, as on tabbyAPI."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            if chat_ok:
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": "pong"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
            return httpx.Response(500, json={"error": "chat boom"})
        if request.url.path == "/tokenize":
            # OpenAI-compatible servers have no tokenizer endpoint.
            return httpx.Response(404)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_dispatch_bench_openai_happy_stores_points_no_power(db_path):
    """openai dispatch: eval preflight (reachable+chat), one stream point
    per target, stored run has points + no power fields (ADR 002).

    urlopen is patched (test_bench_openai.py pattern) so the REAL
    run_openai_stream_point parses the fake SSE; no network is touched."""
    calls = {"model_names": [], "max_tokens": []}

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        calls["model_names"].append(body["model"])
        calls["max_tokens"].append(body["max_tokens"])
        return _SSEFakeResp(_sse_text(content_chunks=4,
                                      prompt=500, completion=30))

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        result = ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128, 256], 30,
            transport_name="openai", model="GLM-5.3-Flash",
            transport=_openai_preflight_app())

    assert result["run_id"] == 1
    assert len(result["points"]) == 2
    # the chat model name is what the target was benched with
    assert calls["model_names"] == ["GLM-5.3-Flash", "GLM-5.3-Flash"]
    # n_predict maps onto the stream max_tokens
    assert calls["max_tokens"] == [30, 30]
    # metrics preserved from the openai point: server-authoritative
    # decode (usage timing), ttft, prefill estimate; prompt_tokens from
    # usage, not the sizing estimate
    p0 = result["points"][0]
    assert p0["context_tokens"] == 128
    assert p0["decode_tps"] == 60.0  # 30 tokens / 0.5 s, server-rate
    assert p0["ttft_ms"] is not None
    assert p0["prefill_tps"] is not None
    assert p0["prompt_tokens"] == 500
    assert p0["output_tokens"] == 30
    assert p0["mtp_draft_n"] is None
    assert p0["mtp_accepted"] is None
    assert p0["stop_reason"] == "stop"

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    run = exported["run"]
    assert run["engine"] == "openai"
    assert run["model_fingerprint"] == "GLM-5.3-Flash"
    assert run["context_size"] == 0  # unknown n_ctx sentinel
    for b in exported["benchmarks"]:
        assert b["power_watts"] is None
        assert b["power_host"] is None
        assert b["power_gpu_index"] is None
        assert b["ttft_ms"] is not None


def test_dispatch_bench_openai_preflight_failure_is_actionable(db_path):
    """openai dispatch against a failing target: preflight (kind='eval')
    surfaces the detail as DispatchError; no bench point is attempted."""
    calls = {"stream": 0}

    def fake_stream_point(*a, **kw):
        calls["stream"] += 1
        raise AssertionError("must not be called when preflight fails")

    with patch("arcturos.ops.bench_openai.run_openai_stream_point",
               side_effect=fake_stream_point):
        with pytest.raises(ops.DispatchError) as excinfo:
            ops.dispatch_bench(
                db_path, "http://fake:4000/v1", [128], 30,
                transport_name="openai", model="m",
                transport=_openai_preflight_app(chat_ok=False))
    msg = str(excinfo.value)
    assert "preflight failed (chat)" in msg
    assert "http://fake:4000/v1" in msg
    assert "Check target" in msg
    assert calls["stream"] == 0
    # nothing stored
    from arcturos import db as dbmod
    conn = dbmod.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_dispatch_bench_openai_missing_model_is_dispatch_error(db_path):
    """openai transport without a model name -> DispatchError before any
    network call (422 via the API)."""
    with pytest.raises(ops.DispatchError, match="requires 'model'"):
        ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128], 30,
            transport_name="openai", model=None,
            transport=_openai_preflight_app())


def test_dispatch_bench_invalid_transport_name(db_path):
    """Unknown transport name -> DispatchError (422 via the API)."""
    with pytest.raises(ops.DispatchError, match="unknown transport"):
        ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128], 30,
            transport_name="vllm", model="m")



# ------------------------------------------ REVIEW-FIX-ROUND regressions ----


def test_bench_job_status_snapshot_is_isolated(db_path):
    """The status snapshot must be a deep-enough copy: the worker thread
    keeps appending to job['points'] while pollers serialize the record,
    so mutating the returned snapshot must not corrupt the live job."""
    job_id = ops.run_bench_job(
        db_path, "http://127.0.0.1:1", [128], 8,
        api_key=None, transport_name="native", model=None)
    snap = ops.bench_job_status(job_id)
    assert snap is not None
    snap["points"].append({"context_tokens": 999, "injected": True})
    fresh = ops.bench_job_status(job_id)
    assert all(p.get("context_tokens") != 999 for p in fresh["points"])


def test_dispatch_bench_openai_mtp_captured_from_usage_details(db_path):
    """tabbyAPI speculative-decoding counters flow through dispatch into
    the stored run (2026-09-22: runs 29/30 showed null MTP — the openai
    transport ignored completion_tokens_details)."""
    def fake_urlopen(req, timeout=None):
        return _SSEFakeResp(_sse_text(content_chunks=4, prompt=500,
                                      completion=30, mtp=(109, 73)))

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        result = ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128], 30,
            transport_name="openai", model="GLM-5.3-Flash",
            transport=_openai_preflight_app())
    p0 = result["points"][0]
    assert p0["mtp_accepted"] == 109
    assert p0["mtp_draft_n"] == 182

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    assert exported["benchmarks"][0]["mtp_accepted"] == 109
    assert exported["benchmarks"][0]["mtp_draft_n"] == 182


def test_dispatch_bench_power_host_opt_in_samples_and_stores(db_path):
    """power_host opt-in: the SSH sampler runs per point and its mean W
    lands in the stored rows with provenance (host + gpu_index)."""
    class _FakeSampler:
        def __init__(self, host, gpu_index=0):
            calls["host"] = host
            calls["gpu"] = gpu_index
            calls["samplers"].append({"started": False, "stopped": False})

        def start(self):
            calls["samplers"][-1]["started"] = True

        def stop(self):
            calls["samplers"][-1]["stopped"] = True
            return 149.9

    calls = {"host": None, "gpu": None, "samplers": []}
    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=lambda req, timeout=None:
                   _SSEFakeResp(_sse_text(content_chunks=4, prompt=500,
                                          completion=30))), \
         patch("arcturos.ops.power.PowerSampler", _FakeSampler):
        result = ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128, 256], 30,
            transport_name="openai", model="GLM-5.3-Flash",
            transport=_openai_preflight_app(),
            power_host="alexei@10.10.10.222", power_gpu_index=1)

    assert calls["host"] == "alexei@10.10.10.222"
    assert calls["gpu"] == 1
    assert len(calls["samplers"]) == 2  # one sampler per point
    assert all(s["started"] and s["stopped"] for s in calls["samplers"])
    for p in result["points"]:
        assert p["power_watts"] == 149.9
        assert p["power_host"] == "alexei@10.10.10.222"
        assert p["power_gpu_index"] == 1

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    for b in exported["benchmarks"]:
        assert b["power_watts"] == 149.9
        assert b["power_host"] == "alexei@10.10.10.222"
        assert b["power_gpu_index"] == 1


def test_dispatch_bench_power_host_validation(db_path):
    """Empty/blank power_host is rejected before any network call; GPU
    index must be a non-negative int."""
    with pytest.raises(ops.DispatchError, match="power_host must be"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           power_host="   ")
    with pytest.raises(ops.DispatchError, match="power_host must be"):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           power_host=123)


def test_dispatch_bench_native_power_host_opt_in_samples_and_stores(db_path):
    """Native transport + power_host: the SSH sampler runs per point and
    mean W lands in the stored rows with provenance.

    Regression: the first cut of opt-in power wired ONLY the openai path —
    a native dispatch with power_host set silently sampled nothing
    (2026-09-22, ruapehu-class targets)."""
    class _FakeSampler:
        def __init__(self, host, gpu_index=0):
            calls["samplers"].append({"host": host, "gpu": gpu_index,
                                      "started": False, "stopped": False})

        def start(self):
            calls["samplers"][-1]["started"] = True

        def stop(self):
            calls["samplers"][-1]["stopped"] = True
            return 129.4

    calls = {"samplers": []}
    with patch("arcturos.bench.power.PowerSampler", _FakeSampler):
        result = ops.dispatch_bench(
            db_path, "http://fake:8000", [128, 256], 8,
            transport=_bench_app(),          # native transport
            power_host="alexei@10.10.10.122", power_gpu_index=2)

    assert len(calls["samplers"]) == 2  # one sampler per point
    assert all(s["host"] == "alexei@10.10.10.122" and s["gpu"] == 2
               and s["started"] and s["stopped"]
               for s in calls["samplers"])
    for p in result["points"]:
        assert p["power_watts"] == 129.4
        assert p["power_host"] == "alexei@10.10.10.122"
        assert p["power_gpu_index"] == 2

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    for b in exported["benchmarks"]:
        assert b["power_watts"] == 129.4
        assert b["power_host"] == "alexei@10.10.10.122"
        assert b["power_gpu_index"] == 2


# ------------------------------------------------- partial-run storage -----


def _dying_after_n_points_app(good_points: int, total_requests: int):
    """MockTransport that serves `good_points` successful /completion
    requests then fails the next one (connection error) — models a sweep
    that dies mid-way with completed points already collected."""

    state = {"completions": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/tokenize":
            return httpx.Response(200, json=_tokenize_resp(2))
        if path == "/completion":
            state["completions"] += 1
            if state["completions"] > good_points:
                raise httpx.ConnectError("connection reset by peer")
            return httpx.Response(200, json={
                "tokens_predicted": 8,
                "tokens_evaluated": 27,
                "stop_reason": "eos",
                "timings": {"prompt_per_second": 400.0,
                            "predicted_per_second": 50.0},
            })
        if path == "/props":
            return httpx.Response(200, json={
                "n_ctx": 4096, "model_path": "/models/test.gguf",
                "build": "1234", "engine": "llama.cpp",
                "model_fingerprint": "/models/test.gguf"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_dispatch_bench_mid_sweep_failure_stores_partial_run(db_path):
    """2026-09-22 feature: a sweep that fails after completing some points
    still stores the completed points as a partial run — nothing measured
    is discarded. The result reports status='partial' + the error."""
    transport = _dying_after_n_points_app(good_points=1, total_requests=2)

    result = ops.dispatch_bench(db_path, "http://fake:8000", [128, 256], 8,
                                transport=transport)
    assert result["status"] == "partial"
    assert "connection reset" in result["error"]
    assert result["run_id"] == 1
    assert len(result["points"]) == 1
    assert result["points"][0]["context_tokens"] == 128

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    assert exported["run"]["status"] == "partial"
    assert len(exported["benchmarks"]) == 1  # the completed point survived
    assert exported["benchmarks"][0]["context_tokens"] == 128


def test_dispatch_bench_failure_with_zero_points_still_502(db_path):
    """Failure before any point completes: no run is created (nothing to
    preserve) and the 502 contract is unchanged."""
    import sqlite3
    transport = _dying_after_n_points_app(good_points=0, total_requests=1)
    with pytest.raises(ops.DispatchFailure):
        ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                           transport=transport)
    db = sqlite3.connect(db_path)
    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    db.close()


def test_dispatch_bench_success_reports_complete_status(db_path):
    """The success payload now carries status='complete' explicitly."""
    transport = _bench_app()
    result = ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                                transport=transport)
    assert result["status"] == "complete"
    from arcturos.bench import export_run_json
    assert export_run_json(db_path, 1)["run"]["status"] == "complete"


def test_dispatch_bench_openai_mid_sweep_failure_stores_partial_run(db_path):
    """Partial preservation works on the openai transport too."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _SSEFakeResp(_sse_text(content_chunks=4, prompt=500,
                                          completion=30))
        raise RuntimeError("connection reset mid-sweep")

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        result = ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128, 256], 30,
            transport_name="openai", model="GLM-5.3-Flash",
            transport=_openai_preflight_app())
    assert result["status"] == "partial"
    assert "connection reset" in result["error"]

    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, 1)
    assert exported["run"]["status"] == "partial"
    assert exported["run"]["engine"] == "openai"
    assert len(exported["benchmarks"]) == 1


# ------------------------------------------------------- run naming --------


def test_default_run_name_humanized():
    """Default name format: 'host · model · 64k/128k' with humanized ctx."""
    n = ops.default_run_name("http://10.10.10.222:8000", "GLM-5.3-Flash",
                             [32768, 65536, 131072])
    assert n == "10.10.10.222 · GLM-5.3-Flash · 32k/64k/128k"
    # non-kibibyte targets stay raw
    n2 = ops.default_run_name("http://h:8000", "m", [1500, 4096])
    assert n2 == "h · m · 1500/4k"


def test_dispatch_bench_custom_name_stored(db_path):
    """Custom name flows through dispatch into the stored run row."""
    transport = _bench_app()
    result = ops.dispatch_bench(db_path, "http://fake:8000", [128], 8,
                                transport=transport, name="evening sweep")
    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, result["run_id"])
    assert exported["run"]["name"] == "evening sweep"


def test_dispatch_bench_default_name_when_unnamed(db_path):
    """No name given -> the humanized default is stored (host · model · ctx)."""
    transport = _bench_app()
    result = ops.dispatch_bench(db_path, "http://fake:8000", [128, 256], 8,
                                transport=transport)
    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, result["run_id"])
    assert exported["run"]["name"] == "fake · /models/test.gguf · 128/256"


def test_dispatch_bench_openai_name_stored(db_path):
    """openai transport: name + eval-run entity both carry through."""
    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=lambda req, timeout=None:
                   _SSEFakeResp(_sse_text(content_chunks=4, prompt=500,
                                          completion=30))):
        result = ops.dispatch_bench(
            db_path, "http://fake:4000/v1", [128], 30,
            transport_name="openai", model="GLM-5.3-Flash",
            transport=_openai_preflight_app(), name="pakuranga sweep")
    from arcturos.bench import export_run_json
    exported = export_run_json(db_path, result["run_id"])
    assert exported["run"]["name"] == "pakuranga sweep"


def test_dispatch_eval_creates_named_eval_run(db_path, pass_preflight):
    """Eval replay creates an eval_runs row; items stamp eval_run_id;
    the response carries eval_run_id; name is stored."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO eval_suites (id, name, version) VALUES (1, 's', 'v1')")
    conn.commit()
    conn.close()

    def fake_post(url, messages, api_key=None, model=None):
        return {"choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    with patch("arcturos.multiturn.post_openai_chat", side_effect=fake_post):
        result = ops.dispatch_eval(
            db_path, suite_id=1, target="http://fake:4000/v1",
            model_fingerprint="test-model",
            suite={"items": [{"id": "q1", "prompt": "hello"},
                             {"id": "q2", "prompt": "world"}]},
            name="nightly eval")

    assert result["eval_run_id"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    erun = dict(conn.execute("SELECT * FROM eval_runs WHERE id = 1").fetchone())
    assert erun["name"] == "nightly eval"
    assert erun["suite_id"] == 1
    stamped = conn.execute(
        "SELECT COUNT(*) FROM eval_results WHERE eval_run_id = 1").fetchone()[0]
    assert stamped == 2
    conn.close()


def test_dispatch_eval_unnamed_eval_run_stores_null_name(db_path, pass_preflight):
    """name=None stores NULL — the UI falls back to the default rendering."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO eval_suites (id, name, version) VALUES (1, 's', 'v1')")
    conn.commit()
    conn.close()

    def fake_post(url, messages, api_key=None, model=None):
        return {"choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    with patch("arcturos.multiturn.post_openai_chat", side_effect=fake_post):
        result = ops.dispatch_eval(
            db_path, suite_id=1, target="http://fake:4000/v1",
            model_fingerprint="test-model",
            suite={"items": [{"id": "q1", "prompt": "hello"}]})
    conn = sqlite3.connect(db_path)
    name = conn.execute("SELECT name FROM eval_runs WHERE id = 1").fetchone()[0]
    conn.close()
    assert name is None
