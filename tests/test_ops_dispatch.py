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
                "build": "1234", "engine": "llama.cpp"})
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
        assert "preflight failed (reachable" in msg
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


def _sse_text(content_chunks=4, prompt=500, completion=30, stop="stop"):
    """SSE body shaped like an OpenAI chat completion stream with usage."""
    import json as _json
    lines = []
    for i in range(content_chunks):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"},
                          "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": stop}],
         "usage": {"prompt_tokens": prompt,
                   "completion_tokens": completion}}))
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
    # metrics preserved from the openai point: client-side decode, ttft,
    # prefill estimate; prompt_tokens from usage, not the sizing estimate
    p0 = result["points"][0]
    assert p0["context_tokens"] == 128
    assert p0["decode_tps"] is not None
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
