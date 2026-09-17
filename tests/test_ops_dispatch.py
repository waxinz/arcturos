"""Dispatch layer tests — bench/eval kick-off paths (ops.py).

Uses httpx.MockTransport to fake the llama.cpp native endpoints for the
bench path; the eval path fakes /v1/chat/completions via multiturn's
urlreq (patched).
"""

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


def _pass_preflight(monkeypatch=None):
    """Patch ops.preflight.preflight to a stub that always passes.

    The eval-path tests exercise dispatch logic, not preflight — the target
    URL is fake and would hit real DNS otherwise."""
    from arcturos.ops import preflight as pf_mod
    stub = pf_mod.PreflightResult(
        target="stub", kind="eval",
        checks=[{"name": "reachable", "ok": True, "detail": "stub"},
                {"name": "chat", "ok": True, "detail": "stub"}])
    pf_mod.preflight = lambda *a, **kw: stub


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


def test_dispatch_eval_single_turn(db_path):
    _mk_suite(db_path)
    _pass_preflight()
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


def test_dispatch_eval_multiturn(db_path):
    _mk_suite(db_path)
    _pass_preflight()
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


def test_dispatch_eval_empty_suite(db_path):
    _mk_suite(db_path)
    _pass_preflight()
    with pytest.raises(ops.DispatchError, match="no items"):
        ops.dispatch_eval(db_path, 1, "http://fake:4000/v1", "m", {"items": []})


def test_dispatch_eval_unknown_suite(db_path):
    with pytest.raises(ops.DispatchError, match="not found"):
        ops.dispatch_eval(db_path, 999, "http://fake:4000/v1", "m",
                          {"items": [{"id": "q1", "prompt": "x"}]})


def test_dispatch_eval_missing_prompt(db_path):
    _mk_suite(db_path)
    _pass_preflight()
    with pytest.raises(ops.DispatchError, match="missing prompt"):
        ops.dispatch_eval(db_path, 1, "http://fake:4000/v1", "m",
                          {"items": [{"id": "q1"}]})


def test_dispatch_eval_rolls_back_on_failure(db_path):
    """Second item raises -> nothing from the batch is stored."""
    _mk_suite(db_path)
    _pass_preflight()
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
