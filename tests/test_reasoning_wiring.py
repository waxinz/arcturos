"""Per-model reasoning controls (2026-10-15): optional reasoning_effort +
reasoning_budget on every eval-suite-run start surface.

Covers: the validator, the wire-body contract of post_openai_chat (fields
present when set, byte-identical body when unset), dispatch validation,
the eval-job API (422s + passthrough), and the judge combined kick-off
(per-side reasoning reaches the right side's replay only).
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from arcturos import db as dbmod
from arcturos import judging
from arcturos import multiturn as mt
from arcturos import ops
from arcturos.main import create_app


# ----------------------------- validator --------------------------------

@pytest.mark.parametrize("effort,budget", [
    (None, None), ("minimal", None), ("high", None), (None, 2048),
    ("low", 512), ("medium", 1), ("high", 32768),
    ("xhigh", None), ("max", 512),
])
def test_validate_reasoning_accepts_valid(effort, budget):
    eff, bud = mt.validate_reasoning(effort, budget)
    assert (eff, bud) == (effort, budget)


@pytest.mark.parametrize("effort", ["extreme", "", "HIGH", 7, True, ["low"]])
def test_validate_reasoning_rejects_bad_effort(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        mt.validate_reasoning(effort, None)


@pytest.mark.parametrize("budget", [0, -1, 32769, True, False, "2048", 1.5])
def test_validate_reasoning_rejects_bad_budget(budget):
    with pytest.raises(ValueError, match="reasoning_budget"):
        mt.validate_reasoning("low", budget)


# --------------------------- wire body ----------------------------------

class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()


def _capture_request(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(mt.urlreq, "urlopen", fake_urlopen)
    return captured


def test_chat_body_includes_reasoning_when_set(monkeypatch):
    captured = _capture_request(monkeypatch)
    resp = mt.post_openai_chat(
        "http://t:1/v1", [{"role": "user", "content": "hi"}],
        reasoning_effort="high", reasoning_budget=2048)
    assert resp["choices"][0]["message"]["content"] == "ok"
    body = captured["body"]
    assert body["reasoning_effort"] == "high"
    assert body["reasoning_budget"] == 2048
    # max_tokens unset → field omitted entirely (2026-10-25: None never
    # reaches the wire — the Maya/DSF <= 1024 throttle lesson); an
    # explicit value is passed through unchanged.
    assert "max_tokens" not in body
    assert body["stream"] is False
    assert captured["url"].endswith("/v1/chat/completions")
    # 600s request budget (2026-10-26, raised from 300): thinky models
    # can legitimately think for minutes and shared-server stalls eat
    # the old 300s (run #17's four 300s burns).
    assert captured["timeout"] == 600
    captured2 = _capture_request(monkeypatch)
    mt.post_openai_chat("http://t:1/v1", [{"role": "user", "content": "hi"}],
                        max_tokens=512)
    assert captured2["body"]["max_tokens"] == 512


def test_chat_body_omits_reasoning_when_unset(monkeypatch):
    captured = _capture_request(monkeypatch)
    mt.post_openai_chat("http://t:1/v1", [{"role": "user", "content": "hi"}])
    body = captured["body"]
    # byte-identical historical contract: the keys never appear as null
    assert "reasoning_effort" not in body
    assert "reasoning_budget" not in body


# ------------------------- dispatch + job API ---------------------------

_SUITE = {"items": [{"id": "q1", "dimension": "reasoning", "prompt": "hi"}]}


def test_dispatch_eval_rejects_bad_reasoning():
    with pytest.raises(ops.DispatchError, match="reasoning_effort"):
        ops.dispatch_eval(db_path=":memory:", suite_id=1,
                          target="http://t:1/v1", model_fingerprint="m",
                          suite=_SUITE, reasoning_effort="extreme")
    with pytest.raises(ops.DispatchError, match="reasoning_budget"):
        ops.dispatch_eval(db_path=":memory:", suite_id=1,
                          target="http://t:1/v1", model_fingerprint="m",
                          suite=_SUITE, reasoning_budget=0)


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "reason.db"
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    conn.execute("INSERT INTO eval_suites (id, name, version) VALUES (1, 's', 'v1')")
    conn.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (1, ?, 'now')",
        (json.dumps(_SUITE),))
    conn.commit()
    conn.close()
    app = create_app(db_path=db_path)
    return TestClient(app), db_path


def test_eval_job_api_reasoning_422s(client):
    tc, _ = client
    base = {"suite_id": 1, "target": "http://t:1/v1",
            "model_fingerprint": "m", "suite": _SUITE}
    r = tc.post("/api/ops/eval/jobs", json={**base, "reasoning_effort": "bogus"})
    assert r.status_code == 422 and "reasoning_effort" in r.json()["detail"]
    r = tc.post("/api/ops/eval/jobs", json={**base, "reasoning_budget": 0})
    assert r.status_code == 422 and "reasoning_budget" in r.json()["detail"]
    r = tc.post("/api/ops/eval/jobs", json={**base, "reasoning_budget": "2048"})
    assert r.status_code == 422 and "reasoning_budget" in r.json()["detail"]


def test_eval_job_api_reasoning_passthrough(client, monkeypatch):
    tc, _ = client
    seen: dict = {}

    def fake_run_eval_job(**kw):
        seen.update(kw)
        return "job-42"

    monkeypatch.setattr(ops, "run_eval_job", fake_run_eval_job)
    r = tc.post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://t:1/v1", "model_fingerprint": "m",
        "suite": _SUITE, "reasoning_effort": "low", "reasoning_budget": 1024})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "job-42"
    assert seen["reasoning_effort"] == "low"
    assert seen["reasoning_budget"] == 1024


def test_eval_job_max_tokens_optional(client, monkeypatch):
    # 2026-10-25 (Maya/DSF <= 1024 throttle): max_tokens is optional —
    # absent/None forwards None (the replay then OMITS the wire field,
    # server budget); an explicit positive int passes through; junk 422s.
    tc, _ = client
    seen: dict = {}

    def fake_run_eval_job(**kw):
        seen.update(kw)
        return "job-42"

    monkeypatch.setattr(ops, "run_eval_job", fake_run_eval_job)
    base = {"suite_id": 1, "target": "http://t:1/v1",
            "model_fingerprint": "m", "suite": _SUITE}
    r = tc.post("/api/ops/eval/jobs", json=base)
    assert r.status_code == 200, r.text
    assert seen["max_tokens"] is None
    r = tc.post("/api/ops/eval/jobs", json={**base, "max_tokens": 4096})
    assert r.status_code == 200, r.text
    assert seen["max_tokens"] == 4096
    for bad in (0, -5, "1024", True):
        r = tc.post("/api/ops/eval/jobs", json={**base, "max_tokens": bad})
        assert r.status_code == 422, (bad, r.text)
        assert "max_tokens" in r.json()["detail"]


# ------------------- judge combined kick-off (per side) -----------------

def _wait_job(fetch, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    st = None
    while time.monotonic() < deadline:
        res = fetch(job_id)
        if res.status_code == 200:
            st = res.json()
            if st.get("status") in ("done", "error", "partial", "aborted"):
                return st
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {st}")


def test_judge_target_reasoning_validation(client):
    tc, _ = client
    body = {"suite_id": 1, "judge_url": "http://judge:9/v1",
            "judge_model": "j",
            "target_a": {"url": "http://a:1/v1", "model": "m-a",
                         "reasoning_effort": "extreme"},
            "target_b": {"url": "http://b:2/v1", "model": "m-b"}}
    r = tc.post("/api/ops/judge/jobs", json=body)
    assert r.status_code == 422 and "reasoning_effort" in r.json()["detail"]
    r2 = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a:1/v1", "model": "m-a",
                             "reasoning_budget": 0}})
    assert r2.status_code == 422 and "reasoning_budget" in r2.json()["detail"]


def test_combined_job_reasoning_reaches_each_side(client, monkeypatch):
    tc, _ = client
    replay_kw: list[tuple[str, dict]] = []

    def chat(target, messages, api_key=None, model="test", **kw):
        replay_kw.append((model, kw))
        return {"choices": [{"message": {"content": "ok " + model}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr(
        "arcturos.ops.preflight.preflight",
        lambda *a, **k: type("PF", (), {"ok": lambda self: True})())
    monkeypatch.setattr("arcturos.multiturn.post_openai_chat", chat)
    monkeypatch.setattr(judging, "_post_chat", lambda url, payload, api_key=None: {
        "choices": [{"message": {
            "content": "VERDICT: 1\nConfidence: high\nRationale: fine"}}]})
    r = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "target_a": {"url": "http://a:1/v1", "model": "m-a",
                     "reasoning_effort": "high", "reasoning_budget": 2048},
        "target_b": {"url": "http://b:2/v1", "model": "m-b",
                     "reasoning_effort": "low"}})
    assert r.status_code == 200, r.text
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"),
                   r.json()["job_id"])
    assert st["status"] == "done", st
    by_model = {}
    for model, kw in replay_kw:
        by_model.setdefault(model, []).append(kw)
    assert by_model["m-a"] and all(
        kw.get("reasoning_effort") == "high" and kw.get("reasoning_budget") == 2048
        for kw in by_model["m-a"])
    assert by_model["m-b"] and all(
        kw.get("reasoning_effort") == "low" and kw.get("reasoning_budget") is None
        for kw in by_model["m-b"])
    # 2026-10-25: the combined kick-off defaults to OMITTING max_tokens
    # (server budget — the Maya/DSF <= 1024 throttle lesson); an explicit
    # eval_max_tokens passes through to the replay calls.
    assert all(kw.get("max_tokens") is None for kws in by_model.values()
               for kw in kws)
    replay_kw.clear()
    r2 = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "target_a": {"url": "http://a:1/v1", "model": "m-a"},
        "target_b": {"url": "http://b:2/v1", "model": "m-b"},
        "eval_max_tokens": 4096})
    assert r2.status_code == 200, r2.text
    st2 = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"),
                    r2.json()["job_id"])
    assert st2["status"] == "done", st2
    assert replay_kw and all(kw.get("max_tokens") == 4096
                             for _, kw in replay_kw)
