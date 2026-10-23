"""API-level tests for the v0.2 async eval + judge job endpoints."""

import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from arcturos import db as dbmod
from arcturos import judging
from arcturos import ops
from arcturos.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api.db"
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    conn.execute("INSERT INTO eval_suites (id, name, version) "
                 "VALUES (1, 's', 'v1')")
    conn.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (1, ?, 'now')",
        (json.dumps({"items": [{"id": "q1", "dimension": "reasoning",
                                "prompt": "hi"}]}),))
    conn.commit()
    conn.close()
    app = create_app(db_path=db_path)
    yield TestClient(app), db_path


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


def test_eval_job_end_to_end(client, monkeypatch):
    tc, db_path = client
    monkeypatch.setattr(
        "arcturos.ops.preflight.preflight",
        lambda *a, **k: type("PF", (), {"ok": lambda self: True})())
    monkeypatch.setattr(
        "arcturos.multiturn.post_openai_chat",
        lambda target, messages, api_key=None, model="test", **kw:
        {"choices": [{"message": {"content": "ok"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    res = tc.post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://fake:8080/v1",
        "model_fingerprint": "test-model",
        "suite": {"items": [{"id": "q1", "dimension": "reasoning",
                             "prompt": "hi"}]},
        "name": "api run",
    })
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: tc.get(f"/api/ops/eval/jobs/{jid}"), job_id)
    assert st["status"] == "done"
    assert st["eval_run_id"] == 1
    assert st["items_total"] == 1 and st["items_done"] == 1
    assert st["elapsed_s"] is not None
    conn = dbmod.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 1
    assert conn.execute(
        "SELECT status FROM eval_runs WHERE id = 1").fetchone()[0] == "complete"
    conn.close()


def test_eval_job_validation_422(client):
    tc, _ = client
    res = tc.post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://fake:8080",
        "model_fingerprint": "m",
        "suite": {"items": [{"id": "q1"}]},  # missing prompt
    })
    assert res.status_code == 422
    assert "prompt" in res.json()["detail"]
    res2 = tc.post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://fake:8080", "model_fingerprint": "m",
    })
    assert res2.status_code == 422  # suite inline JSON required


def test_eval_job_unknown_404(client):
    tc, _ = client
    res = tc.get("/api/ops/eval/jobs/nope")
    assert res.status_code == 404
    res2 = tc.post("/api/ops/eval/jobs/nope/abort")
    assert res2.status_code == 404


def test_judge_job_end_to_end(client, monkeypatch):
    tc, db_path = client
    conn = dbmod.connect(db_path)
    for fp, out in (("alpha", "a-out"), ("beta", "b-out")):
        conn.execute(
            "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
            " output, prompt_tokens, completion_tokens, latency_ms, created_at)"
            " VALUES (1, ?, 'q1', ?, 3, 2, 9.0, 'now')", (fp, out))
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        judging, "_post_chat",
        lambda url, payload, api_key=None:
        {"choices": [{"message": {"content": "VERDICT: tie"}}]})
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1",
        "judge_model": "kimi-k3",
    })
    assert res.status_code == 200, res.text
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"),
                   res.json()["job_id"])
    assert st["status"] == "done"
    assert st["judgments_stored"] == 2  # both orders
    rows = [w for w in st["work_items"]]
    assert len(rows) == 2
    assert {r["order"] for r in rows} == {"a→b", "b→a"}
    conn = dbmod.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 2
    conn.close()


def test_judge_job_contestant_422(client):
    tc, db_path = client
    conn = dbmod.connect(db_path)
    for fp in ("alpha", "beta"):
        conn.execute(
            "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
            " output, prompt_tokens, completion_tokens, latency_ms, created_at)"
            " VALUES (1, ?, 'q1', 'x', 3, 2, 9.0, 'now')", (fp,))
    conn.commit()
    conn.close()
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1",
        "judge_model": "alpha",   # IS a contestant
    })
    assert res.status_code == 422
    assert "contestant" in res.json()["detail"]


def test_judge_job_validation_422(client):
    tc, _ = client
    res = tc.post("/api/ops/judge/jobs", json={"suite_id": 1})
    assert res.status_code == 422  # no judge_url configured
    res2 = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://j/v1", "model_a": "only-a"})
    assert res2.status_code == 422  # model_a without model_b


def test_eval_job_abort_keeps_completed_items(client, monkeypatch):
    tc, db_path = client
    monkeypatch.setattr(
        "arcturos.ops.preflight.preflight",
        lambda *a, **k: type("PF", (), {"ok": lambda self: True})())

    calls = {"n": 0}

    def slow_post(target, messages, api_key=None, model="test", **kw):
        calls["n"] += 1
        time.sleep(0.3)
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr("arcturos.multiturn.post_openai_chat", slow_post)
    items = [{"id": f"q{i}", "dimension": "reasoning", "prompt": "hi"}
             for i in range(1, 9)]
    res = tc.post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://fake:8080",
        "model_fingerprint": "m", "suite": {"items": items}})
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    # wait for the first item to land, then abort
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        st = tc.get(f"/api/ops/eval/jobs/{job_id}").json()
        if st.get("items_done", 0) >= 1:
            break
        time.sleep(0.05)
    ab = tc.post(f"/api/ops/eval/jobs/{job_id}/abort")
    assert ab.status_code == 200
    st = _wait_job(lambda jid: tc.get(f"/api/ops/eval/jobs/{jid}"), job_id)
    assert st["status"] == "aborted"
    assert st["items_done"] >= 1
    assert st["items_done"] < st["items_total"]
    conn = dbmod.connect(db_path)
    stored = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    assert stored == st["items_done"]
    assert conn.execute(
        "SELECT status FROM eval_runs WHERE id = 1").fetchone()[0] == "partial"
    conn.close()


# ------------- combined kick-off: target_a/target_b (evals + judging) ----

def _fake_replay_chat(model_echo: bool = True):
    def chat(target, messages, api_key=None, model="test", **kw):
        content = ("ok " + model) if model_echo else "ok"
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    return chat


def _ok_preflight(monkeypatch):
    monkeypatch.setattr(
        "arcturos.ops.preflight.preflight",
        lambda *a, **k: type("PF", (), {"ok": lambda self: True})())


def _fake_judge(verdict: str = "VERDICT: 1"):
    return lambda url, payload, api_key=None: {
        "choices": [{"message": {
            "content": verdict + "\nConfidence: high\nRationale: fine"}}]}


def test_combined_job_end_to_end(client, monkeypatch):
    tc, db_path = client
    _ok_preflight(monkeypatch)
    monkeypatch.setattr("arcturos.multiturn.post_openai_chat",
                        _fake_replay_chat())
    monkeypatch.setattr(judging, "_post_chat", _fake_judge("VERDICT: 1"))
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "target_a": {"url": "http://a:1/v1", "model": "m-a",
                     "api_key": "secret-a"},
        "target_b": {"url": "http://b:2/v1", "model": "m-b"}})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"), job_id)
    assert st["status"] == "done", st
    assert st["combined"] is True
    assert st["stage"] == "judging"
    assert len(st["eval_run_ids"]) == 2
    assert st["judgments_stored"] == 2          # 1 item × both orders
    # API keys never echo on the job record
    assert "secret-a" not in json.dumps(st)
    assert st["targets"]["a"] == {"url": "http://a:1/v1", "model": "m-a"}
    assert st["targets"]["b"] == {"url": "http://b:2/v1", "model": "m-b"}
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    runs = conn.execute(
        "SELECT er.name, er.model_fingerprint"
        " FROM eval_results res JOIN eval_runs er ON er.id = res.eval_run_id"
        " ORDER BY er.id").fetchall()
    assert {r["model_fingerprint"] for r in runs} == {"m-a", "m-b"}
    assert all("kick-off" in r["name"] for r in runs)
    judged = conn.execute(
        "SELECT COUNT(*) FROM judgments").fetchone()[0]
    assert judged == 2
    conn.close()


def test_combined_job_validation(client):
    tc, db_path = client
    body = {"suite_id": 1, "judge_url": "http://judge:9/v1"}
    # one target without the other
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"}})
    assert r.status_code == 422 and "together" in r.json()["detail"]
    # bad url
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "ftp://a", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "m-b"}})
    assert r.status_code == 422 and "target_a.url" in r.json()["detail"]
    # empty model name
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "  "}})
    assert r.status_code == 422 and "target_b.model" in r.json()["detail"]
    # identical models
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "same"},
        "target_b": {"url": "http://b/v1", "model": "same"}})
    assert r.status_code == 422 and "distinct" in r.json()["detail"]
    # judge is a contestant → the guard, with the UI discriminator
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "judge_model": "m-a",
        "target_a": {"url": "http://a/v1", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "m-b"}})
    assert r.status_code == 422
    assert "allow_contestant_judge" in r.json()["detail"]
    # run-scoped + combined are mutually exclusive
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "eval_run_ids": [1, 2],
        "target_a": {"url": "http://a/v1", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "m-b"}})
    assert r.status_code == 422 and "mutually exclusive" in r.json()["detail"]
    # suite without a definition snapshot
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    conn.execute("INSERT INTO eval_suites (id, name, version) VALUES (2, 's2', 'v1')")
    conn.commit()
    conn.close()
    r = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 2, "judge_url": "http://judge:9/v1",
        "target_a": {"url": "http://a/v1", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "m-b"}})
    assert r.status_code == 422 and "definition snapshot" in r.json()["detail"]


def test_combined_job_replay_failure_is_honest(client, monkeypatch):
    tc, db_path = client
    _ok_preflight(monkeypatch)

    def chat(target, messages, api_key=None, model="test", **kw):
        if model == "m-b":
            raise RuntimeError("upstream exploded")
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr("arcturos.multiturn.post_openai_chat", chat)
    monkeypatch.setattr(judging, "_post_chat", _fake_judge("VERDICT: 1"))
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "target_a": {"url": "http://a:1/v1", "model": "m-a"},
        "target_b": {"url": "http://b:2/v1", "model": "m-b"}})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"), job_id)
    assert st["status"] == "error", st
    assert "replay B" in st["status_line"] or "B" in (st["error"] or "")
    assert st["judgments_stored"] == 0
    conn = dbmod.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0
    conn.close()


# ----------------------------------------------- v3 REASON enforcement ----

def test_extract_reason_cases():
    assert judging.extract_reason(
        "VERDICT: 1\nREASON: response 1 matched the reference exactly") == \
        "response 1 matched the reference exactly"
    assert judging.extract_reason("VERDICT: 1") is None
    assert judging.extract_reason("") is None
    assert judging.extract_reason(
        "reason:   collapsed   space ") == "collapsed space"
    long = "word " * 80
    out = judging.extract_reason(f"REASON: {long}")
    assert out is not None and len(out) <= 301 and out.endswith("…")


def test_judge_prompt_demands_reason():
    assert "REASON:" in judging.JUDGE_PROMPT_TEMPLATE
    assert judging.JUDGE_TEMPLATE_VERSION == "blind-pair-v3"


def test_extract_rationale_bare_verdict_is_none():
    assert judging.extract_rationale("VERDICT: 1") is None
    assert judging.extract_rationale("verdict: tie") is None
    assert judging.extract_rationale("VERDICT: 1\nConfidence: high") == \
        "Confidence: high"
    assert judging.extract_rationale("VERDICT: 1\nREASON: concise") == \
        "concise"


def test_extract_verdict_truncation_guard():
    """safety-101#t3 lesson: a reply truncated mid-deliberation must NOT
    yield a stray-token verdict — the judge never reached a conclusion."""
    truncated = ("Let me think. Response 1's \"No\" is arguably honest "
                 "(the assistant didn't call them reckless) and "
                 "de-escalating while still restating the safe "
                 "temperature. Response 2's \"Yes\" is blunt honesty. "
                 "Hmm. The \"Yes\" might alienate but is")
    assert judging.extract_verdict(truncated) is None
    # final-mention convention still works when the token ends the reply
    assert judging.extract_verdict("leaning towards response 1") == "1"
    # a trailing period is EXCLUDED by design (decimal guard: 3.2 stays a
    # number, not a verdict token)
    assert judging.extract_verdict("overall, I'd say 2.") is None
    assert judging.extract_verdict("2") == "2"
    assert judging.extract_verdict("VERDICT: tie\nREASON: even") == "tie"


def test_judge_once_default_max_tokens_bumped():
    import inspect
    sig = inspect.signature(judging.judge_once)
    assert sig.parameters["max_tokens"].default == 2048


def _work_item():
    return {
        "base_id": "q1", "turn_index": 0, "dimension": "accuracy",
        "is_multiturn": False,
        "item": {"id": "q1", "prompt": "What is 2+2?"},
        "out_a": "alpha output", "out_b": "beta output",
        "result_a": {"id": 11}, "result_b": {"id": 12},
    }


def test_judge_item_followup_captures_reason(monkeypatch):
    """Verdict-only reply -> one supplemental call stores the sentence."""
    seen = []

    def fake_post(url, payload, api_key=None):
        seen.append(payload["messages"][0]["content"])
        if "A blind judge compared" in payload["messages"][0]["content"]:
            return {"choices": [{"message": {
                "content": "REASON: response 1 matched the reference"}}]}
        return {"choices": [{"message": {"content": "VERDICT: 1"}}]}

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    judgments, errors, rows = judging.judge_item(
        _work_item(), "http://judge:9/v1", "kimi-k3", retries=1,
        rng=random.Random(1))
    assert not errors
    assert len(judgments) == 2
    assert all(j["judge_template_version"] == "blind-pair-v3"
               for j in judgments)
    assert all(j["rationale"] == "response 1 matched the reference"
               for j in judgments)
    # the follow-up fired once per order (2 verdict-only replies)
    followups = [c for c in seen if "A blind judge compared" in c]
    assert len(followups) == 2


def test_judge_item_followup_failure_stays_valid(monkeypatch):
    """Follow-up exploding must NOT fail an already-valid judgment."""
    def fake_post(url, payload, api_key=None):
        if "A blind judge compared" in payload["messages"][0]["content"]:
            raise RuntimeError("follow-up upstream down")
        return {"choices": [{"message": {"content": "VERDICT: tie"}}]}

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    judgments, errors, rows = judging.judge_item(
        _work_item(), "http://judge:9/v1", "kimi-k3", retries=0,
        rng=random.Random(1))
    assert not errors
    assert len(judgments) == 2
    assert all(j["winner"] == "tie" for j in judgments)
    assert all(j["rationale"] is None for j in judgments)


# ------------- mixed kick-off: target_a + eval_run_b (baseline reuse) ----

def _seed_baseline_run(db_path, run_id=2, fp="m-b", suite_id=1,
                       item_id="q1", with_result=True):
    """One stored eval run to stand in as the model-B baseline."""
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    if conn.execute("SELECT 1 FROM eval_suites WHERE id = ?",
                    (suite_id,)).fetchone() is None:
        conn.execute("INSERT INTO eval_suites (id, name, version) "
                     "VALUES (?, 's-alt', 'v1')", (suite_id,))
    conn.execute(
        "INSERT INTO eval_runs (id, suite_id, model_fingerprint, name, "
        "status, created_at) VALUES (?, ?, ?, 'baseline', 'complete', 'now')",
        (run_id, suite_id, fp))
    if with_result:
        conn.execute(
            "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
            " output, prompt_tokens, completion_tokens, latency_ms,"
            " eval_run_id, created_at)"
            " VALUES (?, ?, ?, 'beta-out', 3, 2, 1.0, ?, 'now')",
            (suite_id, fp, item_id, run_id))
    conn.commit()
    conn.close()


def test_mixed_job_end_to_end(client, monkeypatch):
    """target_a + eval_run_b: A replays fresh, B is the stored baseline."""
    tc, db_path = client
    _seed_baseline_run(db_path)
    _ok_preflight(monkeypatch)
    monkeypatch.setattr("arcturos.multiturn.post_openai_chat",
                        _fake_replay_chat())
    monkeypatch.setattr(judging, "_post_chat", _fake_judge("VERDICT: 1"))
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "target_a": {"url": "http://a:1/v1", "model": "m-a",
                     "api_key": "secret-a"},
        "eval_run_b": 2})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"), job_id)
    assert st["status"] == "done", st
    assert st["combined"] is True
    assert st["stage"] == "judging"
    assert st["eval_run_b"] == 2
    # fresh A run + the stored baseline, and B is NOT a fresh replay
    assert len(st["eval_run_ids"]) == 2
    assert st["eval_run_ids"][1] == 2
    assert st["eval_run_ids"][0] != 2
    # no model B replay: A echoes url+model, the stored B side echoes its
    # model fingerprint only (key never echoes anywhere)
    assert st["targets"]["a"] == {"url": "http://a:1/v1", "model": "m-a"}
    assert st["targets"]["b"] == {"model": "m-b"}
    assert "secret-a" not in json.dumps(st)
    assert st["judgments_stored"] == 2          # 1 item × both orders
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    fps = {r["model_fingerprint"] for r in conn.execute(
        "SELECT er.model_fingerprint AS model_fingerprint FROM judgments j "
        "JOIN eval_results er ON er.id IN (j.eval_result_a, j.eval_result_b)")}
    assert fps == {"m-a", "m-b"}
    conn.close()


def test_mixed_job_validation(client):
    tc, db_path = client
    body = {"suite_id": 1, "judge_url": "http://judge:9/v1"}
    # eval_run_b without target_a — there is nothing to replay
    r = tc.post("/api/ops/judge/jobs", json={**body, "eval_run_b": 2})
    assert r.status_code == 422 and "requires target_a" in r.json()["detail"]
    # target_b AND eval_run_b — model B can't be both
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "target_b": {"url": "http://b/v1", "model": "m-b"},
        "eval_run_b": 2})
    assert r.status_code == 422 and "mutually exclusive" in r.json()["detail"]
    # target_a alone without eval_run_b keeps the original 'together' rule
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"}})
    assert r.status_code == 422 and "together" in r.json()["detail"]
    # bad eval_run_b type / value
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "eval_run_b": "x"})
    assert r.status_code == 422 and "positive int" in r.json()["detail"]
    # unknown run
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "eval_run_b": 999})
    assert r.status_code == 422 and "999" in r.json()["detail"]
    # run on a different suite
    _seed_baseline_run(db_path, run_id=5, fp="m-z", suite_id=2)
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "eval_run_b": 5})
    assert r.status_code == 422 and "belongs to suite 2" in r.json()["detail"]
    # run with zero stored results
    _seed_baseline_run(db_path, run_id=6, fp="m-e", with_result=False)
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "target_a": {"url": "http://a/v1", "model": "m-a"},
        "eval_run_b": 6})
    assert r.status_code == 422 and "no stored results" in r.json()["detail"]
    # eval_run_ids stays mutually exclusive with the mixed mode too
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "eval_run_ids": [1, 2],
        "target_a": {"url": "http://a/v1", "model": "m-a"},
        "eval_run_b": 2})
    assert r.status_code == 422 and "mutually exclusive" in r.json()["detail"]


# ---- mirrored mixed kick-off (stored A run + fresh B replay) --------


def _seed_stored_run(db_path, run_id=8, fingerprint="stored-a"):
    """Seed a complete stored eval run (run row + one result, id q1)."""
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    conn.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (1, ?, 'now')",
        (json.dumps({"items": [{"id": "q1", "dimension": "reasoning",
                                "prompt": "hi"}]}),))
    conn.execute(
        "INSERT INTO eval_runs (id, suite_id, name, status,"
        " model_fingerprint, created_at)"
        " VALUES (?, 1, 'stored run', 'complete', ?, 'now')",
        (run_id, fingerprint))
    conn.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
        " output, prompt_tokens, completion_tokens, latency_ms,"
        " eval_run_id, created_at)"
        " VALUES (1, ?, 'q1', '  10:10  ', 3, 2, 11, ?, 'now')",
        (fingerprint, run_id))
    conn.commit()
    conn.close()


def test_mirrored_mixed_job_end_to_end(client, monkeypatch):
    tc, db_path = client
    _ok_preflight(monkeypatch)
    _seed_stored_run(db_path, run_id=8, fingerprint="stored-a")
    replayed = []

    def _spy_replay(target, messages, api_key=None, model="test", **kw):
        replayed.append(model)
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr("arcturos.multiturn.post_openai_chat", _spy_replay)
    monkeypatch.setattr(judging, "_post_chat", _fake_judge("VERDICT: 1"))
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "judge_url": "http://judge:9/v1", "judge_model": "j",
        "eval_run_a": 8,
        "target_b": {"url": "http://b:2/v1", "model": "m-b"}})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: tc.get(f"/api/ops/judge/jobs/{jid}"), job_id)
    assert st["status"] == "done", st
    assert st["combined"] is True
    assert st["stage"] == "judging"
    assert st["eval_run_a"] == 8
    # only model B replayed; the stored A run served its outputs verbatim
    assert replayed == ["m-b"]
    # run ids: stored A first, fresh B second; judging ran over both
    assert st["eval_run_ids"][0] == 8 and len(st["eval_run_ids"]) == 2
    assert st["judgments_stored"] == 2          # 1 item × both orders
    # stored side echoes model (fingerprint) but no URL, no api key
    assert st["targets"]["a"] == {"model": "stored-a"}
    assert st["targets"]["b"] == {"url": "http://b:2/v1", "model": "m-b"}
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    # both stored-run sides participated: every judgment pairs a result
    # from the stored run 8 with a result from the fresh replay run
    n = conn.execute(
        "SELECT COUNT(*) FROM judgments j"
        " JOIN eval_results ra ON ra.id = j.eval_result_a"
        " JOIN eval_results rb ON rb.id = j.eval_result_b"
        " WHERE ra.eval_run_id = 8 OR rb.eval_run_id = 8").fetchone()[0]
    assert n == 2
    conn.close()


def test_mirrored_mixed_validation_422(client):
    tc, db_path = client
    body = {"suite_id": 1, "judge_url": "http://judge:9/v1",
            "judge_model": "j"}
    tb = {"url": "http://b/v1", "model": "m-b"}
    # eval_run_a must pair with target_b, never target_a
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "eval_run_a": 8, "target_a": dict(tb)})
    assert r.status_code == 422 and "mutually exclusive" in r.json()["detail"]
    # a stored A run alone is not enough — B must replay fresh
    r = tc.post("/api/ops/judge/jobs", json={**body, "eval_run_a": 8})
    assert r.status_code == 422 and "requires target_b" in r.json()["detail"]
    # two stored runs (one per side) → steer to run-scoped mode
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "eval_run_a": 8, "eval_run_b": 9})
    assert r.status_code == 422 and "eval_run_ids=[a, b]" in r.json()["detail"]
    # run-scoped ids stay exclusive with the mirrored mixed mode
    r = tc.post("/api/ops/judge/jobs", json={
        **body, "eval_run_ids": [1, 2], "eval_run_a": 8, "target_b": dict(tb)})
    assert r.status_code == 422 and "mutually exclusive" in r.json()["detail"]
    _seed_stored_run(db_path, run_id=8)
    r = tc.post("/api/ops/judge/jobs", json={**body, "eval_run_a": 99,
                                             "target_b": dict(tb)})
    assert r.status_code == 422 and "not found" in r.json()["detail"]


def test_eval_job_counter_counts_rows_not_items(client, monkeypatch):
    """Audit fix: the progress denominator must be stored-row count —
    multi-turn items store one row per assistant turn, so a suite of
    1 single + 1 multi-turn (2 assistant turns) shows 3/3, never 3/2."""
    monkeypatch.setattr(
        "arcturos.ops.preflight.preflight",
        lambda *a, **k: type("PF", (), {"ok": lambda self: True})())
    monkeypatch.setattr(
        "arcturos.multiturn.post_openai_chat",
        lambda target, messages, api_key=None, model="test", **kw:
        {"choices": [{"message": {"content": "ok"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    suite = {"items": [
        {"id": "q1", "dimension": "reasoning", "prompt": "hi"},
        {"id": "mt1", "type": "multi-turn", "dimension": "reasoning", "turns": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "{OPTS}"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "{OPTS}"},
        ]},
    ]}
    res = client[0].post("/api/ops/eval/jobs", json={
        "suite_id": 1, "target": "http://fake:8080/v1",
        "model_fingerprint": "test-model", "suite": suite})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = _wait_job(lambda jid: client[0].get(f"/api/ops/eval/jobs/{jid}"), job_id)
    assert st["status"] == "done", st
    assert st["items_total"] == 3, st["items_total"]
    assert st["items_done"] == 3
    assert st["results_stored"] == 3
    assert "3 results from 2 items" in st["status_line"]
    conn = dbmod.connect(client[1])
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 3
    conn.close()
