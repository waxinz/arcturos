"""API-level tests for the v0.2 async eval + judge job endpoints."""

import json
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
