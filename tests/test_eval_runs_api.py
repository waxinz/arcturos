"""API tests for the Results-page flow (2026-10-15):

- GET /api/eval-runs (runs table, newest first, result counts)
- GET /api/eval-runs/{id} (run detail: run + stored result rows)
- POST /api/ops/judge/jobs with eval_run_ids (run-scoped judging)
- GET /api/judgments/pair (SxS rows, order-independent)
- drill-down pages carry the shell identity
"""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from arcturos import db as dbmod
from arcturos.main import create_app


def _seed(db_path):
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    conn.execute("INSERT INTO eval_suites (id, name, version) "
                 "VALUES (1, 'core', 'v1')")
    conn.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (1, ?, 'now')",
        (json.dumps({"items": [{"id": "q1", "dimension": "reasoning",
                                "prompt": "hi"}]}),))
    # two replay runs of DIFFERENT models + a third run (same suite,
    # third model) used to prove run scoping
    for rid, fp in ((1, "model-a"), (2, "model-b"), (3, "model-c")):
        conn.execute(
            "INSERT INTO eval_runs (id, suite_id, model_fingerprint, name, "
            "status, created_at) VALUES (?, 1, ?, ?, 'complete', 'now')",
            (rid, fp, f"run {fp}"))
    for rid, fp, out in ((1, "model-a", "alpha"), (2, "model-b", "beta"),
                         (3, "model-c", "gamma")):
        conn.execute(
            "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, "
            "output, prompt_tokens, completion_tokens, latency_ms, "
            "eval_run_id, created_at) "
            "VALUES (1, ?, 'q1', ?, 3, 2, 1.0, ?, 'now')", (fp, out, rid))
    conn.commit()
    conn.close()


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "runs.db"
    _seed(db_path)
    app = create_app(db_path=db_path)
    return TestClient(app), db_path


def _wait_job(tc, job_id, timeout=10):
    """Poll a judge job to a terminal state and return its record."""
    deadline = time.monotonic() + timeout
    job: dict = {}
    while time.monotonic() < deadline:
        jj = tc.get(f"/api/ops/judge/jobs/{job_id}").json()
        assert isinstance(jj, dict)
        job = jj
        if job["status"] in ("done", "error", "aborted"):
            return job
        time.sleep(0.05)
    return job


def test_list_eval_runs_newest_first_with_counts(client):
    tc, _ = client
    rows = tc.get("/api/eval-runs").json()
    assert [r["id"] for r in rows] == [3, 2, 1]          # newest first
    assert rows[0]["suite_name"] == "core"
    assert rows[0]["model_fingerprint"] == "model-c"
    assert all(r["result_count"] == 1 for r in rows)


def test_eval_run_detail_and_404(client):
    tc, _ = client
    data = tc.get("/api/eval-runs/2").json()
    assert data["run"]["model_fingerprint"] == "model-b"
    assert data["run"]["suite_name"] == "core"
    assert [r["item_id"] for r in data["results"]] == ["q1"]
    assert data["results"][0]["eval_run_id"] == 2
    assert tc.get("/api/eval-runs/999").status_code == 404


def test_judge_job_scoped_to_selected_runs(client, monkeypatch):
    tc, db_path = client
    # fake multimodal judge: prefers the FIRST presented response
    monkeypatch.setattr(
        "arcturos.judging._post_chat",
        lambda url, payload, api_key=None, timeout=120: {
            "choices": [{"message": {"content":
                        "VERDICT: 1\nConfidence: 0.9\nRationale: first wins"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "eval_run_ids": [1, 2], "judge_model": "judge-x"})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    deadline = time.monotonic() + 10
    job: dict | None = None
    while time.monotonic() < deadline:
        j = tc.get(f"/api/ops/judge/jobs/{job_id}").json()
        assert isinstance(j, dict)
        job = j
        if job["status"] in ("done", "error", "aborted"):
            break
        time.sleep(0.05)
    assert job is not None and job["status"] == "done", job
    assert job["eval_run_ids"] == [1, 2]

    # the stored judgments reference ONLY results from runs 1 and 2 —
    # model-c (run 3) must not appear even though it has results too
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    fps = {r["model_fingerprint"] for r in conn.execute(
        "SELECT er.model_fingerprint AS model_fingerprint FROM judgments j "
        "JOIN eval_results er ON er.id IN (j.eval_result_a, j.eval_result_b)")}
    conn.close()
    assert fps == {"model-a", "model-b"}


def test_judge_job_eval_run_ids_validation(client, monkeypatch):
    tc, _ = client
    # instant-failing judge so the abort path never touches the network
    monkeypatch.setattr(
        "arcturos.judging._post_chat",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    for bad in ([1], "no", [0, 1], [1, "x"]):
        res = tc.post("/api/ops/judge/jobs",
                      json={"suite_id": 1, "eval_run_ids": bad})
        assert res.status_code == 422, bad
    # runs 1+3 = models a+c: a valid pair — should NOT 422 (judge_model
    # must be explicit: "(default)" trips the judge-not-contestant guard)
    res = tc.post("/api/ops/judge/jobs",
                  json={"suite_id": 1, "eval_run_ids": [1, 3],
                        "judge_model": "judge-x"})
    assert res.status_code == 200, res.text
    tc.post(f"/api/ops/judge/jobs/{res.json()['job_id']}/abort")


def test_judge_job_abort_partial_preservation(client, monkeypatch):
    """Regression (2026-10-15): the Stop button 500'd — abort_judge_job
    called flag.set() but AbortFlag only has request()."""
    tc, _ = client
    gate = threading.Event()

    def blocking_judge(url, payload, api_key=None, timeout=120):
        gate.wait(timeout=5)               # hold mid-job until released
        return {"choices": [{"message": {"content":
                "VERDICT: 1\nConfidence: 0.9\nRationale: first wins"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr("arcturos.judging._post_chat", blocking_judge)
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "eval_run_ids": [1, 2], "judge_model": "judge-x"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    time.sleep(0.4)                        # let the job reach the judge call
    stop = tc.post(f"/api/ops/judge/jobs/{job_id}/abort")
    assert stop.status_code == 200, stop.text   # was a 500 before the fix
    assert stop.json()["abort_requested"] is True
    gate.set()                             # release the in-flight call
    deadline = time.monotonic() + 10
    j: dict = {}
    while time.monotonic() < deadline:
        jj = tc.get(f"/api/ops/judge/jobs/{job_id}").json()
        assert isinstance(jj, dict)
        j = jj
        if j["status"] in ("done", "error", "aborted"):
            break
        time.sleep(0.05)
    assert j["status"] == "aborted", j
    # partial preservation: the order-0 call had already completed and is
    # kept even though the job ended aborted
    assert j["stored"] >= 1, j


def test_judgments_pair_sxs_order_independent(client, monkeypatch):
    tc, db_path = client
    # deterministic judge: B (second presented) always wins
    monkeypatch.setattr(
        "arcturos.judging._post_chat",
        lambda url, payload, api_key=None, timeout=120: {
            "choices": [{"message": {"content":
                        "VERDICT: 2\nConfidence: 0.8\nRationale: second wins"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    res = tc.post("/api/ops/judge/jobs", json={
        "suite_id": 1, "eval_run_ids": [1, 2], "judge_model": "judge-x"})
    assert res.status_code == 200, res.text
    _wait_job(tc, res.json()["job_id"])
    fwd = tc.get("/api/judgments/pair",
                 params={"suite_id": 1, "model_a": "model-a",
                         "model_b": "model-b"}).json()
    rev = tc.get("/api/judgments/pair",
                 params={"suite_id": 1, "model_a": "model-b",
                         "model_b": "model-a"}).json()
    assert len(fwd) == len(rev) == 3          # 2 order rows + flip tiebreak
    # Rows store in CANONICAL a→b orientation regardless of query order:
    # fingerprint_a is always model-a; the verdict is remapped to the
    # canonical side. With a SECOND-ALWAYS-WINS judge the two orders must
    # produce OPPOSITE canonical winners — that flip IS the position-bias
    # signal the double judging exists to measure — and the tiebreak row
    # (same judge, fresh random order) resolves the split.
    assert {r["fingerprint_a"] for r in fwd + rev} == {"model-a"}
    assert {r["fingerprint_b"] for r in fwd + rev} == {"model-b"}
    order_rows = [r for r in fwd if r["role"] != "tiebreak"]
    tb_rows = [r for r in fwd if r["role"] == "tiebreak"]
    assert len(order_rows) == 2 and len(tb_rows) == 1
    assert sorted(r["winner"] for r in order_rows) == ["a", "b"]
    assert tb_rows[0]["winner"] in ("a", "b")
    for r in fwd + rev:
        assert r["output_a"] == "alpha"       # canonical a's stored output
        assert r["item_id"] == "q1"
    # query order must not matter: identical row sets both ways
    assert {(r["id"], r["winner"]) for r in fwd} == \
           {(r["id"], r["winner"]) for r in rev}
    assert tc.get("/api/judgments/pair",
                  params={"suite_id": 1, "model_a": "model-a",
                          "model_b": "model-c"}).json() == []


def test_drilldown_pages_carry_shell_identity(client):
    tc, _ = client
    run_page = tc.get("/evals/runs/1")
    assert run_page.status_code == 200
    assert 'data-page="evalrundetail"' in run_page.text
    assert 'class="topnav"' in run_page.text          # nav injected
    assert '<!--NAV-->' not in run_page.text
    jp = tc.get("/judgments/pair")
    assert jp.status_code == 200
    assert 'data-page="jpair"' in jp.text
    assert 'class="topnav"' in jp.text
    # category summary card (leads the page, above the meta strip)
    assert 'id="jsum"' in jp.text
    assert "Category summary" in jp.text
