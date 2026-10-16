"""judging.py unit tests (v0.2) — verdict parsing, pair selection, position
swap, judge-not-contestant, job lifecycle."""

import json
import time

import pytest

from arcturos import db as dbmod
from arcturos import judging


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(tmp_path / "j.db")
    dbmod.init_db(c)
    yield c
    c.close()


def _mk_suite(conn, items=None, suite_id=1):
    conn.execute("INSERT INTO eval_suites (id, name, version) "
                 "VALUES (?, 's', 'v1')", (suite_id,))
    if items:
        conn.execute(
            "INSERT INTO suite_definitions (suite_id, payload, created_at) "
            "VALUES (?, ?, 'now')",
            (suite_id, json.dumps({"items": items})))
    conn.commit()


_result_seq = iter(range(1000))


def _mk_result(conn, suite_id, fp, item_id, output="ans", item_id_suffix=None):
    cur = conn.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
        " output, prompt_tokens, completion_tokens, latency_ms, created_at)"
        " VALUES (?, ?, ?, ?, 5, 3, 12.5, 'now')",
        (suite_id, fp, item_id, output))
    conn.commit()
    return cur.lastrowid


# --------------------------------------------------------- verdict parsing --

def test_extract_verdict_strict_line():
    assert judging.extract_verdict("VERDICT: 1") == "1"
    assert judging.extract_verdict("VERDICT: 2") == "2"
    assert judging.extract_verdict("VERDICT: tie") == "tie"
    assert judging.extract_verdict("verdict - Tie") == "tie"


def test_extract_verdict_reasoning_chatter_fallback():
    raw = "Let me think... response 1 is okay but 2 has the right shape.\nVERDICT: 2"
    assert judging.extract_verdict(raw) == "2"
    # last bare token wins (final-mention convention)
    assert judging.extract_verdict("I lean 1... actually no, 2") == "2"


def test_extract_verdict_unparseable():
    assert judging.extract_verdict("neither response satisfies me") is None
    assert judging.extract_verdict("") is None


# ------------------------------------------------- fingerprints / guard ----

def test_normalize_fingerprint_collapses_host_suffix():
    assert judging.normalize_fingerprint("GLM-5.3-Flash (demo-host-a:8000)") \
        == "GLM-5.3-Flash"
    assert judging.normalize_fingerprint("GLM-5.3-Flash") == "GLM-5.3-Flash"


def test_guard_judge_not_contestant():
    fps = ["GLM-5.3-Flash", "qwen3.8-27b"]
    judging.guard_judge_not_contestant("kimi-k3", fps)  # fine
    with pytest.raises(ValueError, match="contestant"):
        judging.guard_judge_not_contestant("GLM-5.3-Flash", fps)
    # substring: family self-preference risk
    with pytest.raises(ValueError, match="contestant"):
        judging.guard_judge_not_contestant("GLM-5.3-Flash-uncensored", fps)
    # explicit allow override
    judging.guard_judge_not_contestant("GLM-5.3-Flash", fps, allow=True)


def test_rubric_for_known_and_unknown():
    assert "REASONING" in judging.rubric_for("reasoning")
    assert "INSTRUCTION FOLLOWING" in judging.rubric_for("instruction-following")
    assert judging.rubric_for("vibes") == judging.rubric_for(None)


# ------------------------------------------------------------ select_pairs --

def test_select_pairs_latest_row_wins_and_two_model_gate(conn):
    _mk_suite(conn)
    ra1 = _mk_result(conn, 1, "alpha", "q1", "old answer")
    ra2 = _mk_result(conn, 1, "alpha", "q1", "new answer")  # later row
    rb = _mk_result(conn, 1, "beta", "q1", "beta answer")
    _mk_result(conn, 1, "alpha", "solo", "only me")  # single-model item

    work = judging.select_pairs(conn, 1)
    assert len(work) == 1
    w = work[0]
    assert w["base_id"] == "q1"
    assert w["result_a"]["id"] == ra2  # latest row supersedes
    assert w["out_a"] == "new answer"
    assert w["out_b"] == "beta answer"
    assert w["fp_a"] == "alpha" and w["fp_b"] == "beta"


def test_select_pairs_fingerprint_host_suffix_normalized(conn):
    _mk_suite(conn)
    _mk_result(conn, 1, "alpha (demo-host-a:8000)", "q1", "x")
    _mk_result(conn, 1, "alpha", "q1", "y")  # same model after normalization
    _mk_result(conn, 1, "beta", "q1", "z")
    work = judging.select_pairs(conn, 1)
    # alpha's two rows collapse to one model; only q1 x beta judges
    assert len(work) == 1
    assert {work[0]["fp_a"], work[0]["fp_b"]} == {"alpha", "beta"}


def test_select_pairs_model_pair_filter(conn):
    _mk_suite(conn)
    for fp in ("alpha", "beta", "gamma"):
        _mk_result(conn, 1, fp, "q1", f"{fp} out")
    work = judging.select_pairs(conn, 1, model_a="beta", model_b="gamma")
    assert len(work) == 1
    assert {work[0]["fp_a"], work[0]["fp_b"]} == {"beta", "gamma"}


def test_select_pairs_multiturn_per_turn_items_with_history(conn):
    _mk_suite(conn, items=[{
        "id": "mt1", "type": "multi-turn", "dimension": "writing",
        "turns": [
            {"role": "user", "content": "write a haiku"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "now make it funnier"},
            {"role": "assistant", "content": None},
        ]}])
    _mk_result(conn, 1, "alpha", "mt1", "haiku v1")
    _mk_result(conn, 1, "alpha", "mt1#t2", "funny v1")
    _mk_result(conn, 1, "beta", "mt1", "haiku v2")
    _mk_result(conn, 1, "beta", "mt1#t2", "funny v2")

    work = judging.select_pairs(conn, 1)
    assert [w["turn_index"] for w in work] == [1, 2]
    t2 = work[1]
    assert t2["out_a"] == "funny v1"
    assert t2["history_a"] == ["haiku v1"]
    assert t2["history_b"] == ["haiku v2"]
    assert t2["dimension"] == "writing"
    assert t2["is_multiturn"] is True


# ------------------------------------------------------------- judge_item --

def _judge_resp(text):
    return {"choices": [{"message": {"content": text}}]}


def test_judge_item_both_orders_stored_with_remap(monkeypatch):
    calls = []

    def fake_post(url, payload, api_key=None):
        calls.append(payload)
        # The judge always prefers response 2 regardless of presentation.
        # v3-style two-line reply: with a REASON present the follow-up
        # nudge must NOT fire (asserted below — exactly 2 calls).
        return _judge_resp("VERDICT: 2\nREASON: response 2 was sharper")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    work_item = {
        "base_id": "q1", "turn_index": None,
        "result_a": {"id": 11}, "result_b": {"id": 22},
        "out_a": "answer A", "out_b": "answer B",
        "history_a": [], "history_b": [],
        "fp_a": "alpha", "fp_b": "beta",
        "dimension": "reasoning",
        "item": {"id": "q1", "prompt": "hi"},
        "is_multiturn": False,
    }
    judgments, errors, rows = judging.judge_item(
        work_item, "http://judge/v1", "kimi-k3", rng=__import__("random").Random(7))
    assert errors == []
    assert len(judgments) == 2  # both presentation orders
    assert len(calls) == 2
    # The two calls must present DIFFERENT orders (that is the whole point
    # of the swap): one call shows A first, the other B first.
    def _orders_shown(content):
        return content.find("answer A") < content.find("answer B")
    assert [_orders_shown(c["messages"][0]["content"]) for c in calls] \
        == [True, False] or \
        [_orders_shown(c["messages"][0]["content"]) for c in calls] \
        == [False, True]
    # The judge that always picks response 2 must net out as one a + one b
    assert {j["winner"] for j in judgments} == {"a", "b"}
    assert all(j["judge_template_version"] == judging.JUDGE_TEMPLATE_VERSION
               for j in judgments)
    # v3: the REASON line is captured verbatim as the stored rationale
    assert all(j["rationale"] == "response 2 was sharper" for j in judgments)
    assert sorted(r["order"] for r in rows) == ["a→b", "b→a"]
    # progress verdicts match the judgments' winners as a set
    assert {r["verdict"] for r in rows} == {"a", "b"}


def test_judge_item_tie_and_invalid_verdict_retry(monkeypatch):
    seq = ["garbage without a verdict", "still nothing", "VERDICT: tie"]

    def fake_post(url, payload, api_key=None):
        if not seq:
            # v3 follow-up nudge for the bare "VERDICT: tie" — reply with
            # a reason so the nudge path terminates cleanly.
            return _judge_resp("REASON: neither response stood out")
        return _judge_resp(seq.pop(0))

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    work_item = {
        "base_id": "q1", "turn_index": None,
        "result_a": {"id": 11}, "result_b": {"id": 22},
        "out_a": "A", "out_b": "B", "history_a": [], "history_b": [],
        "fp_a": "alpha", "fp_b": "beta", "dimension": "writing",
        "item": {"id": "q1", "prompt": "hi"}, "is_multiturn": False,
    }
    judgments, errors, rows = judging.judge_item(
        work_item, "http://judge/v1", "kimi-k3", retries=1)
    # order 0: two invalid attempts -> error row; order 1: tie
    assert len(errors) == 1
    assert "order1" in errors[0]
    assert len(judgments) == 1
    assert judgments[0]["winner"] == "tie"
    verdict_rows = [r for r in rows if r["verdict"] is not None]
    assert len(verdict_rows) == 1


# --------------------------------------------------------- dispatch/jobs ----

def _seed_pair(conn):
    _mk_suite(conn, items=[{"id": "q1", "dimension": "reasoning",
                            "prompt": "What is 7*6?"}])
    _mk_result(conn, 1, "alpha", "q1", "42")
    _mk_result(conn, 1, "beta", "q1", "forty-two")


def test_dispatch_judge_stores_two_rows_per_item(conn, monkeypatch):
    _seed_pair(conn)

    def fake_post(url, payload, api_key=None):
        return _judge_resp("VERDICT: 1")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    result = judging.dispatch_judge(_dbpath(conn), 1,
                                    judge_url="http://judge/v1",
                                    judge_model="kimi-k3")
    assert result["status"] == "complete"
    assert result["stored"] == 2  # both orders for the one item
    assert result["invalid"] == 0
    rows = conn.execute("SELECT * FROM judgments ORDER BY id").fetchall()
    assert len(rows) == 2
    # A judge that ALWAYS picks response 1 is position-biased; position
    # swap must neutralize it into one a-win + one b-win (tie in aggregate).
    assert {r["winner"] for r in rows} == {"a", "b"}


def _dbpath(conn):
    row = conn.execute("PRAGMA database_list").fetchone()
    return row[2]


def test_dispatch_judge_rejects_contestant_judge(conn):
    _seed_pair(conn)
    with pytest.raises(ValueError, match="contestant"):
        judging.dispatch_judge(_dbpath(conn), 1, judge_url="http://judge/v1",
                               judge_model="alpha")


def test_dispatch_judge_no_pairs_honest_error(conn):
    _mk_suite(conn)
    _mk_result(conn, 1, "alpha", "q1", "only one model")
    with pytest.raises(ValueError, match="no judgeable pairs"):
        judging.dispatch_judge(_dbpath(conn), 1, judge_url="http://judge/v1",
                               judge_model="kimi-k3")


def test_judge_job_lifecycle_done(conn, monkeypatch):
    _seed_pair(conn)
    monkeypatch.setattr(judging, "_post_chat",
                        lambda url, payload, api_key=None:
                        _judge_resp("VERDICT: tie"))
    job_id = judging.run_judge_job(_dbpath(conn), 1,
                                   judge_url="http://judge/v1",
                                   judge_model="kimi-k3")
    deadline = time.monotonic() + 10
    st = None
    while time.monotonic() < deadline:
        st = judging.judge_job_status(job_id)
        if st and st["status"] in ("done", "error", "aborted"):
            break
        time.sleep(0.05)
    assert st["status"] == "done"
    assert st["judgments_stored"] == 2
    assert st["items_total"] == 1
    assert len(st["work_items"]) == 2  # one progress row per judgment
    assert st["elapsed_s"] is not None
    assert judging.judge_job_status("no-such-job") is None


def test_judge_job_contestant_rejected_synchronously(conn):
    """The guard lives at the dispatch boundary: run_judge_job raises
    BEFORE any thread starts (the API maps this to 422)."""
    _seed_pair(conn)
    with pytest.raises(ValueError, match="contestant"):
        judging.run_judge_job(_dbpath(conn), 1, judge_url="http://judge/v1",
                              judge_model="alpha")
