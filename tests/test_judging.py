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


def test_select_pairs_trims_serving_preamble(conn):
    """The judge sees answers, not serving-stack wrappers (evals-101 #8v#9).

    qwen-style endpoints store a leading blank-line preamble verbatim;
    left in, it reached the judge as "stray blank lines" and decided
    otherwise-identical pairs. The worklist trims; the store stays raw
    (append-only) — same doctrine as the reference scorer.
    """
    _mk_suite(conn, items=[{"id": "q1", "dimension": "accuracy",
                            "prompt": "only the time"}])
    _mk_result(conn, 1, "alpha", "q1", "\n\n  10:10  \n")
    _mk_result(conn, 1, "beta", "q1", "10:10")
    work = judging.select_pairs(conn, 1)
    assert len(work) == 1
    assert work[0]["out_a"] == "10:10"
    assert work[0]["out_b"] == "10:10"
    # the store itself stays untouched (verbatim, append-only)
    raw = conn.execute(
        "SELECT output FROM eval_results WHERE model_fingerprint='alpha'"
    ).fetchone()[0]
    assert raw == "\n\n  10:10  \n"


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
        # nudge must NOT fire (asserted below — exactly 2 order calls).
        # The two orders then DISAGREE (a_first→b, b_first→a), so the
        # flip-resolution tiebreak fires: call 3 comes back a tie, which
        # degrades the split to a draw.
        if len(calls) < 3:
            return _judge_resp("VERDICT: 2\nREASON: response 2 was sharper")
        return _judge_resp("VERDICT: tie\nREASON: cannot separate them")

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
    assert len(judgments) == 3  # both presentation orders + flip tiebreak
    assert len(calls) == 3
    # The two calls must present DIFFERENT orders (that is the whole point
    # of the swap): one call shows A first, the other B first.
    def _orders_shown(content):
        return content.find("answer A") < content.find("answer B")
    order_calls = calls[:2]  # the tiebreak (call 3) is covered below
    assert [_orders_shown(c["messages"][0]["content"]) for c in order_calls] \
        == [True, False] or \
        [_orders_shown(c["messages"][0]["content"]) for c in order_calls] \
        == [False, True]
    # Order rows: the judge that always picks response 2 nets one a + one b
    # (the decided split); the tiebreak row resolves it to a draw.
    order_judgs = judgments[:2]
    assert {j["winner"] for j in order_judgs} == {"a", "b"}
    assert judgments[2]["winner"] == "tie"
    assert judgments[2]["role"] == "tiebreak"
    assert all(j["role"] == "order" for j in order_judgs)
    assert all(j["judge_template_version"] == judging.JUDGE_TEMPLATE_VERSION
               for j in judgments)
    # v3: the REASON line is captured verbatim as the stored rationale
    assert [j["rationale"] for j in order_judgs] \
        == ["response 2 was sharper"] * 2
    assert judgments[2]["rationale"] == "cannot separate them"
    assert sorted(r["order"] for r in rows) == ["a→b", "b→a", "tiebreak"]
    # progress verdicts match the judgments' winners as a set
    assert {r["verdict"] for r in rows} == {"a", "b", "tie"}
    # Forensics (2026-10-23): every row records which side was presented
    # first, and the raw reply is kept verbatim. The call↔judgment pairing
    # is positional (orders[i] produces calls[i] and judgments[i]), so the
    # FULL attribution chain is checkable: presented-A-first call ↔
    # presentation 'a_first' ↔ winner 'b' for an always-picks-response-2
    # judge. This is the chain that shipped incoherent on sa101-agent-004
    # #521 when presentation was ephemeral.
    raw = "VERDICT: 2\nREASON: response 2 was sharper"
    assert len(calls) == len(judgments) == 3
    for call, j in zip(order_calls, order_judgs):
        content = call["messages"][0]["content"]
        a_first_shown = content.find("answer A") < content.find("answer B")
        assert j["presentation"] == ("a_first" if a_first_shown else "b_first")
        assert j["raw_reply"] == raw
        # an always-picks-response-2 judge remaps: first-presented loses
        assert j["winner"] == ("b" if a_first_shown else "a")
    assert {j["presentation"] for j in order_judgs} == {"a_first", "b_first"}
    tb = judgments[2]
    assert tb["raw_reply"] == "VERDICT: tie\nREASON: cannot separate them"
    assert tb["presentation"] in ("a_first", "b_first")


def test_judge_item_tiebreak_resolves_decided_split(monkeypatch):
    """Always-picks-first judge → the orders split a/b; the tiebreak call
    (same always-first behavior, fresh random order) breaks the tie in the
    row math: the tiebreak's presented-first side ends with 2 wins vs 1."""
    calls = []

    def fake_post(url, payload, api_key=None):
        calls.append(payload)
        return _judge_resp("VERDICT: 1\nREASON: first presented is better")

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
        work_item, "http://judge/v1", "kimi-k3",
        rng=__import__("random").Random(3))
    assert errors == []
    assert len(judgments) == 3 and len(calls) == 3
    by_role = {}
    for j in judgments:
        by_role.setdefault(j["role"], []).append(j)
    assert sorted(by_role) == ["order", "tiebreak"]
    assert len(by_role["order"]) == 2
    # orders split (always-first judge remaps to opposite canonical sides)
    assert {j["winner"] for j in by_role["order"]} == {"a", "b"}
    # tiebreak: same judge behavior on a fresh random order
    tb = by_role["tiebreak"][0]
    assert tb["winner"] == ("a" if tb["presentation"] == "a_first" else "b")
    # resolution net: the tiebreak side now has 2 wins, the other side 1
    wins = {"a": 0, "b": 0}
    for j in judgments:
        wins[j["winner"]] += 1
    assert sorted(wins.values()) == [1, 2]
    # progress rows carry the tiebreak label with no error
    tb_row = [r for r in rows if r["order"] == "tiebreak"]
    assert len(tb_row) == 1 and tb_row[0]["error"] is None


def test_judge_item_no_tiebreak_when_orders_agree(monkeypatch):
    calls = []

    def fake_post(url, payload, api_key=None):
        calls.append(payload)
        return _judge_resp("VERDICT: tie\nREASON: even quality")

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
        work_item, "http://judge/v1", "kimi-k3",
        rng=__import__("random").Random(3))
    assert errors == []
    assert len(judgments) == len(calls) == 2  # agreeing orders: no tiebreak
    assert all(j["role"] == "order" and j["winner"] == "tie"
               for j in judgments)
    assert {r["order"] for r in rows} == {"a→b", "b→a"}


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
                                    judge_model="kimi-k3",
                                    tiebreak_flips=False)
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


# ------------------------------------------------------- both_bad (v4-ref) --

def test_extract_verdict_both_bad_variants():
    assert judging.extract_verdict("VERDICT: both_bad") == "both_bad"
    assert judging.extract_verdict("VERDICT: both-bad") == "both_bad"
    assert judging.extract_verdict("VERDICT: both bad") == "both_bad"
    assert judging.extract_verdict("VERDICT: BOTH_BAD") == "both_bad"
    # bare-token fallback at end (no VERDICT line)
    assert judging.extract_verdict("thinking... both_bad") == "both_bad"
    # mid-sentence stray mention -> None (truncation guard)
    assert judging.extract_verdict("one could be both_bad but ...") is None
    # tie still parses; 'bad' alone is not a verdict
    assert judging.extract_verdict("VERDICT: tie") == "tie"
    assert judging.extract_verdict("both are bad.") is None


def test_render_reference_kinds():
    assert judging._render_reference({"check": "exact", "value": "20"}) == \
        "The exact correct answer: '20'"
    assert "±0.5" in judging._render_reference(
        {"check": "numeric", "value": 2.4, "tolerance": 0.5})
    assert "must include" in judging._render_reference(
        {"check": "contains", "value": "bandwidth"})
    assert "42, 43" in judging._render_reference(
        {"check": "one_of", "values": [42, 43]})
    assert "pattern" in judging._render_reference(
        {"check": "regex", "pattern": r"\A20(\.0+)?\s*\Z"})
    # absent / unknown shapes -> None (plain v3 template, never invented)
    assert judging._render_reference(None) is None
    assert judging._render_reference({"check": "exact"}) is None
    assert judging._render_reference("20") is None
    assert judging._render_reference({"check": "mystery", "value": 1}) is None
    assert judging._render_reference({"check": "one_of", "values": []}) is None


def test_judge_once_uses_ref_template_only_with_reference(monkeypatch):
    seen = {}

    def fake_post(url, payload, api_key=None):
        seen["text"] = payload["messages"][0]["content"]
        return _judge_resp("VERDICT: both_bad\nREASON: both contradict 20")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    v = judging.judge_once("http://judge/v1", "kimi-k3", "What is 20 C in F?",
                           "20 C", "293.15 K", "reasoning",
                           reference={"check": "exact", "value": "68 F"})
    assert "REFERENCE" in seen["text"]
    assert "both_bad" in seen["text"]
    assert "'68 F'" in seen["text"]
    assert v["winner"] == "both_bad"
    assert v["template_version"] == judging.JUDGE_TEMPLATE_VERSION_REF

    seen.clear()
    v3 = judging.judge_once("http://judge/v1", "kimi-k3", "q", "a", "b", None)
    assert "REFERENCE" not in seen["text"]
    assert "both_bad" not in seen["text"]
    assert v3["template_version"] == judging.JUDGE_TEMPLATE_VERSION


def test_judge_item_stores_both_bad_as_qualified_tie(monkeypatch):
    def fake_post(url, payload, api_key=None):
        return _judge_resp("VERDICT: both_bad\nREASON: both wrong vs 68 F")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    work_item = {
        "base_id": "q1", "turn_index": None,
        "result_a": {"id": 11}, "result_b": {"id": 22},
        "out_a": "20 C", "out_b": "293.15 K", "history_a": [], "history_b": [],
        "fp_a": "alpha", "fp_b": "beta", "dimension": "reasoning",
        "item": {"id": "q1", "prompt": "convert",
                 "reference": {"check": "exact", "value": "68 F"}},
        "is_multiturn": False,
    }
    judgments, errors, rows = judging.judge_item(
        work_item, "http://judge/v1", "kimi-k3", retries=0)
    assert not errors
    assert len(judgments) == 2           # both presentation orders
    for j in judgments:
        assert j["winner"] == "tie"      # draw semantics preserved
        assert j["verdict_detail"] == "both_bad"
        assert j["judge_template_version"] == judging.JUDGE_TEMPLATE_VERSION_REF
    assert all(r["verdict"] == "both_bad" for r in rows)  # progress rows


def test_judge_item_plain_tie_keeps_no_detail(monkeypatch):
    def fake_post(url, payload, api_key=None):
        return _judge_resp("VERDICT: tie\nREASON: even quality")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    work_item = {
        "base_id": "q1", "turn_index": None,
        "result_a": {"id": 11}, "result_b": {"id": 22},
        "out_a": "A", "out_b": "B", "history_a": [], "history_b": [],
        "fp_a": "alpha", "fp_b": "beta", "dimension": "writing",
        "item": {"id": "q1", "prompt": "hi",
                 "reference": {"check": "exact", "value": "x"}},
        "is_multiturn": False,
    }
    judgments, errors, rows = judging.judge_item(
        work_item, "http://judge/v1", "kimi-k3", retries=0)
    assert judgments[0]["winner"] == "tie"
    assert judgments[0]["verdict_detail"] is None
    assert rows[0]["verdict"] == "tie"


def test_dispatch_judge_persists_verdict_detail(conn, monkeypatch):
    _seed_pair(conn)

    def fake_post(url, payload, api_key=None):
        return _judge_resp("VERDICT: both_bad\nREASON: both wrong")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    result = judging.dispatch_judge(_dbpath(conn), 1,
                                    judge_url="http://judge/v1",
                                    judge_model="kimi-k3")
    assert result["status"] == "complete"
    rows = conn.execute(
        "SELECT winner, verdict_detail FROM judgments ORDER BY id").fetchall()
    assert len(rows) == 2  # both presentation orders
    assert all(r["winner"] == "tie" and r["verdict_detail"] == "both_bad"
               for r in rows)


def test_dispatch_judge_persists_presentation_and_raw_reply(conn, monkeypatch):
    """Forensic fields ride the dispatch INSERT: presentation names the
    canonical side shown as Response 1, raw_reply keeps the judge's verbatim
    reply, role distinguishes order rows from the flip-resolution tiebreak.
    For a judge that always prefers the first-presented side the stored
    winner maps a_first→'a', b_first→'b' — and the orders split, so the
    default tiebreak fires and resolves the pair 2–1 in the row math (the
    attribution chain that was unprovable on sa101-agent-004 #521)."""
    _seed_pair(conn)

    raw = "VERDICT: 1\nREASON: first presented is better"

    def fake_post(url, payload, api_key=None):
        return _judge_resp(raw)

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    result = judging.dispatch_judge(_dbpath(conn), 1,
                                    judge_url="http://judge/v1",
                                    judge_model="kimi-k3")
    assert result["status"] == "complete"
    assert result["stored"] == 3  # both orders + flip-resolution tiebreak
    rows = conn.execute(
        "SELECT winner, presentation, raw_reply, role FROM judgments "
        "ORDER BY id").fetchall()
    assert len(rows) == 3
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    assert sorted(by_role) == ["order", "tiebreak"]
    assert len(by_role["order"]) == 2
    # always-picks-first judge → the presented-first side wins in both orders
    assert {r["winner"] for r in by_role["order"]} == {"a", "b"}
    for r in by_role["order"]:
        assert r["winner"] == ("a" if r["presentation"] == "a_first" else "b")
    assert {r["presentation"] for r in by_role["order"]} \
        == {"a_first", "b_first"}
    tb = by_role["tiebreak"][0]
    assert tb["winner"] == ("a" if tb["presentation"] == "a_first" else "b")
    assert all(r["raw_reply"] == raw for r in rows)


def test_migration_adds_verdict_detail_to_legacy_db(tmp_path):
    """A pre-both-bad DB (judgments without verdict_detail) migrates on
    connect: the column appears, legacy rows keep winner='tie' with NULL
    detail, and a fresh both-bad insert works."""
    import sqlite3

    p = tmp_path / "legacy.db"
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    # minimal legacy schema: just the tables/columns the migration touches
    c.executescript("""
        CREATE TABLE eval_suites (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  name TEXT, version TEXT);
        CREATE TABLE eval_results (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                   suite_id INTEGER, model_fingerprint TEXT,
                                   item_id TEXT, output TEXT,
                                   prompt_tokens INTEGER, completion_tokens INTEGER,
                                   latency_ms INTEGER, created_at TEXT);
        CREATE TABLE judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_result_a INTEGER NOT NULL REFERENCES eval_results(id),
            eval_result_b INTEGER NOT NULL REFERENCES eval_results(id),
            judge_model TEXT NOT NULL,
            judge_template_version TEXT NOT NULL,
            winner TEXT NOT NULL CHECK (winner IN ('a', 'b', 'tie')),
            confidence REAL,
            rationale TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO eval_suites (name, version) VALUES ('legacy', 'v1');
        INSERT INTO eval_results (suite_id, model_fingerprint, item_id, output,
                                  prompt_tokens, completion_tokens, latency_ms,
                                  created_at) VALUES
            (1, 'alpha', 'q1', 'x', 1, 1, 1, 'now'),
            (1, 'beta', 'q1', 'y', 1, 1, 1, 'now');
        INSERT INTO judgments (eval_result_a, eval_result_b, judge_model,
                               judge_template_version, winner, confidence,
                               rationale, created_at) VALUES
            (1, 2, 'old-judge', 'blind-pair-v1', 'tie', NULL, NULL, 'now');
    """)
    c.commit()
    c.close()

    conn = dbmod.connect(p)
    dbmod.init_db(conn)  # runs _migrate (house pattern: explicit init)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(judgments)")}
    assert "verdict_detail" in cols
    # forensic + role columns (2026-10-23) migrate the same way
    assert "presentation" in cols
    assert "raw_reply" in cols
    assert "role" in cols
    row = conn.execute(
        "SELECT winner, verdict_detail, presentation, raw_reply, role "
        "FROM judgments").fetchone()
    assert row["winner"] == "tie" and row["verdict_detail"] is None
    assert row["presentation"] is None and row["raw_reply"] is None
    assert row["role"] is None
    # new-style insert works on the migrated table
    conn.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at, "
        "verdict_detail, presentation, raw_reply, role) VALUES (1, 2, 'j', "
        "'blind-pair-v4-ref', 'tie', NULL, NULL, 'now', 'both_bad', "
        "'a_first', 'VERDICT: both_bad', 'tiebreak')")
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM judgments WHERE verdict_detail = "
        "'both_bad'").fetchone()[0]
    assert n == 1
    forensics = conn.execute(
        "SELECT presentation, raw_reply, role FROM judgments WHERE "
        "verdict_detail = 'both_bad'").fetchone()
    assert forensics["presentation"] == "a_first"
    assert forensics["raw_reply"] == "VERDICT: both_bad"
    assert forensics["role"] == "tiebreak"
    conn.close()


def test_both_bad_end_to_end_dispatch_report_csv(conn, monkeypatch):
    """Audit glue: judge pipeline -> store -> suite_report -> pair_summary ->
    CSV, one both-bad verdict flowing through every surface."""
    import csv as _csv
    import io as _io

    from arcturos import reports
    _mk_suite(conn, items=[{"id": "q1", "dimension": "reasoning",
                            "prompt": "7*6?",
                            "reference": {"check": "exact", "value": "42"}}])
    _mk_result(conn, 1, "alpha", "q1", "forty-one")
    _mk_result(conn, 1, "beta", "q1", "48")

    seen_payloads = []

    def fake_post(url, payload, api_key=None):
        seen_payloads.append(payload["messages"][0]["content"])
        return _judge_resp("VERDICT: both_bad\nREASON: neither says 42")

    monkeypatch.setattr(judging, "_post_chat", fake_post)
    result = judging.dispatch_judge(_dbpath(conn), 1,
                                    judge_url="http://judge/v1",
                                    judge_model="kimi-k3")
    assert result["status"] == "complete" and result["stored"] == 2
    # the reference actually reached the judge prompt (v4-ref variant)
    assert any("[REFERENCE]" in p and "'42'" in p for p in seen_payloads)

    rep = reports.suite_report(conn, 1)
    assert rep["overall"]["counts"] == {"a": 0, "b": 0, "tie": 0, "both_bad": 2}
    assert rep["overall"]["both_bad_rate"] == 100.0
    assert rep["pairs"][0]["both_bad_rate"] == 100.0
    assert all(v["counts"]["both_bad"] == 2
               for v in rep["by_dimension"].values())
    assert all(it["verdict_detail"] == "both_bad"
               for it in rep["item_level"])
    # draw math: both sides 0.5 of 2 -> 50.0
    s = reports.pair_summary(conn, 1, "alpha", "beta")
    assert s["overall"]["score_a"] == 50.0
    assert s["overall"]["score_b"] == 50.0
    csv_text = reports.export_report_csv(rep)
    rows = list(_csv.reader(_io.StringIO(csv_text)))
    header = rows[0]
    prow = next(r for r in rows[1:] if len(r) > 3
                and r[2] == "alpha" and r[3] == "beta")
    cells = dict(zip(header, prow))
    assert cells["both_bad"] == "2"
    assert float(cells["pct_both_bad"]) == 100.0
    assert float(cells["pct_tie"]) == 0.0
