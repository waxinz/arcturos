"""reports.py v0.2 tests — dimension lens, Wilson CIs, reference scores,
judge-generation scoping."""

import json

import pytest

from arcturos import db as dbmod
from arcturos import reports


@pytest.fixture()
def db(tmp_path):
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_db(conn)
    yield conn
    conn.close()


def _suite(db, items, name="s", version="v1"):
    cur = db.execute("INSERT INTO eval_suites (name, version) VALUES (?, ?)",
                     (name, version))
    suite_id = cur.lastrowid
    db.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (?, ?, 'now')", (suite_id, json.dumps({"items": items})))
    db.commit()
    return suite_id


def _result(db, suite_id, fp, item_id, output="out", run_id=None):
    cur = db.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id,"
        " output, prompt_tokens, completion_tokens, latency_ms, eval_run_id,"
        " created_at) VALUES (?, ?, ?, ?, 5, 3, 12.5, ?, 'now')",
        (suite_id, fp, item_id, output, run_id))
    db.commit()
    return cur.lastrowid


def _judgment(db, ra, rb, winner, judge="judge-x", template="blind-pair-v2"):
    cur = db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model,"
        " judge_template_version, winner, confidence, rationale, created_at)"
        " VALUES (?, ?, ?, ?, ?, NULL, 'r', 'now')",
        (ra, rb, judge, template, winner))
    db.commit()
    return cur.lastrowid


def test_wilson_interval_basics():
    assert reports.wilson_interval(None, 0) is None
    # p=0.5, n=100 -> roughly 40.7..59.3
    lo, hi = reports.wilson_interval(0.5, 100)
    assert 39.0 <= lo <= 42.0 and 58.0 <= hi <= 61.0
    # small n -> wide interval
    lo2, hi2 = reports.wilson_interval(0.5, 5)
    assert (hi2 - lo2) > (hi - lo)
    # p=0 -> interval hugs zero but isn't exactly zero
    lo3, hi3 = reports.wilson_interval(0.0, 10)
    assert lo3 == 0.0 and 0 < hi3 < 31


def test_by_dimension_is_the_headline_lens(db):
    items = [
        {"id": "r1", "dimension": "reasoning", "prompt": "p1"},
        {"id": "w1", "dimension": "writing", "prompt": "p2"},
        {"id": "r2", "dimension": "reasoning", "prompt": "p3"},
    ]
    sid = _suite(db, items)
    ra1, rb1 = _result(db, sid, "alpha", "r1"), _result(db, sid, "beta", "r1")
    ra2, rb2 = _result(db, sid, "alpha", "w1"), _result(db, sid, "beta", "w1")
    _judgment(db, ra1, rb1, "a")   # reasoning -> alpha
    _judgment(db, ra2, rb2, "b")   # writing -> beta
    rep = reports.suite_report(db, sid)
    assert rep["by_dimension"]["reasoning"]["counts"] == {"a": 1, "b": 0, "tie": 0}
    assert rep["by_dimension"]["writing"]["counts"] == {"a": 0, "b": 1, "tie": 0}
    assert rep["by_dimension"]["reasoning"]["percent"]["a"] == 100.0
    assert rep["by_dimension"]["reasoning"]["wilson_a"] is not None
    # overall still aggregates both
    assert rep["overall"]["counts"] == {"a": 1, "b": 1, "tie": 0}


def test_reference_scores_scored_at_report_time(db):
    items = [
        {"id": "m1", "dimension": "reasoning", "prompt": "p",
         "reference": {"check": "numeric", "value": 42}},
        {"id": "m2", "dimension": "reasoning", "prompt": "p",
         "reference": {"check": "numeric", "value": 7}},
    ]
    sid = _suite(db, items)
    _result(db, sid, "alpha", "m1", "the answer is 42")
    _result(db, sid, "alpha", "m2", "I think it is 9")
    _result(db, sid, "beta", "m1", "42")
    _result(db, sid, "beta", "m2", "7")

    ref = reports.reference_scores(db, sid)
    alpha = ref["models"]["alpha"]
    assert alpha["overall"]["passed"] == 1 and alpha["overall"]["failed"] == 1
    assert alpha["overall"]["pass_pct"] == 50.0
    assert alpha["overall"]["wilson"]["low"] is not None
    assert alpha["by_dimension"]["reasoning"]["total"] == 2
    beta = ref["models"]["beta"]
    assert beta["overall"]["passed"] == 2  # exact numeric matches
    assert ref["reference_items"] == 2
    # embedded in the full report too
    rep = reports.suite_report(db, sid)
    assert rep["reference_scores"]["models"]["beta"]["overall"]["passed"] == 2


def test_reference_scores_retroactive_spec_improvement(db):
    """The ADR 007 §2 promise: an improved reference spec applies
    retroactively — score from a NEWER definition snapshot, stored rows
    untouched."""
    items_v1 = [{"id": "m1", "dimension": "reasoning", "prompt": "p",
                 "reference": {"check": "exact", "value": "42"}}]
    sid = _suite(db, items_v1)
    _result(db, sid, "alpha", "m1", "The answer is 42.")
    # v1 exact match: fails (output has prose)
    assert reports.reference_scores(db, sid)["models"]["alpha"]["overall"]["passed"] == 0
    # store an improved snapshot: regex finds the number anywhere
    db.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (?, ?, 'later')", (sid, json.dumps({"items": [
            {"id": "m1", "dimension": "reasoning", "prompt": "p",
             "reference": {"check": "regex", "pattern": r"\b42\b"}}]})))
    db.commit()
    assert reports.reference_scores(db, sid)["models"]["alpha"]["overall"]["passed"] == 1
    # stored row untouched
    assert db.execute("SELECT output FROM eval_results").fetchone()["output"] \
        == "The answer is 42."


def test_judge_generation_scoping(db):
    sid = _suite(db, [{"id": "q1", "dimension": "reasoning", "prompt": "p"}])
    ra, rb = _result(db, sid, "alpha", "q1"), _result(db, sid, "beta", "q1")
    _judgment(db, ra, rb, "a", judge="judge-old", template="blind-pair-v1")
    _judgment(db, ra, rb, "a", judge="judge-old", template="blind-pair-v1")
    _judgment(db, ra, rb, "b", judge="judge-new", template="blind-pair-v2")
    _judgment(db, ra, rb, "b", judge="judge-new", template="blind-pair-v2")

    rep_all = reports.suite_report(db, sid)
    assert rep_all["judgments_stored"] == 4
    assert rep_all["overall"]["counts"] == {"a": 2, "b": 2, "tie": 0}
    assert len(rep_all["generations"]) == 2

    rep_new = reports.suite_report(db, sid, judge_model="judge-new",
                                   judge_template="blind-pair-v2")
    assert rep_new["judgments_stored"] == 2
    assert rep_new["overall"]["counts"] == {"a": 0, "b": 2, "tie": 0}

    rep_old = reports.suite_report(db, sid, judge_model="judge-old")
    assert rep_old["judgments_stored"] == 2
    assert rep_old["overall"]["counts"]["a"] == 2


def test_metadata_map_defaults_to_uncategorized(db):
    sid = _suite(db, [{"id": "q1", "prompt": "p"}])  # no dimension/category
    meta = reports.metadata_map_for(db, sid)
    assert meta["q1"] == {"category": "uncategorized",
                          "dimension": "uncategorized"}
    # Turn-suffixed rows inherit the honest bucket too.
    assert meta["q1#t2"] == {"category": "uncategorized",
                             "dimension": "uncategorized"}


def test_multiturn_turn_rows_inherit_base_dimension(db):
    """Live-smoke regression: core-reasoning-101#t2 landed under
    'uncategorized' because the metadata map lacked #tN suffix keys.
    Per-turn rows must inherit the base item's dimension/category."""
    sid = _suite(db, [{"id": "mt-1", "prompt": "p", "type": "multi-turn",
                       "dimension": "reasoning", "category": "chain"}])
    dmap = reports.dimension_map_for(db, sid)
    assert dmap["mt-1"] == "reasoning"
    assert dmap["mt-1#t2"] == "reasoning"
    assert dmap["mt-1#t3"] == "reasoning"
    cmap = reports.category_map_for(db, sid)
    assert cmap["mt-1#t2"] == "chain"


def test_flip_detection_via_position_swap_rows(db):
    """The PRD's flip count: the same item judged in both orders may
    disagree; both rows stay and aggregate per row (no derived rows)."""
    sid = _suite(db, [{"id": "q1", "dimension": "reasoning", "prompt": "p"}])
    ra, rb = _result(db, sid, "alpha", "q1"), _result(db, sid, "beta", "q1")
    _judgment(db, ra, rb, "a")
    _judgment(db, ra, rb, "b")  # the other order disagreed
    rep = reports.suite_report(db, sid)
    assert rep["overall"]["counts"] == {"a": 1, "b": 1, "tie": 0}
    # both raw rows visible at item level
    assert len(rep["item_level"]) == 2
