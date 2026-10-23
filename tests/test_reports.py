"""J5 report aggregation tests — win rates, categories, exports, edge cases."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos.db import connect, init_db  # noqa: E402
from arcturos import reports  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    conn = connect(tmp_path / "reports_test.db")
    init_db(conn)
    return conn


def seed_fixture(db, suite_name="smoke", version="v1"):
    cur = db.execute(
        "INSERT INTO eval_suites (name, version) VALUES (?, ?)",
        (suite_name, version),
    )
    suite_id = cur.lastrowid

    models = ["model-alpha", "model-beta"]
    items = [("sr-001", "logic"), ("sr-002", "counting"), ("sr-003", "writing")]
    result_ids = {}
    for model in models:
        for item_id, _cat in items:
            cur = db.execute(
                "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, "
                "output, prompt_tokens, completion_tokens, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (suite_id, model, item_id, f"{model} says", 20, 30, 100.0, "now"),
            )
            result_ids.setdefault(item_id, []).append(cur.lastrowid)
    db.commit()
    return suite_id, result_ids


def seed_judgment(db, ra, rb, winner, judge_model="judge-x",
                  judge_template="blind-pair-v1"):
    db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ra, rb, judge_model, judge_template, winner, 0.8, None, "now"),
    )
    db.commit()


def seed_definition(db, suite_id, items):
    """Store a definition snapshot mapping item ids to dimensions."""
    import json as _json

    db.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (?, ?, 'now')",
        (suite_id, _json.dumps({"items": items})),
    )
    db.commit()


def test_unknown_suite_raises(db):
    with pytest.raises(ValueError):
        reports.suite_report(db, 999)


def test_winrate_percentages(db):
    suite_id, result_ids = seed_fixture(db)
    # alpha wins logic, tie on counting, beta wins writing
    (ra1, rb1), (ra2, rb2), (ra3, rb3) = result_ids["sr-001"], result_ids["sr-002"], result_ids["sr-003"]
    seed_judgment(db, ra1, rb1, "a")
    seed_judgment(db, ra2, rb2, "tie")
    seed_judgment(db, ra3, rb3, "b")

    rep = reports.suite_report(db, suite_id)
    assert rep["prompts"] == 3
    assert rep["models"] == ["model-alpha", "model-beta"]
    assert rep["overall"]["counts"] == {"a": 1, "b": 1, "tie": 1, "both_bad": 0}
    assert rep["overall"]["percent"] == {"a": 33.3, "b": 33.3, "tie": 33.3,
                                         "both_bad": 0.0}
    assert rep["overall"]["total"] == 3
    assert len(rep["pairs"]) == 1
    pair = rep["pairs"][0]
    assert pair["counts"] == {"a": 1, "b": 1, "tie": 1, "both_bad": 0}
    assert pair["total"] == 3


def test_winner_side_tracked_by_model_not_row_order(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "b")  # winner=b == model-beta (rb's model)
    rep = reports.suite_report(db, suite_id)
    # pair is sorted; model-beta sorts after alpha so counted as side "b"
    assert rep["pairs"][0]["counts"] == {"a": 0, "b": 1, "tie": 0, "both_bad": 0}


def test_empty_suite_report(db):
    suite_id, _ = seed_fixture(db)
    rep = reports.suite_report(db, suite_id)
    assert rep["overall"]["total"] == 0
    assert rep["overall"]["percent"] == {"a": None, "b": None, "tie": None,
                                         "both_bad": None}
    assert rep["pairs"] == []


def test_csv_export_matches_counts(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "a")
    rep = reports.suite_report(db, suite_id)
    csv_text = reports.export_report_csv(rep)
    lines = csv_text.strip().splitlines()
    # v0.2: pair section + dimension section (header, pair row, blank,
    # dimension header, one dimension row)
    assert len(lines) == 5
    assert "model-alpha,model-beta" in lines[1]
    assert "1,0,0,0,1" in lines[1]  # wins_a, wins_b, ties, both_bad, total
    assert lines[0].startswith("suite,version,model_a")
    assert "wilson_a_low" in lines[0]  # v0.2: Wilson interval columns
    assert "both_bad" in lines[0]  # 2026-10-23: both-wrong bucket exported
    assert lines[3] == "dimension,wins_a,wins_b,ties,both_bad,total"
    assert lines[4].startswith("uncategorized,")  # no definition -> uncategorized


def test_category_map_breakdown(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "a")
    cmap = {"sr-001": "logic", "sr-002": "counting", "sr-003": "writing"}
    dmap = {"sr-001": "reasoning", "sr-002": "reasoning", "sr-003": "writing"}
    rep = reports.suite_report(db, suite_id, category_map=cmap,
                               dimension_map=dmap)
    assert "logic" in rep["by_category"]
    # v0.2 shape: counts nested under "counts", percent under "percent"
    assert rep["by_category"]["logic"]["counts"]["a"] == 1
    assert rep["by_category"]["logic"]["total"] == 1
    assert rep["by_dimension"]["reasoning"]["counts"]["a"] == 1
    assert "uncategorized" not in rep["by_category"]  # fully mapped suite
    rep2 = reports.suite_report(db, suite_id)  # no map -> uncategorized
    assert rep2["by_category"]["uncategorized"]["total"] == 1
    assert rep2["by_dimension"]["uncategorized"]["total"] == 1


def test_category_map_for_reads_definition_snapshot(db):
    """category_map_for rebuilds the map from the stored definition JSON.
    v0.2: items WITHOUT a category land under 'uncategorized' (honest
    bucket) instead of being dropped from the map."""
    import json as _json

    cur = db.execute(
        "INSERT INTO eval_suites (name, version) VALUES ('cat-suite', 'v1')")
    suite_id = cur.lastrowid
    definition = {"items": [
        {"id": "sr-001", "category": "logic"},
        {"id": "sr-002", "category": "counting"},
        {"id": "sr-003"},  # no category -> 'uncategorized' bucket
    ]}
    db.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (?, ?, 'now')", (suite_id, _json.dumps(definition)))
    db.commit()

    cmap = reports.category_map_for(db, suite_id)
    # Multi-turn per-turn rows (#t2+) inherit the base item's category.
    assert cmap["sr-001"] == "logic"
    assert cmap["sr-002"] == "counting"
    assert cmap["sr-003"] == "uncategorized"
    assert cmap["sr-001#t2"] == "logic"
    assert cmap["sr-003#t3"] == "uncategorized"

    # No snapshot for an unknown suite -> empty map, report still works.
    assert reports.category_map_for(db, 999) == {}


# ------------------------------------------------- pair page summary -------


def test_pair_summary_per_dimension_math(db):
    """Exact per-dimension math: counts, percent, score, Wilson present."""
    suite_id, result_ids = seed_fixture(db)
    seed_definition(db, suite_id, [
        {"id": "sr-001", "dimension": "reasoning"},
        {"id": "sr-002", "dimension": "reasoning"},
        {"id": "sr-003", "dimension": "writing"},
    ])
    (ra1, rb1), (ra2, rb2), (ra3, rb3) = (result_ids["sr-001"],
                                          result_ids["sr-002"],
                                          result_ids["sr-003"])
    # canonical rows: ra is always model-alpha (seed_fixture order)
    seed_judgment(db, ra1, rb1, "a")
    seed_judgment(db, ra1, rb1, "a")
    seed_judgment(db, ra1, rb1, "tie")
    seed_judgment(db, ra2, rb2, "b")
    seed_judgment(db, ra3, rb3, "a")

    s = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    assert s["suite_id"] == suite_id
    assert s["model_a"] == "model-alpha"
    assert s["model_b"] == "model-beta"
    assert s["judges"] == ["judge-x"]
    assert s["templates"] == ["blind-pair-v1"]
    assert s["total"] == 5

    # rows sorted n desc, then dimension name asc
    assert [r["dimension"] for r in s["rows"]] == ["reasoning", "writing"]

    reasoning = s["rows"][0]
    assert reasoning["counts"] == {"a": 2, "b": 1, "tie": 1, "both_bad": 0}
    assert reasoning["n"] == 4
    assert reasoning["percent"] == {"a": 50.0, "b": 25.0, "tie": 25.0,
                                    "both_bad": 0.0}
    # Arena score: wins + half ties, 0-100
    assert reasoning["score_a"] == 62.5
    assert reasoning["score_b"] == 37.5
    wilson = reasoning["wilson_a"]          # (lo, hi) in percent units
    assert wilson is not None and len(wilson) == 2
    assert 0.0 <= wilson[0] <= wilson[1] <= 100.0

    writing = s["rows"][1]
    assert writing["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 0}
    assert writing["n"] == 1
    assert writing["percent"] == {"a": 100.0, "b": 0.0, "tie": 0.0,
                                  "both_bad": 0.0}
    assert writing["score_a"] == 100.0
    assert writing["score_b"] == 0.0

    overall = s["overall"]
    assert overall["dimension"] == "overall"
    assert overall["counts"] == {"a": 3, "b": 1, "tie": 1, "both_bad": 0}
    assert overall["n"] == 5
    assert overall["percent"] == {"a": 60.0, "b": 20.0, "tie": 20.0,
                                  "both_bad": 0.0}
    assert overall["score_a"] == 70.0
    assert overall["score_b"] == 30.0
    assert overall["wilson_a"] is not None


def test_pair_summary_orientation_invariant(db):
    """Swapping model_a/model_b mirrors the a/b columns; n/ties/rows stay."""
    suite_id, result_ids = seed_fixture(db)
    (ra1, rb1), (ra2, rb2), (ra3, rb3) = (result_ids["sr-001"],
                                          result_ids["sr-002"],
                                          result_ids["sr-003"])
    seed_judgment(db, ra1, rb1, "a")
    seed_judgment(db, ra1, rb1, "tie")
    seed_judgment(db, ra2, rb2, "b")
    seed_judgment(db, ra3, rb3, "a")

    fwd = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    rev = reports.pair_summary(db, suite_id, "model-beta", "model-alpha")

    assert fwd["total"] == rev["total"] == 4
    assert [r["dimension"] for r in fwd["rows"]] == \
        [r["dimension"] for r in rev["rows"]]
    for f, r in zip(fwd["rows"] + [fwd["overall"]],
                    rev["rows"] + [rev["overall"]]):
        assert r["counts"] == {"a": f["counts"]["b"], "b": f["counts"]["a"],
                               "tie": f["counts"]["tie"],
                               "both_bad": f["counts"].get("both_bad", 0)}
        assert r["n"] == f["n"]
        assert r["percent"] == {"a": f["percent"]["b"],
                                "b": f["percent"]["a"],
                                "tie": f["percent"]["tie"],
                                "both_bad": f["percent"].get("both_bad")}
        assert r["score_a"] == f["score_b"]
        assert r["score_b"] == f["score_a"]
        # the interval tracks A's win share, so it mirrors with the columns
        assert r["wilson_a"] == reports.wilson_interval(
            f["counts"]["b"] / f["n"], f["n"])
    assert rev["judges"] == fwd["judges"]
    assert rev["templates"] == fwd["templates"]


def test_pair_summary_judge_filter_scoping(db):
    """judge_model / judge_template scope counts AND the judges/templates lists."""
    suite_id, result_ids = seed_fixture(db)
    (ra1, rb1), (ra2, rb2) = result_ids["sr-001"], result_ids["sr-002"]
    seed_judgment(db, ra1, rb1, "a")                                   # judge-x / v1
    seed_judgment(db, ra1, rb1, "b", judge_model="judge-y")            # v1
    seed_judgment(db, ra2, rb2, "a", judge_model="judge-y",
                  judge_template="blind-pair-v2")

    base = (suite_id, "model-alpha", "model-beta")
    all_gen = reports.pair_summary(db, *base)
    assert all_gen["total"] == 3
    assert all_gen["judges"] == ["judge-x", "judge-y"]
    assert all_gen["templates"] == ["blind-pair-v1", "blind-pair-v2"]

    y_only = reports.pair_summary(db, *base, judge_model="judge-y")
    assert y_only["total"] == 2
    assert y_only["overall"]["counts"] == {"a": 1, "b": 1, "tie": 0, "both_bad": 0}
    assert y_only["judges"] == ["judge-y"]
    assert y_only["templates"] == ["blind-pair-v1", "blind-pair-v2"]

    v2_only = reports.pair_summary(db, *base, judge_template="blind-pair-v2")
    assert v2_only["total"] == 1
    assert v2_only["overall"]["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 0}
    assert v2_only["templates"] == ["blind-pair-v2"]

    both = reports.pair_summary(db, *base, judge_model="judge-y",
                                judge_template="blind-pair-v2")
    assert both["total"] == 1
    assert both["overall"]["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 0}

    unknown_judge = reports.pair_summary(db, *base, judge_model="judge-z")
    assert unknown_judge["total"] == 0
    assert unknown_judge["rows"] == []


def test_pair_summary_multiturn_rows_fold_into_base_dimension(db):
    """'#t2' turn rows inherit the base item's dimension (no # rows leak)."""
    suite_id, result_ids = seed_fixture(db)
    seed_definition(db, suite_id, [{"id": "sr-001", "dimension": "reasoning"},
                                   {"id": "sr-002", "dimension": "reasoning"},
                                   {"id": "sr-003", "dimension": "writing"}])
    ra_t2 = db.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, "
        "output, prompt_tokens, completion_tokens, latency_ms, created_at) "
        "VALUES (?, 'model-alpha', 'sr-001#t2', 'turn out', 20, 30, 1.0, 'now')",
        (suite_id,)).lastrowid
    rb_t2 = db.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, "
        "output, prompt_tokens, completion_tokens, latency_ms, created_at) "
        "VALUES (?, 'model-beta', 'sr-001#t2', 'turn out', 20, 30, 1.0, 'now')",
        (suite_id,)).lastrowid
    db.commit()
    (ra1, rb1), (ra3, rb3) = result_ids["sr-001"], result_ids["sr-003"]
    seed_judgment(db, ra1, rb1, "a")
    seed_judgment(db, ra_t2, rb_t2, "b")
    seed_judgment(db, ra3, rb3, "a")               # writing stays its own row

    s = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    assert [r["dimension"] for r in s["rows"]] == ["reasoning", "writing"]
    reasoning = s["rows"][0]
    assert reasoning["n"] == 2                      # base + #t2 folded in
    assert reasoning["counts"] == {"a": 1, "b": 1, "tie": 0, "both_bad": 0}
    assert s["rows"][1]["n"] == 1
    assert all("#" not in r["dimension"] for r in s["rows"])


def test_pair_summary_uncategorized_fallback(db):
    """No definition snapshot -> every judgment lands in 'uncategorized'."""
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "a")

    s = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    assert s["total"] == 1
    assert [r["dimension"] for r in s["rows"]] == ["uncategorized"]
    assert s["rows"][0]["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 0}
    assert s["overall"]["n"] == 1


def test_pair_summary_empty_pair_honest(db):
    """Unknown pair -> rows [], overall n=0, null percents, zero scores."""
    suite_id, _ = seed_fixture(db)

    s = reports.pair_summary(db, suite_id, "model-none", "model-other")
    assert s["rows"] == []
    assert s["total"] == 0
    assert s["judges"] == []
    assert s["templates"] == []
    o = s["overall"]
    assert o["dimension"] == "overall"
    assert o["counts"] == {"a": 0, "b": 0, "tie": 0, "both_bad": 0}
    assert o["n"] == 0
    assert o["percent"] == {"a": None, "b": None, "tie": None,
                            "both_bad": None}
    assert o["wilson_a"] is None
    assert o["score_a"] == 0
    assert o["score_b"] == 0


def test_both_bad_counts_as_own_bucket_and_half_draw(db):
    """both-bad rows (winner='tie' + verdict_detail='both_bad') keep the
    0.5/0.5 draw in score math but surface as their own bucket + rate."""
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at, "
        "verdict_detail) VALUES (?, ?, 'judge-x', 'blind-pair-v4-ref', "
        "'tie', 0.9, NULL, 'now', 'both_bad')", (ra, rb))
    db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at) "
        "VALUES (?, ?, 'judge-x', 'blind-pair-v1', 'a', 0.9, NULL, 'now')",
        (ra, rb))
    db.commit()
    rep = reports.suite_report(db, suite_id)
    assert rep["overall"]["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 1}
    assert rep["overall"]["both_bad_rate"] == 50.0
    pair = rep["pairs"][0]
    assert pair["counts"]["both_bad"] == 1
    assert pair["both_bad_rate"] == 50.0
    # score math: a = 1 win + 0.5*draw = 1.5 of 2 -> 75.0
    assert pair["percent"]["a"] == 50.0
    # pair_summary lens: same bucket split + draw semantics
    s = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    assert s["overall"]["counts"] == {"a": 1, "b": 0, "tie": 0, "both_bad": 1}
    assert s["overall"]["both_bad_rate"] == 50.0
    assert s["overall"]["score_a"] == 75.0
    assert s["overall"]["score_b"] == 25.0
    # a plain tie stays in the tie bucket (no detail -> not both-bad)
    db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at) "
        "VALUES (?, ?, 'judge-x', 'blind-pair-v1', 'tie', 0.9, NULL, 'now')",
        (ra, rb))
    db.commit()
    s2 = reports.pair_summary(db, suite_id, "model-alpha", "model-beta")
    assert s2["overall"]["counts"] == {"a": 1, "b": 0, "tie": 1, "both_bad": 1}
    assert s2["overall"]["score_a"] == round(100 * (1 + 0.5 * 1 + 0.5 * 1) / 3, 1)
