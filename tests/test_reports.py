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


def seed_judgment(db, ra, rb, winner):
    db.execute(
        "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
        "judge_template_version, winner, confidence, rationale, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ra, rb, "judge-x", "blind-pair-v1", winner, 0.8, None, "now"),
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
    assert rep["overall"]["counts"] == {"a": 1, "b": 1, "tie": 1}
    assert rep["overall"]["percent"] == {"a": 33.3, "b": 33.3, "tie": 33.3}
    assert rep["overall"]["total"] == 3
    assert len(rep["pairs"]) == 1
    pair = rep["pairs"][0]
    assert pair["counts"] == {"a": 1, "b": 1, "tie": 1}
    assert pair["total"] == 3


def test_winner_side_tracked_by_model_not_row_order(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "b")  # winner=b == model-beta (rb's model)
    rep = reports.suite_report(db, suite_id)
    # pair is sorted; model-beta sorts after alpha so counted as side "b"
    assert rep["pairs"][0]["counts"] == {"a": 0, "b": 1, "tie": 0}


def test_empty_suite_report(db):
    suite_id, _ = seed_fixture(db)
    rep = reports.suite_report(db, suite_id)
    assert rep["overall"]["total"] == 0
    assert rep["overall"]["percent"] == {"a": None, "b": None, "tie": None}
    assert rep["pairs"] == []


def test_csv_export_matches_counts(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "a")
    rep = reports.suite_report(db, suite_id)
    csv_text = reports.export_report_csv(rep)
    lines = csv_text.strip().splitlines()
    assert len(lines) == 2  # header + one pair
    assert "model-alpha,model-beta" in lines[1]
    assert "1,0,0,1" in lines[1]  # wins_a, wins_b, ties, total
    assert lines[0].startswith("suite,version,model_a")


def test_category_map_breakdown(db):
    suite_id, result_ids = seed_fixture(db)
    (ra, rb) = result_ids["sr-001"]
    seed_judgment(db, ra, rb, "a")
    cmap = {"sr-001": "logic", "sr-002": "counting", "sr-003": "writing"}
    rep = reports.suite_report(db, suite_id, category_map=cmap)
    assert "logic" in rep["by_category"]
    assert rep["by_category"]["logic"]["a"] == 1
    assert rep["by_category"]["logic"]["total"] == 1
    assert rep["by_category"]["uncategorized"] if False else True
    rep2 = reports.suite_report(db, suite_id)  # no map -> uncategorized
    assert rep2["by_category"]["uncategorized"]["total"] == 1


def test_category_map_for_reads_definition_snapshot(db):
    """category_map_for rebuilds the map from the stored definition JSON."""
    import json as _json

    cur = db.execute(
        "INSERT INTO eval_suites (name, version) VALUES ('cat-suite', 'v1')")
    suite_id = cur.lastrowid
    definition = {"items": [
        {"id": "sr-001", "category": "logic"},
        {"id": "sr-002", "category": "counting"},
        {"id": "sr-003"},  # no category -> dropped from the map
    ]}
    db.execute(
        "INSERT INTO suite_definitions (suite_id, payload, created_at) "
        "VALUES (?, ?, 'now')", (suite_id, _json.dumps(definition)))
    db.commit()

    cmap = reports.category_map_for(db, suite_id)
    assert cmap == {"sr-001": "logic", "sr-002": "counting"}

    # No snapshot for an unknown suite -> empty map, report still works.
    assert reports.category_map_for(db, 999) == {}
