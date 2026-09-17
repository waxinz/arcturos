"""J5 — eval performance reports.

Aggregates stored eval_results + judgments into suite-level win-rate reports:
per suite × model-pair counts and percentages (a/b/tie), per-category
breakdown, and exportable dicts. Read-only over the append-only store.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def _fetch(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return db.execute(sql, params).fetchall()


def suite_report(db: sqlite3.Connection, suite_id: int,
                 category_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Win-rate report for one suite.

    category_map maps item_id -> category (from the suite definition JSON;
    the DB stores item_id only, so categories are joined at report time).
    Items without a mapping land under "uncategorized".

    Returns {"suite": {...}, "prompts": N, "models": [...], "pairs": [...],
             "overall": {...}, "by_category": {...}, "item_level": [...]}.
    Raises ValueError for unknown suite_id.
    """
    suite = _fetch(db, "SELECT * FROM eval_suites WHERE id = ?", (suite_id,))
    if not suite:
        raise ValueError(f"unknown suite_id {suite_id}")
    suite = dict(suite[0])

    results = _fetch(
        db,
        "SELECT * FROM eval_results WHERE suite_id = ? ORDER BY id",
        (suite_id,),
    )
    result_ids = {r["id"]: r for r in results}

    judgments = _fetch(
        db,
        """SELECT j.* FROM judgments j
           JOIN eval_results ra ON ra.id = j.eval_result_a
           JOIN eval_results rb ON rb.id = j.eval_result_b
           WHERE ra.suite_id = ? ORDER BY j.id""",
        (suite_id,),
    )

    overall = {"a": 0, "b": 0, "tie": 0}
    per_pair: dict[tuple[str, str], dict[str, Any]] = {}
    per_category: dict[str, dict[str, Any]] = {}
    item_level: list[dict[str, Any]] = []

    for j in judgments:
        ra = result_ids.get(j["eval_result_a"])
        rb = result_ids.get(j["eval_result_b"])
        if ra is None or rb is None:
            continue  # defensive: judgment rows are FK-validated, but stay robust
        pair = tuple(sorted((ra["model_fingerprint"], rb["model_fingerprint"])))
        entry = per_pair.setdefault(
            pair, {"model_a": pair[0], "model_b": pair[1], "a": 0, "b": 0, "tie": 0}
        )
        # winner is relative to stored eval_result_a/eval_result_b, not sorted pair
        winner = j["winner"]
        if winner == "a":
            counted_model = ra["model_fingerprint"]
        elif winner == "b":
            counted_model = rb["model_fingerprint"]
        else:
            counted_model = None
        counted_side = None
        if counted_model is not None:
            counted_side = "a" if counted_model == pair[0] else "b"
        if counted_side:
            entry[counted_side] += 1
        else:
            entry["tie"] += 1
        overall[winner] += 1

        item = dict(ra["item_id"] and {})  # placeholder to keep shape simple
        item = {
            "item_id": ra["item_id"],
            "judgment_id": j["id"],
            "winner": winner,
            "model_a": ra["model_fingerprint"],
            "model_b": rb["model_fingerprint"],
            "judge_model": j["judge_model"],
            "template": j["judge_template_version"],
            "confidence": j["confidence"],
            "rationale": j["rationale"],
        }
        item_level.append(item)

        category = (category_map or {}).get(
            ra["item_id"], "uncategorized"
        )
        cat_entry = per_category.setdefault(
            category, {"a": 0, "b": 0, "tie": 0, "total": 0}
        )
        if counted_side:
            cat_entry[counted_side] += 1
        else:
            cat_entry["tie"] += 1
        cat_entry["total"] += 1

    total = sum(overall.values())
    overall_pct = {
        k: round(v / total * 100, 1) if total else None for k, v in overall.items()
    }

    pairs_out = []
    for (ma, mb), entry in per_pair.items():
        t = entry["a"] + entry["b"] + entry["tie"]
        pairs_out.append({
            "model_a": ma, "model_b": mb,
            "counts": {k: entry[k] for k in ("a", "b", "tie")},
            "percent": {k: round(entry[k] / t * 100, 1) if t else None
                        for k in ("a", "b", "tie")},
            "total": t,
        })

    return {
        "suite": suite,
        "prompts": len({r["item_id"] for r in results}),
        "models": sorted({r["model_fingerprint"] for r in results}),
        "results_stored": len(results),
        "judgments_stored": len(judgments),
        "pairs": pairs_out,
        "overall": {"counts": overall, "percent": overall_pct, "total": total},
        "by_category": per_category,
        "item_level": item_level,
    }


def export_report_csv(report: dict[str, Any]) -> str:
    """CSV lines: suite-level pair win rates. Header row included."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["suite", "version", "model_a", "model_b",
                     "wins_a", "wins_b", "ties", "total",
                     "pct_a", "pct_b", "pct_tie"])
    for pair in report["pairs"]:
        writer.writerow([
            report["suite"]["name"], report["suite"]["version"],
            pair["model_a"], pair["model_b"],
            pair["counts"]["a"], pair["counts"]["b"], pair["counts"]["tie"],
            pair["total"],
            pair["percent"]["a"], pair["percent"]["b"], pair["percent"]["tie"],
        ])
    return buf.getvalue()


def export_report_json(report: dict[str, Any]) -> dict[str, Any]:
    """Report is already JSON-safe; hook for future sanitization."""
    return report
