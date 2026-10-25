"""J5 — eval performance reports (v0.2).

Aggregates stored eval_results + judgments into suite-level reports:
overall + per-pair + per-DIMENSION win rates (the headline lens — "which
model is better on which axes"), deterministic reference scores (pass
rates needing no judge), judge-generation scoping, item-level drill-down,
and Wilson 95% intervals on every rate. Read-only over the append-only
store.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any


def _fetch(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return db.execute(sql, params).fetchall()


def wilson_interval(p: float | None, n: int,
                    z: float = 1.96) -> tuple[float, float] | None:
    """Wilson 95% score interval for a proportion — closed form, no
    bootstrap needed at eval scale. Returns None when n == 0."""
    if not n:
        return None
    if p is None:
        p = 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - spread) * 100, 1),
            round(min(1.0, centre + spread) * 100, 1))


def metadata_map_for(db: sqlite3.Connection, suite_id: int) -> dict[str, dict[str, str]]:
    """Rebuild the item_id -> metadata map for one suite.

    Dimensions and categories live in the suite definition JSON (the
    append-only DB stores item_id only), so the report-time join reads the
    latest definition snapshot from suite_definitions. Absent metadata ->
    the item lands under {"category": "uncategorized",
    "dimension": "uncategorized"} — honest, never fabricated.
    """
    suite = _latest_definition(db, suite_id)
    items = suite.get("items") or []
    out: dict[str, dict[str, str]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = {
                "category": item.get("category") or "uncategorized",
                "dimension": item.get("dimension") or "uncategorized",
            }
            # Multi-turn replays store per-assistant-turn rows with #t2/#t3
            # suffixes on the item id; they inherit the base item's metadata.
            base = str(item["id"])
            for n in range(2, 9):  # matches multiturn.py MAX_TURNS bound
                out[f"{base}#t{n}"] = out[base]
    return out


def category_map_for(db: sqlite3.Connection, suite_id: int) -> dict[str, str]:
    """Back-compat wrapper: item_id -> category only."""
    return {item_id: meta["category"]
            for item_id, meta in metadata_map_for(db, suite_id).items()}


def dimension_map_for(db: sqlite3.Connection, suite_id: int) -> dict[str, str]:
    """item_id -> dimension (the v0.2 headline axis)."""
    return {item_id: meta["dimension"]
            for item_id, meta in metadata_map_for(db, suite_id).items()}


def _latest_definition(db: sqlite3.Connection, suite_id: int) -> dict[str, Any]:
    try:
        row = db.execute(
            "SELECT payload FROM suite_definitions WHERE suite_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (suite_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return {}
    if row is None or not row["payload"]:
        return {}
    try:
        return json.loads(row["payload"])
    except (TypeError, ValueError):
        return {}


def reference_scores(db: sqlite3.Connection, suite_id: int,
                     metadata: dict[str, dict[str, str]] | None = None,
                     definition: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic scores for reference-carrying items (ADR 007 §2).

    Scored at REPORT time from the newest definition snapshot against
    stored outputs: {model: {"overall": {passed, failed, total, pass_pct,
    wilson}, "by_dimension": {dim: {...}}}, "items": [{item_id, model,
    passed, detail}...]}. Models with no reference-item results are absent.
    """
    from arcturos import evals as evals_mod

    suite = definition if definition is not None else _latest_definition(db, suite_id)
    refs = {i["id"]: i for i in (suite.get("items") or [])
            if isinstance(i, dict) and i.get("id") and i.get("reference")}
    if not refs:
        return {"models": {}, "items": [], "reference_items": 0}
    metadata = metadata if metadata is not None else metadata_map_for(db, suite_id)

    results = _fetch(db, "SELECT * FROM eval_results WHERE suite_id = ? ORDER BY id",
                     (suite_id,))
    # latest row per (item, fingerprint); turn rows fold into the base item
    latest: dict[tuple[str, str], sqlite3.Row] = {}
    for r in results:
        base_id = r["item_id"].split("#")[0]
        if base_id not in refs:
            continue
        latest[(base_id, r["model_fingerprint"])] = r

    models: dict[str, dict[str, Any]] = {}
    items_out: list[dict[str, Any]] = []
    for (item_id, fp), r in sorted(latest.items()):
        item = refs[item_id]
        passed, detail = evals_mod.score_output(r["output"], item["reference"])
        dimension = (metadata.get(item_id) or {}).get("dimension", "uncategorized")
        m = models.setdefault(fp, {"overall": {"passed": 0, "failed": 0},
                                   "by_dimension": {}})
        bucket = m["overall"]
        d = m["by_dimension"].setdefault(dimension, {"passed": 0, "failed": 0})
        key = "passed" if passed else "failed"
        bucket[key] += 1
        d[key] += 1
        items_out.append({
            "item_id": item_id, "model": fp, "passed": passed,
            "detail": detail, "dimension": dimension,
        })

    for m in models.values():
        for bucket in [m["overall"], *m["by_dimension"].values()]:
            total = bucket["passed"] + bucket["failed"]
            bucket["total"] = total
            bucket["pass_pct"] = (round(bucket["passed"] / total * 100, 1)
                                  if total else None)
            lo_hi = wilson_interval(
                bucket["passed"] / total if total else None, total)
            if lo_hi:
                bucket["wilson"] = {"low": lo_hi[0], "high": lo_hi[1]}

    return {"models": models, "items": items_out, "reference_items": len(refs)}


def _pct_entry(counts: dict[str, int]) -> dict[str, Any]:
    """counts {a,b,tie,both_bad} -> {counts, percent, total, wilson}.

    both_bad rides on the tie/draw semantics (0.5/0.5 — Arena/MT-Bench
    convention) but is counted and reported as its own bucket so the
    both-wrong rate stays visible next to the win rates.
    """
    total = (counts["a"] + counts["b"] + counts["tie"]
             + counts.get("both_bad", 0))
    keys = ("a", "b", "tie", "both_bad")
    pct = {k: round(counts.get(k, 0) / total * 100, 1) if total else None
           for k in keys}
    lo_hi = wilson_interval(counts["a"] / total if total else None, total)
    out: dict[str, Any] = {"counts": dict(counts), "percent": pct, "total": total}
    if lo_hi:
        # the interval applies to model a's win share (the headline number)
        out["wilson_a"] = {"low": lo_hi[0], "high": lo_hi[1]}
    return out


def suite_report(db: sqlite3.Connection, suite_id: int,
                 category_map: dict[str, str] | None = None,
                 dimension_map: dict[str, str] | None = None,
                 judge_model: str | None = None,
                 judge_template: str | None = None) -> dict[str, Any]:
    """Win-rate report for one suite.

    category_map / dimension_map map item_id -> category / dimension (from
    the suite definition JSON; the DB stores item_id only, so they join at
    report time). Unmapped items land under "uncategorized".

    ``judge_model`` + ``judge_template`` scope every aggregate to ONE judge
    generation (ADR 007 §6); both must match when both are given, either
    alone filters on that field alone. Default (both None): ALL judgments.

    Returns {"suite", "prompts", "models", "pairs", "overall", "by_category",
             "by_dimension", "generations", "reference_scores",
             "judgments_stored", "item_level"}.
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
    if judge_model:
        judgments = [j for j in judgments if j["judge_model"] == judge_model]
    if judge_template:
        judgments = [j for j in judgments
                     if j["judge_template_version"] == judge_template]

    if category_map is None:
        category_map = {k: v["category"]
                        for k, v in metadata_map_for(db, suite_id).items()}
    if dimension_map is None:
        dimension_map = {k: v["dimension"]
                         for k, v in metadata_map_for(db, suite_id).items()}

    overall = {"a": 0, "b": 0, "tie": 0, "both_bad": 0}
    per_pair: dict[tuple[str, str], dict[str, Any]] = {}
    per_category: dict[str, dict[str, Any]] = {}
    per_dimension: dict[str, dict[str, Any]] = {}
    item_level: list[dict[str, Any]] = []
    flips: dict[str, int] = {}

    for j in judgments:
        ra = result_ids.get(j["eval_result_a"])
        rb = result_ids.get(j["eval_result_b"])
        if ra is None or rb is None:
            continue  # defensive: judgment rows are FK-validated, but stay robust
        pair = tuple(sorted((ra["model_fingerprint"], rb["model_fingerprint"])))
        entry = per_pair.setdefault(
            pair, {"model_a": pair[0], "model_b": pair[1],
                   "counts": {"a": 0, "b": 0, "tie": 0, "both_bad": 0}})
        # winner is relative to stored eval_result_a/eval_result_b, not sorted
        # pair; both-bad draws (winner='tie' + verdict_detail='both_bad')
        # count as their own bucket — still 0.5/0.5 in the score math.
        winner = j["winner"]
        is_both_bad = (winner == "tie"
                       and (j["verdict_detail"] or "") == "both_bad")
        counted_model = None
        if winner == "a":
            counted_model = ra["model_fingerprint"]
        elif winner == "b":
            counted_model = rb["model_fingerprint"]
        counted_side = None
        if counted_model is not None:
            counted_side = "a" if counted_model == pair[0] else "b"
        if counted_side:
            entry["counts"][counted_side] += 1
        elif is_both_bad:
            entry["counts"]["both_bad"] += 1
        else:
            entry["counts"]["tie"] += 1
        overall["both_bad" if is_both_bad else winner] += 1

        base_item_id = ra["item_id"].split("#")[0]
        category = category_map.get(ra["item_id"], "uncategorized")
        dimension = dimension_map.get(ra["item_id"], "uncategorized")
        item = {
            "item_id": ra["item_id"],
            "judgment_id": j["id"],
            "winner": winner,
            "verdict_detail": j["verdict_detail"],
            "model_a": ra["model_fingerprint"],
            "model_b": rb["model_fingerprint"],
            "judge_model": j["judge_model"],
            "template": j["judge_template_version"],
            "confidence": j["confidence"],
            "rationale": j["rationale"],
            "category": category,
            "dimension": dimension,
        }
        item_level.append(item)

        for key, bucket in ((category, per_category), (dimension, per_dimension)):
            b = bucket.setdefault(key, {"counts": {"a": 0, "b": 0, "tie": 0,
                                                   "both_bad": 0}})
            if counted_side:
                b["counts"][counted_side] += 1
            elif is_both_bad:
                b["counts"]["both_bad"] += 1
            else:
                b["counts"]["tie"] += 1

    total = sum(overall.values())
    overall_pct = {
        k: round(v / total * 100, 1) if total else None for k, v in overall.items()
    }
    lo_hi = wilson_interval(overall["a"] / total if total else None, total)

    pairs_out = []
    for (ma, mb), entry in per_pair.items():
        counts = entry["counts"]
        t = (counts["a"] + counts["b"] + counts["tie"]
             + counts.get("both_bad", 0))
        pct = {k: round(counts.get(k, 0) / t * 100, 1) if t else None
               for k in ("a", "b", "tie", "both_bad")}
        w = wilson_interval(counts["a"] / t if t else None, t)
        pairs_out.append({
            "model_a": ma, "model_b": mb,
            "counts": dict(counts),
            "percent": pct,
            "total": t,
            "both_bad_rate": pct["both_bad"],
            **({"wilson_a": {"low": w[0], "high": w[1]}} if w else {}),
        })

    # generations: distinct judge generations with counts (ALL judgments,
    # pre-filter — the filter UI needs the full list)
    all_judgments = _fetch(
        db,
        """SELECT j.judge_model, j.judge_template_version, COUNT(*) AS n,
                  MIN(j.id) AS first_id
           FROM judgments j
           JOIN eval_results ra ON ra.id = j.eval_result_a
           WHERE ra.suite_id = ?
           GROUP BY j.judge_model, j.judge_template_version
           ORDER BY first_id DESC""",
        (suite_id,),
    )
    generations = [{"judge_model": g["judge_model"],
                    "template": g["judge_template_version"],
                    "count": g["n"]} for g in all_judgments]

    by_category = {k: _pct_entry(v["counts"]) for k, v in per_category.items()}
    by_dimension = {k: _pct_entry(v["counts"]) for k, v in per_dimension.items()}

    return {
        "suite": suite,
        "prompts": len({r["item_id"].split("#")[0] for r in results}),
        "models": sorted({r["model_fingerprint"] for r in results}),
        "results_stored": len(results),
        "judgments_stored": len(judgments),
        "pairs": pairs_out,
        "overall": {"counts": overall, "percent": overall_pct, "total": total,
                    "both_bad_rate": overall_pct.get("both_bad"),
                    **({"wilson_a": {"low": lo_hi[0], "high": lo_hi[1]}}
                       if lo_hi else {})},
        "by_category": by_category,
        "by_dimension": by_dimension,
        "generations": generations,
        "reference_scores": reference_scores(
            db, suite_id,
            metadata={k: {"category": category_map.get(k, "uncategorized"),
                          "dimension": dimension_map.get(k, "uncategorized")}
                      for k in set(category_map) | set(dimension_map)},
            definition=_latest_definition(db, suite_id)),
        "item_level": item_level,
    }


def pair_summary(db: sqlite3.Connection, suite_id: int, model_a: str,
                 model_b: str, judge_model: str | None = None,
                 judge_template: str | None = None,
                 eval_run_a: int | None = None,
                 eval_run_b: int | None = None) -> dict[str, Any]:
    """Per-dimension win-rate summary for ONE suite x model pair.

    Pair-page lens over the same judgment scope as /api/judgments/pair:
    rows aggregate by the base item's dimension (multi-turn '#tN' rows
    fold into the base item; absent metadata -> 'uncategorized').

    Orientation is normalized to the CALLER's order: the stored winner is
    relative to the canonical eval_result_a/b columns, so when the row's
    fingerprint_a is not the caller's ``model_a`` the verdict is flipped
    ('a'<->'b', ties stay ties). In the response, 'a' ALWAYS means the
    caller's model_a won — calling with the pair swapped mirrors the
    a/b columns, it never changes n, ties, or the row set.

    ``judge_model`` / ``judge_template`` scope the aggregate to ONE judge
    generation when truthy (either field alone filters on that field).
    ``eval_run_a`` + ``eval_run_b`` (2026-10-25) scope the aggregate to
    judgments whose two stored results came from exactly that replay-run
    pair (either orientation): without it the lens is suite x model
    LIFETIME-wide, so a re-run + re-judge of a fixed replay still shows
    the old poisoned batch's verdicts mixed in (run 16's healthy batch
    was buried under run 15's degraded one — accuracy read 65:35 qwen
    when the fresh batch was 20/20 ties). Absent -> lifetime, unchanged.
    Unknown pair -> honest empty: rows [], overall n=0 (no fabricated
    numbers). The Wilson 95% interval applies to model a's win share.

    Returns {"suite_id", "model_a", "model_b", "run_a", "run_b",
             "judges", "templates", "total", "rows", "overall"}; rows
             sorted n desc then dimension name asc. run_a/run_b echo the
             scope (nulls when lifetime-wide).
    """
    sql = """SELECT j.winner, j.judge_model, j.verdict_detail,
                    j.judge_template_version AS template,
                    ra.model_fingerprint AS fa, rb.model_fingerprint AS fb,
                    ra.item_id AS item_id
             FROM judgments j
             JOIN eval_results ra ON ra.id = j.eval_result_a
             JOIN eval_results rb ON rb.id = j.eval_result_b
             WHERE ra.suite_id = ?
               AND ((ra.model_fingerprint = ? AND rb.model_fingerprint = ?)
                 OR (ra.model_fingerprint = ? AND rb.model_fingerprint = ?))"""
    params: list[Any] = [suite_id, model_a, model_b, model_b, model_a]
    if judge_model:
        sql += " AND j.judge_model = ?"
        params.append(judge_model)
    if judge_template:
        sql += " AND j.judge_template_version = ?"
        params.append(judge_template)
    if eval_run_a is not None and eval_run_b is not None:
        sql += (" AND ((ra.eval_run_id = ? AND rb.eval_run_id = ?)"
                " OR (ra.eval_run_id = ? AND rb.eval_run_id = ?))")
        params += [eval_run_a, eval_run_b, eval_run_b, eval_run_a]
    rows = _fetch(db, sql, tuple(params))
    metadata = metadata_map_for(db, suite_id)

    flip = {"a": "b", "b": "a"}
    overall = {"a": 0, "b": 0, "tie": 0, "both_bad": 0}
    per_dim: dict[str, dict[str, int]] = {}
    judges: set[str] = set()
    templates: set[str] = set()
    for r in rows:
        winner: str = r["winner"]
        if r["fa"] != model_a:  # stored canonical order != caller order
            winner = flip.get(winner, winner)
        winner = winner if winner in ("a", "b", "tie") else "tie"
        # both-bad draws (winner='tie' + verdict_detail='both_bad') count as
        # their own bucket; in the score math they still ride the 0.5/0.5
        # draw semantics (Arena/MT-Bench convention).
        is_both_bad = (winner == "tie"
                       and (r["verdict_detail"] or "") == "both_bad")
        bucket_key = "both_bad" if is_both_bad else winner
        dimension = (metadata.get(r["item_id"].split("#")[0], {})
                     ).get("dimension", "uncategorized")
        bucket = per_dim.setdefault(dimension, {"a": 0, "b": 0, "tie": 0,
                                                "both_bad": 0})
        bucket[bucket_key] += 1
        overall[bucket_key] += 1
        judges.add(r["judge_model"])
        templates.add(r["template"])

    def _row(dimension: str, counts: dict[str, int]) -> dict[str, Any]:
        n = (counts["a"] + counts["b"] + counts["tie"]
             + counts.get("both_bad", 0))
        draw_half = 0.5 * (counts["tie"] + counts.get("both_bad", 0))
        return {
            "dimension": dimension,
            "counts": dict(counts),
            "percent": {k: (round(counts.get(k, 0) / n * 100, 1) if n else None)
                        for k in ("a", "b", "tie", "both_bad")},
            "n": n,
            "wilson_a": wilson_interval(counts["a"] / n if n else None, n),
            "score_a": round(100 * (counts["a"] + draw_half) / n, 1)
                       if n else 0,
            "score_b": round(100 * (counts["b"] + draw_half) / n, 1)
                       if n else 0,
            "both_bad_rate": round(100 * counts.get("both_bad", 0) / n, 1)
                             if n else 0,
        }

    dim_rows = [ _row(d, c) for d, c in
                 sorted(per_dim.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])) ]
    overall_row = _row("overall", overall)
    return {
        "suite_id": suite_id,
        "model_a": model_a,
        "model_b": model_b,
        "run_a": eval_run_a,
        "run_b": eval_run_b,
        "judges": sorted(judges),
        "templates": sorted(templates),
        "total": overall_row["n"],
        "rows": dim_rows,
        "overall": overall_row,
    }


def export_report_csv(report: dict[str, Any]) -> str:
    """CSV lines: suite-level pair win rates. Header row included."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["suite", "version", "model_a", "model_b",
                     "wins_a", "wins_b", "ties", "both_bad", "total",
                     "pct_a", "pct_b", "pct_tie", "pct_both_bad",
                     "wilson_a_low", "wilson_a_high"])
    for pair in report["pairs"]:
        wilson = pair.get("wilson_a") or {}
        writer.writerow([
            report["suite"]["name"], report["suite"]["version"],
            pair["model_a"], pair["model_b"],
            pair["counts"]["a"], pair["counts"]["b"], pair["counts"]["tie"],
            pair["counts"].get("both_bad", 0),
            pair["total"],
            pair["percent"]["a"], pair["percent"]["b"], pair["percent"]["tie"],
            pair["percent"].get("both_bad"),
            wilson.get("low", ""), wilson.get("high", ""),
        ])
    # dimension rows appended (v0.2 headline lens)
    writer.writerow([])
    writer.writerow(["dimension", "wins_a", "wins_b", "ties", "both_bad",
                     "total"])
    for dim, entry in sorted(report.get("by_dimension", {}).items()):
        writer.writerow([dim, entry["counts"]["a"], entry["counts"]["b"],
                         entry["counts"]["tie"],
                         entry["counts"].get("both_bad", 0),
                         entry["total"]])
    return buf.getvalue()


def export_report_json(report: dict[str, Any]) -> dict[str, Any]:
    """Report is already JSON-safe; hook for future sanitization."""
    return report
