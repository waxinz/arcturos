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
import statistics
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


# ---- performance dimension (the second eval lens, 2026-11-03) ----------
# Quality answers "who was preferred"; performance answers "who finished
# the work faster". Both derive from the SAME stored rows (eval_results
# carries latency_ms + token counts per replayed item), so performance
# aggregates at REPORT time — no schema change, append-only store
# untouched. Presentation follows the public leaderboards (Artificial
# Analysis: intelligence next to output speed + latency; ML.ENERGY:
# accuracy next to energy per response; Aider: percent correct next to
# cost): quality and speed stay SEPARATE dimensions over the same unit
# of work, latency summarizes as the MEDIAN per item (robust to the few
# slow outliers shared servers produce), and speed never folds into the
# quality score.


def fold_item_perf(rows) -> dict[str, dict[str, Any]]:
    """Fold eval-result rows to per-item performance.

    One single-turn row = one item; a multi-turn item's '#t2/#t3…' turn
    rows SUM into the base item (the model completes the item across
    turns, so the item's latency is the turn sum and its token counts
    are the turn totals). Returns {base_item_id: {latency_ms,
    completion_tokens, prompt_tokens, turns}}.
    """
    items: dict[str, dict[str, Any]] = {}
    for r in rows:
        base = str(r["item_id"]).split("#")[0]
        it = items.setdefault(base, {"latency_ms": 0.0,
                                     "completion_tokens": 0,
                                     "prompt_tokens": 0,
                                     "turns": 0})
        it["latency_ms"] += r["latency_ms"] or 0.0
        it["completion_tokens"] += r["completion_tokens"] or 0
        it["prompt_tokens"] += r["prompt_tokens"] or 0
        it["turns"] += 1
    return items


def perf_stats(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate folded per-item perf into one stats dict.

    output_tps = total completion tokens / total latency seconds — None
    when the server reported no usage (qwen-style completion_tokens=0
    rows) or no latency. An absent metric stays absent, never zero.
    """
    lats = [it["latency_ms"] for it in items.values() if it["latency_ms"]]
    total_ms = sum(it["latency_ms"] for it in items.values())
    total_ctok = sum(it["completion_tokens"] for it in items.values())
    total_ptok = sum(it["prompt_tokens"] for it in items.values())
    turns = sum(it["turns"] for it in items.values())
    tps = (round(total_ctok / (total_ms / 1000.0), 2)
           if total_ctok and total_ms > 0 else None)
    return {
        "items": len(items),
        "turns": turns,
        "total_latency_ms": round(total_ms, 1),
        "median_item_latency_ms": (round(statistics.median(lats), 1)
                                   if lats else None),
        "mean_item_latency_ms": (round(total_ms / len(items), 1)
                                 if items else None),
        "total_prompt_tokens": total_ptok,
        "total_completion_tokens": total_ctok,
        "output_tps": tps,
    }


def speed_comparison(median_a_ms: float | None,
                     median_b_ms: float | None) -> dict[str, Any]:
    """Speed verdict from two per-item medians (the pair lens).

    ratio = slower/faster (>= 1). 'parity' when the slower side sits
    within 1.1x of the faster one — below that the difference reads as
    noise on shared servers, so NO side is named. Honest Nones when
    either median is missing.
    """
    if not median_a_ms or not median_b_ms:
        return {"faster": None, "ratio": None, "band": None}
    ratio = max(median_a_ms, median_b_ms) / min(median_a_ms, median_b_ms)
    if ratio <= 1.1:
        return {"faster": None, "ratio": round(ratio, 2), "band": "parity"}
    faster = "a" if median_a_ms < median_b_ms else "b"
    return {"faster": faster, "ratio": round(ratio, 2), "band": "clear"}


def run_perf_index(db: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """eval_run_id -> light perf aggregates for the runs-list lens.

    Feeds GET /api/eval-runs rows (the Results page's run tables) with
    the same math as the report lens, kept cheap enough for the full
    list: one grouped fetch, fold, per-run stats.
    """
    rows = _fetch(
        db,
        """SELECT eval_run_id, item_id, latency_ms, prompt_tokens,
                  completion_tokens
           FROM eval_results WHERE eval_run_id IS NOT NULL""",
    )
    by_run: dict[int, list] = {}
    for r in rows:
        by_run.setdefault(r["eval_run_id"], []).append(r)
    out: dict[int, dict[str, Any]] = {}
    for rid, rrows in by_run.items():
        stats = perf_stats(fold_item_perf(rrows))
        out[rid] = {"total_latency_ms": stats["total_latency_ms"],
                    "median_item_latency_ms": stats["median_item_latency_ms"],
                    "output_tps": stats["output_tps"]}
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
             "judgments_stored", "item_level", "performance"}.
    Raises ValueError for unknown suite_id.

    ``performance`` (2026-11-03) is the second eval lens: {"runs": [...],
    "unscoped_rows": n}. Each run entry folds that replay run's rows to
    per-item perf (multi-turn turns sum into the item) — items, total
    item time, median/mean per item, token totals, output tok/s when the
    server reported usage. Each ``pairs`` entry carries ``performance``:
    per-side stats over the DISTINCT result rows that pair's judgments
    reference + a speed verdict (parity within 1.1x names no side).
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
    # performance lens: the DISTINCT result rows each pair's judgments
    # reference, per side (both presentation orders share the same rows —
    # dedupe by result id so latency is never double-counted)
    pair_result_ids: dict[tuple[str, str], tuple[set, set]] = {}

    for j in judgments:
        ra = result_ids.get(j["eval_result_a"])
        rb = result_ids.get(j["eval_result_b"])
        if ra is None or rb is None:
            continue  # defensive: judgment rows are FK-validated, but stay robust
        pair = tuple(sorted((ra["model_fingerprint"], rb["model_fingerprint"])))
        ids = pair_result_ids.setdefault(pair, (set(), set()))
        if ra["model_fingerprint"] == pair[0]:
            ids[0].add(ra["id"])
            ids[1].add(rb["id"])
        else:
            ids[0].add(rb["id"])
            ids[1].add(ra["id"])
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
            "latency_a_ms": ra["latency_ms"],
            "latency_b_ms": rb["latency_ms"],
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
        a_ids, b_ids = pair_result_ids.get((ma, mb), (set(), set()))
        a_rows = [result_ids[i] for i in a_ids if i in result_ids]
        b_rows = [result_ids[i] for i in b_ids if i in result_ids]
        a_stats = perf_stats(fold_item_perf(a_rows))
        b_stats = perf_stats(fold_item_perf(b_rows))
        pairs_out.append({
            "model_a": ma, "model_b": mb,
            "counts": dict(counts),
            "percent": pct,
            "total": t,
            "both_bad_rate": pct["both_bad"],
            "performance": {
                "a": a_stats,
                "b": b_stats,
                "speed": speed_comparison(a_stats["median_item_latency_ms"],
                                          b_stats["median_item_latency_ms"]),
                "runs_a": sorted({r["eval_run_id"] for r in a_rows
                                  if r["eval_run_id"] is not None}),
                "runs_b": sorted({r["eval_run_id"] for r in b_rows
                                  if r["eval_run_id"] is not None}),
            },
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

    # performance by replay run: each run's rows fold to per-item perf —
    # the run IS the "who completed the work faster" unit for a combined
    # A/B kick-off (same suite, two models, two runs). Legacy rows with
    # no eval_run_id stay out of the run lens but are counted honestly.
    rows_by_run: dict[int, list] = {}
    unscoped_rows = 0
    for r in results:
        if r["eval_run_id"] is None:
            unscoped_rows += 1
            continue
        rows_by_run.setdefault(r["eval_run_id"], []).append(r)
    perf_runs: list[dict[str, Any]] = []
    run_meta: dict[int, sqlite3.Row] = {}
    if rows_by_run:
        placeholders = ",".join("?" * len(rows_by_run))
        for m in _fetch(db,
                        f"SELECT * FROM eval_runs WHERE id IN ({placeholders})",
                        tuple(rows_by_run)):
            run_meta[m["id"]] = m
    for rid in sorted(rows_by_run):
        meta = run_meta.get(rid)
        perf_runs.append({
            "eval_run_id": rid,
            "model": meta["model_fingerprint"] if meta else None,
            "name": meta["name"] if meta else None,
            "status": meta["status"] if meta else None,
            "created_at": meta["created_at"] if meta else None,
            **perf_stats(fold_item_perf(rows_by_run[rid])),
        })

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
        "performance": {"runs": perf_runs, "unscoped_rows": unscoped_rows},
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
    fold into the base item; absent metadata -> 'uncategorized'). Each
    judged pair (2 presentation-order rows + optional tiebreak) is
    RESOLVED to one verdict before aggregation — majority of the pair's
    rows, tiebreak votes count in full, a tied vote -> draw, both-bad
    rides the draw when every row of the pair carried it — so n counts
    ITEMS judged, not stored rows (2026-10-25, owner hand-count: the
    raw-row math read accuracy n=20 for 10 items and mixed 5 tiebreak
    votes into an n=241 overall).

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
             "judges", "templates", "total", "rows", "overall",
             "performance"}. rows sorted n desc then dimension name asc.
             run_a/run_b echo the scope (nulls when lifetime-wide).

    ``performance`` (2026-11-03): the second lens over the SAME judged
    pairs — per-side stats over the distinct result rows the folded
    pairs reference (multi-turn turn rows fold into the item), a speed
    verdict ('parity' within 1.1x names no side), per-dimension medians,
    and the replay-run ids each side's rows came from. Orientation
    follows the caller: performance['a'] is ALWAYS the caller's
    model_a, matching the win-rate columns above it. ``performance
    ['tied']`` (2026-11-03) cuts to the items where preference has NO
    verdict — every judged turn of the item resolved to a draw (tie or
    both_bad) — because on those prompts wall time alone decides the
    production choice; None when no item is fully tied (absent stays
    absent, never zero).
    """
    sql = """SELECT j.winner, j.judge_model, j.verdict_detail,
                    j.judge_template_version AS template, j.created_at,
                    ra.model_fingerprint AS fa, rb.model_fingerprint AS fb,
                    ra.item_id AS item_id,
                    ra.id AS result_a_id, rb.id AS result_b_id,
                    ra.latency_ms AS latency_a_ms,
                    rb.latency_ms AS latency_b_ms,
                    ra.prompt_tokens AS ptok_a, rb.prompt_tokens AS ptok_b,
                    ra.completion_tokens AS ctok_a,
                    rb.completion_tokens AS ctok_b,
                    ra.eval_run_id AS run_id_a, rb.eval_run_id AS run_id_b
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

    # Resolve each judged pair to ONE verdict before aggregating
    # (2026-10-25, owner hand-count): the raw rows double-count every
    # item (two presentation orders per pair) and mix tiebreak votes
    # into the totals — accuracy read "n=20" for 10 items and the
    # overall n=241 carried 5 tiebreak votes as if they were extra
    # judgments. Fold = the judging pipeline's own row math: majority
    # of the pair's stored rows (tiebreak votes count in full); a tied
    # vote -> draw; both-bad rides the draw when EVERY row of the pair
    # carried verdict_detail='both_bad'. Grouping = result pair + judge
    # generation; the LATEST generation supersedes older ones (a re-judge
    # appends a new template/judge generation — the newest is the
    # operator's current truth, else a re-judge would double-count).
    by_pair: dict[tuple, list] = {}
    for r in rows:
        by_pair.setdefault(
            (r["result_a_id"], r["result_b_id"],
             r["judge_model"], r["template"]), []).append(r)

    # A re-judge under a new template/judge appends a NEW generation on
    # the same result pair (append-only doctrine) — the newest
    # generation SUPERSEDES the old one in the summary, otherwise a
    # re-judge would double-count every item. (created_at, judge, tpl)
    # picks deterministically when two generations share a timestamp.
    latest_gen: dict[tuple[int, int], tuple] = {}
    for key, prows in by_pair.items():
        ra_id, rb_id, jm, tpl = key
        gen_key = (max((p["created_at"] or "") for p in prows), jm, tpl)
        cur = latest_gen.get((ra_id, rb_id))
        if cur is None or gen_key > cur[0]:
            latest_gen[(ra_id, rb_id)] = (gen_key, prows)

    flip = {"a": "b", "b": "a"}
    overall = {"a": 0, "b": 0, "tie": 0, "both_bad": 0}
    per_dim: dict[str, dict[str, int]] = {}
    judges: set[str] = set()
    templates: set[str] = set()
    # performance lens (second dimension): synthetic per-turn rows per
    # side, deduped by result id across the folded pairs — fold_item_perf
    # then sums multi-turn turn rows into their base item
    perf_syn_a: list[dict[str, Any]] = []
    perf_syn_b: list[dict[str, Any]] = []
    perf_dim_a: dict[str, list] = {}
    perf_dim_b: dict[str, list] = {}
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    # quality-tied lens: an ITEM is "quality-tied" when EVERY judged
    # turn-pair of the item resolved to a draw (tie or both_bad) — the
    # prompts where preference has no verdict, so speed alone decides
    judged_turns: dict[str, int] = {}
    tied_turns: dict[str, int] = {}
    runs_a: set[int] = set()
    runs_b: set[int] = set()
    for _, (_, pair_rows) in latest_gen.items():
        r = pair_rows[0]          # canonical fields identical within a pair
        # canonical-space majority (tiebreak votes count in full;
        # tied vote -> draw), THEN the caller-order flip below
        votes = [(p["winner"] if p["winner"] in ("a", "b") else "tie")
                 for p in pair_rows]
        vote_counts = {v: votes.count(v) for v in ("a", "b", "tie")}
        top = max(vote_counts.values())
        tops = [v for v, c in vote_counts.items() if c == top]
        resolved = tops[0] if len(tops) == 1 else "tie"
        is_both_bad = (resolved == "tie" and all(
            (p["verdict_detail"] or "") == "both_bad" for p in pair_rows))
        winner: str = resolved
        if r["fa"] != model_a:  # stored canonical order != caller order
            winner = flip.get(winner, winner)
        winner = winner if winner in ("a", "b", "tie") else "tie"
        # both-bad draws ride the draw semantics (Arena/MT-Bench
        # convention); the flag was resolved per PAIR above.
        bucket_key = "both_bad" if is_both_bad else winner
        dimension = (metadata.get(r["item_id"].split("#")[0], {})
                     ).get("dimension", "uncategorized")
        bucket = per_dim.setdefault(dimension, {"a": 0, "b": 0, "tie": 0,
                                                "both_bad": 0})
        bucket[bucket_key] += 1
        overall[bucket_key] += 1
        base_item = r["item_id"].split("#")[0]
        judged_turns[base_item] = judged_turns.get(base_item, 0) + 1
        if bucket_key in ("tie", "both_bad"):
            tied_turns[base_item] = tied_turns.get(base_item, 0) + 1
        judges.add(r["judge_model"])
        templates.add(r["template"])
        # performance lens: caller-oriented synthetic rows, deduped by
        # result id (both presentation rows of a judged pair reference
        # the same two results — each side's latency counts ONCE)
        if r["fa"] == model_a:
            ra_id, rb_id = r["result_a_id"], r["result_b_id"]
            ra_row = {"item_id": r["item_id"],
                      "latency_ms": r["latency_a_ms"],
                      "prompt_tokens": r["ptok_a"],
                      "completion_tokens": r["ctok_a"]}
            rb_row = {"item_id": r["item_id"],
                      "latency_ms": r["latency_b_ms"],
                      "prompt_tokens": r["ptok_b"],
                      "completion_tokens": r["ctok_b"]}
            run_a_id, run_b_id = r["run_id_a"], r["run_id_b"]
        else:
            ra_id, rb_id = r["result_b_id"], r["result_a_id"]
            ra_row = {"item_id": r["item_id"],
                      "latency_ms": r["latency_b_ms"],
                      "prompt_tokens": r["ptok_b"],
                      "completion_tokens": r["ctok_b"]}
            rb_row = {"item_id": r["item_id"],
                      "latency_ms": r["latency_a_ms"],
                      "prompt_tokens": r["ptok_a"],
                      "completion_tokens": r["ctok_a"]}
            run_a_id, run_b_id = r["run_id_b"], r["run_id_a"]
        if ra_id not in seen_a:
            seen_a.add(ra_id)
            perf_syn_a.append(ra_row)
            perf_dim_a.setdefault(dimension, []).append(ra_row)
            if run_a_id is not None:
                runs_a.add(run_a_id)
        if rb_id not in seen_b:
            seen_b.add(rb_id)
            perf_syn_b.append(rb_row)
            perf_dim_b.setdefault(dimension, []).append(rb_row)
            if run_b_id is not None:
                runs_b.add(run_b_id)

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

    # performance aggregation over the SAME folded pairs (second lens):
    # per-side stats + speed verdict + per-dimension medians
    a_perf = perf_stats(fold_item_perf(perf_syn_a))
    b_perf = perf_stats(fold_item_perf(perf_syn_b))
    by_dim_perf: dict[str, dict[str, Any]] = {}
    for dim in sorted(set(perf_dim_a) | set(perf_dim_b)):
        da_stats = perf_stats(fold_item_perf(perf_dim_a.get(dim, [])))
        db_stats = perf_stats(fold_item_perf(perf_dim_b.get(dim, [])))
        by_dim_perf[dim] = {
            "a": da_stats,
            "b": db_stats,
            "speed": speed_comparison(da_stats["median_item_latency_ms"],
                                      db_stats["median_item_latency_ms"]),
        }
    performance = {
        "a": a_perf,
        "b": b_perf,
        "speed": speed_comparison(a_perf["median_item_latency_ms"],
                                  b_perf["median_item_latency_ms"]),
        "by_dimension": by_dim_perf,
        "runs_a": sorted(runs_a),
        "runs_b": sorted(runs_b),
    }
    # quality-tied cut: among items where the judge preferred NEITHER side
    # (every judged turn a draw), wall time is the deciding metric — the
    # production decision the tie rate alone cannot make
    tied_items = {base for base, judged in judged_turns.items()
                  if judged and tied_turns.get(base, 0) == judged}
    if tied_items:
        tied_a = perf_stats(fold_item_perf(
            [r for r in perf_syn_a if r["item_id"].split("#")[0] in tied_items]))
        tied_b = perf_stats(fold_item_perf(
            [r for r in perf_syn_b if r["item_id"].split("#")[0] in tied_items]))
        performance["tied"] = {
            "items": len(tied_items),
            "a": tied_a,
            "b": tied_b,
            "speed": speed_comparison(tied_a["median_item_latency_ms"],
                                      tied_b["median_item_latency_ms"]),
        }
    else:
        performance["tied"] = None
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
        "performance": performance,
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
                     "wilson_a_low", "wilson_a_high",
                     "median_ms_a", "median_ms_b",
                     "total_s_a", "total_s_b",
                     "tok_s_a", "tok_s_b",
                     "speed_ratio", "speed_faster"])
    for pair in report["pairs"]:
        wilson = pair.get("wilson_a") or {}
        perf = pair.get("performance") or {}
        pa, pb = perf.get("a") or {}, perf.get("b") or {}
        speed = perf.get("speed") or {}
        total_s = lambda stats: (  # noqa: E731  (inline cell formatting)
            round(stats["total_latency_ms"] / 1000.0, 1)
            if stats.get("total_latency_ms") is not None else "")
        writer.writerow([
            report["suite"]["name"], report["suite"]["version"],
            pair["model_a"], pair["model_b"],
            pair["counts"]["a"], pair["counts"]["b"], pair["counts"]["tie"],
            pair["counts"].get("both_bad", 0),
            pair["total"],
            pair["percent"]["a"], pair["percent"]["b"], pair["percent"]["tie"],
            pair["percent"].get("both_bad"),
            wilson.get("low", ""), wilson.get("high", ""),
            pa.get("median_item_latency_ms", "") or "",
            pb.get("median_item_latency_ms", "") or "",
            total_s(pa), total_s(pb),
            pa.get("output_tps", "") if pa.get("output_tps") is not None else "",
            pb.get("output_tps", "") if pb.get("output_tps") is not None else "",
            speed.get("ratio") if speed.get("ratio") is not None else "",
            speed.get("faster") or "",
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
