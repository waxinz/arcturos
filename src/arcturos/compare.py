"""J2 comparison logic — benchmark series + run diffs over the Arcturos store.

Baseline overlay support (§4.6): ``build_benchmarks_payload`` optionally
carries a pinned reference run's series + metadata, and
``baseline_deltas`` computes honest per-point deltas (missing either side
→ no row, never a fabricated zero).

Pure functions over a ``sqlite3.Connection`` (no FastAPI imports): the API
endpoints in :mod:`arcturos.main` are thin wrappers, and tests can exercise
this module directly through the TestClient.

Conventions (PRD §J2 / docs/ux-design.md §4.2):

* Every allowlisted metric is returned for every point a run actually has;
  a point whose metric is ``NULL`` in storage is emitted with ``value: null``
  — honest nulls, never dropped, never turned into ``0``.
* ``mtp_acceptance`` is computed as ``mtp_accepted / mtp_draft_n`` when both
  fields are present and ``mtp_draft_n > 0``; otherwise ``null``.
* Runs are append-only, so a run can hold several benchmark rows at the same
  ``context_tokens`` (a re-run supersedes, it never overwrites — PRD §J1).
  Comparison views resolve duplicates to the **latest** row per context
  length (highest rowid), because that is the superseding measurement.
* Deltas are always ``run_b - run_a``, pct ``(delta / a) * 100``; either is
  ``null`` when the base side is missing or zero.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional

# Canonical metric order for the allowlist. ``mtp_acceptance`` is derived;
# every other name is a column on ``benchmarks``.
ALLOWED_METRICS = (
    "decode_tps",
    "prefill_tps",
    "ttft_ms",
    "wall_s",
    "output_tokens",
    "mtp_acceptance",
    "power_watts",
    "streams",
    "decode_tps_combined",
    "prefill_tps_combined",
)

# Which direction counts as "better" for delta coloring in the run-diff view.
# tps/mtp higher-is-better; latencies lower-is-better; output_tokens and
# power_watts are informational (no green/red verdict).
METRIC_DIRECTIONS = {
    "decode_tps": "higher",
    "prefill_tps": "higher",
    "ttft_ms": "lower",
    "wall_s": "lower",
    "output_tokens": "neutral",
    "mtp_acceptance": "higher",
    "power_watts": "neutral",
    "streams": "neutral",
    "decode_tps_combined": "higher",
    "prefill_tps_combined": "higher",
}

# Direct benchmark column per metric; None = computed (mtp_acceptance).
_METRIC_COLUMN = {
    "decode_tps": "decode_tps",
    "prefill_tps": "prefill_tps",
    "ttft_ms": "ttft_ms",
    "wall_s": "wall_s",
    "output_tokens": "output_tokens",
    "mtp_acceptance": None,
    "power_watts": "power_watts",
    "streams": "streams",
    "decode_tps_combined": "decode_tps_combined",
    "prefill_tps_combined": "prefill_tps_combined",
}

_SHARD_SUFFIX = re.compile(r"-\d+-of-\d+$")


def host_label(server_url: str) -> str:
    """Compact host tag for legends: 'http://bench.example.com:8000' -> 'bench.example.com'.

    Distinguishes multi-host runs in compare views (ADR 002: power metrics
    and queueing behavior are host-specific, so the host must be visible).
    """
    try:
        from urllib.parse import urlparse
        return urlparse(server_url).hostname or server_url
    except (ValueError, TypeError):
        return server_url


def short_fingerprint(fingerprint: str) -> str:
    """Human-scale legend name for a model fingerprint.

    A fingerprint that looks like a filesystem path (the common llama.cpp
    case — a GGUF snapshot path) is reduced to its basename with the
    ``.gguf`` extension and the ``-NNNNN-of-NNNNN`` shard suffix removed:
    ``…/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf`` becomes
    ``DeepSeek-V4-Flash-0731-UD-IQ3_XXS``. Non-path labels (e.g. the demo
    seed strings) are returned verbatim.
    """
    if not fingerprint:
        return fingerprint
    if "/" not in fingerprint:
        return fingerprint
    name = fingerprint.rstrip("/").split("/")[-1]
    if name.endswith(".gguf"):
        name = name[: -len(".gguf")]
    name = _SHARD_SUFFIX.sub("", name)
    return name or fingerprint


def metric_value(row: sqlite3.Row, metric: str) -> Optional[float]:
    """Extract one metric from a benchmark row (None for missing data).

    ``mtp_acceptance`` is computed from the two raw MTP counters; any other
    metric is read straight from its column.
    """
    if metric == "mtp_acceptance":
        draft_n = row["mtp_draft_n"]
        accepted = row["mtp_accepted"]
        if draft_n is None or accepted is None or draft_n == 0:
            return None
        return accepted / draft_n
    return row[metric]


def fetch_run(db: sqlite3.Connection, run_id: int) -> Optional[sqlite3.Row]:
    return db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def fetch_benchmarks(db: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Benchmark rows for a run, latest measurement per context length.

    Rows are ordered by ``context_tokens`` (then rowid, newest last) and
    deduplicated to the highest rowid per context length — the superseding
    measurement under the append-only model (PRD §J1).
    """
    rows = db.execute(
        "SELECT rowid AS _rid, * FROM benchmarks WHERE run_id = ? "
        "ORDER BY context_tokens, rowid",
        (run_id,),
    ).fetchall()
    latest: dict[int, sqlite3.Row] = {}
    for row in rows:
        latest[row["context_tokens"]] = row  # later rowid wins
    return [latest[ctx] for ctx in sorted(latest)]


def build_benchmarks_payload(
    db: sqlite3.Connection, run_ids: list[int], metrics: list[str],
    baseline_run_id: int | None = None,
) -> dict:
    """Series payload for GET /api/compare/benchmarks.

    ``run_ids`` and ``metrics`` are assumed validated by the caller (existing
    runs, allowlisted metrics). Callers also guarantee run order; series are
    keyed by ``str(run_id)`` per the JSON contract.

    ``baseline_run_id`` (optional): when set, the payload carries the pinned
    reference's points under ``series['baseline']`` + a ``baseline`` metadata
    block, so views can render the dashed reference line and delta columns
    (§4.6). The baseline run does NOT need to be in ``run_ids``.
    """
    runs = [dict(fetch_run(db, rid)) for rid in run_ids]
    series: dict[str, dict[str, list[dict]]] = {}
    for metric in metrics:
        series[metric] = {}
        for rid in run_ids:
            points = [
                {"context_tokens": row["context_tokens"], "value": metric_value(row, metric)}
                for row in fetch_benchmarks(db, rid)
            ]
            series[metric][str(rid)] = points

    payload: dict[str, Any] = {
        "runs": [
            {
                "id": run["id"],
                "name": run.get("name"),
                "model_fingerprint_short": short_fingerprint(run["model_fingerprint"]),
                "model_fingerprint": run["model_fingerprint"],
                "engine": run["engine"],
                # multi-host compare (J7/ADR 002): the host is visible so two
                # runs on different boxes are never silently conflated
                "server_url": run["server_url"],
                "host_label": host_label(run["server_url"]),
            }
            for run in runs
        ],
        "metrics": metrics,
        "series": series,
    }

    if baseline_run_id is not None:
        baseline_run = fetch_run(db, baseline_run_id)
        if baseline_run is None:
            raise ValueError(f"baseline run id {baseline_run_id} not found")
        baseline_points: dict[str, list[dict]] = {}
        for metric in metrics:
            baseline_points[metric] = [
                {"context_tokens": row["context_tokens"], "value": metric_value(row, metric)}
                for row in fetch_benchmarks(db, baseline_run_id)
            ]
        payload["baseline"] = {
            "run_id": baseline_run_id,
            "name": baseline_run["name"] if "name" in baseline_run.keys() else None,
            "model_fingerprint": baseline_run["model_fingerprint"],
            "model_fingerprint_short": short_fingerprint(baseline_run["model_fingerprint"]),
            "engine": baseline_run["engine"],
            "host_label": host_label(baseline_run["server_url"]),
        }
        # keyed by metric so each chart can overlay its own dashed line
        payload["series"]["baseline"] = baseline_points
    return payload


def baseline_deltas(payload: dict[str, Any], run_id: int) -> dict[str, list[dict]]:
    """Per-point deltas of one run against the payload's baseline (§4.6).

    For each metric: the baseline point at the same context_tokens, the
    run's value, and the absolute + percent delta. Points where either side
    is missing (None) are excluded — a delta against nothing is not a
    number, and honest nulls beat fabricated zeros. Higher-is-better for
    tok/s and tokens; lower-is-better for TTFT and wall — the percent is
    signed so the view can colour good/bad without knowing semantics.
    """
    if "baseline" not in payload:
        raise ValueError("payload has no baseline to compare against")
    baseline_points = payload["series"]["baseline"]
    # series is keyed [metric][run_id]; the run must be IN the payload's
    # run list for its series to exist (the endpoint guarantees that).
    if str(run_id) not in {str(r["id"]) for r in payload["runs"]}:
        raise ValueError(f"run id {run_id} is not part of this payload")
    deltas: dict[str, list[dict]] = {}
    for metric, base_pts in baseline_points.items():
        if metric not in payload["series"]:
            continue
        run_points = payload["series"][metric].get(str(run_id), [])
        base_by_ctx = {p["context_tokens"]: p["value"] for p in base_pts}
        rows = []
        for point in run_points:
            base = base_by_ctx.get(point["context_tokens"])
            if base is None or point["value"] is None:
                continue
            delta = point["value"] - base
            rows.append({
                "context_tokens": point["context_tokens"],
                "baseline_value": base,
                "value": point["value"],
                "delta": delta,
                "delta_pct": (delta / base) if base else None,
            })
        if rows:
            deltas[metric] = rows
    return deltas


def build_run_diff_payload(
    db: sqlite3.Connection, run_a_id: int, run_b_id: int
) -> dict:
    """Per-context delta table for two runs + engine metadata diff.

    The caller guarantees both runs exist. Every allowlisted metric appears
    in every context row (``null`` when a side is missing); ``delta`` is
    ``b - a`` and ``pct`` is ``(delta / a) * 100``, both ``null`` when the
    base side is missing or zero. Engine metadata (n_ctx, model path, engine,
    server URL) is compared field-by-field with a ``same`` flag.
    """
    run_a = dict(fetch_run(db, run_a_id))
    run_b = dict(fetch_run(db, run_b_id))
    points_a = fetch_benchmarks(db, run_a_id)
    points_b = fetch_benchmarks(db, run_b_id)

    by_ctx_a = {row["context_tokens"]: row for row in points_a}
    by_ctx_b = {row["context_tokens"]: row for row in points_b}

    deltas: list[dict] = []
    for ctx in sorted(set(by_ctx_a) | set(by_ctx_b)):
        row_a, row_b = by_ctx_a.get(ctx), by_ctx_b.get(ctx)
        metrics: dict[str, dict] = {}
        for metric in ALLOWED_METRICS:
            va = metric_value(row_a, metric) if row_a is not None else None
            vb = metric_value(row_b, metric) if row_b is not None else None
            if va is not None and vb is not None:
                delta = vb - va
                pct = (delta / va) * 100 if va != 0 else None
            else:
                delta = pct = None
            metrics[metric] = {"a": va, "b": vb, "delta": delta, "pct": pct}
        deltas.append({"context_tokens": ctx, "metrics": metrics})

    def _field_diff(label_a, label_b):
        return {"a": label_a, "b": label_b, "same": label_a == label_b}

    return {
        "run_a": {
            "id": run_a["id"],
            "name": run_a.get("name"),
            "model_fingerprint_short": short_fingerprint(run_a["model_fingerprint"]),
            "engine": run_a["engine"],
        },
        "run_b": {
            "id": run_b["id"],
            "name": run_b.get("name"),
            "model_fingerprint_short": short_fingerprint(run_b["model_fingerprint"]),
            "engine": run_b["engine"],
        },
        "engine_metadata": {
            "n_ctx": _field_diff(run_a["context_size"], run_b["context_size"]),
            "model_path": _field_diff(
                run_a["model_fingerprint"], run_b["model_fingerprint"]
            ),
            "engine": _field_diff(run_a["engine"], run_b["engine"]),
            "server_url": _field_diff(run_a["server_url"], run_b["server_url"]),
            "host": _field_diff(host_label(run_a["server_url"]),
                                host_label(run_b["server_url"])),
        },
        "directions": dict(METRIC_DIRECTIONS),
        "deltas": deltas,
    }


def export_benchmarks_csv(payload: dict[str, Any]) -> str:
    """CSV export of a compare payload (§4.6 data export).

    One long-format row per (run, metric, context point): the header carries
    ``schema_version`` and the payload carries fingerprint + host fields on
    every row so external analysis can self-describe. Honest nulls: a
    missing metric value is an empty cell, never a fabricated 0.
    """
    import csv
    import io

    SCHEMA_VERSION = "arcturos-compare-v1"
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["schema_version", "run_id", "name", "model_fingerprint",
                     "model_fingerprint_short", "engine", "server_url",
                     "host_label", "metric", "context_tokens", "value"])
    baseline_fp = None
    if "baseline" in payload:
        baseline_fp = payload["baseline"]["model_fingerprint"]
        writer.writerow([SCHEMA_VERSION, payload["baseline"]["run_id"],
                         payload["baseline"].get("name") or "",
                         baseline_fp, payload["baseline"]["model_fingerprint_short"],
                         payload["baseline"]["engine"], "",  # server_url n/a for reference row
                         payload["baseline"]["host_label"], "baseline",
                         "", ""])
    for run in payload["runs"]:
        for metric in payload["metrics"]:
            for point in payload["series"][metric].get(str(run["id"]), []):
                value = point["value"]
                writer.writerow([SCHEMA_VERSION, run["id"],
                                 run.get("name") or "",
                                 run["model_fingerprint"],
                                 run["model_fingerprint_short"],
                                 run["engine"], run["server_url"],
                                 run["host_label"], metric,
                                 point["context_tokens"],
                                 "" if value is None else value])
    return buf.getvalue()
