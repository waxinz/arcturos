"""J2 comparison logic — benchmark series + run diffs over the Arcturos store.

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
from typing import Optional

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
}

_SHARD_SUFFIX = re.compile(r"-\d+-of-\d+$")


def host_label(server_url: str) -> str:
    """Compact host tag for legends: 'http://10.10.10.122:8000' -> '10.10.10.122'.

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
    db: sqlite3.Connection, run_ids: list[int], metrics: list[str]
) -> dict:
    """Series payload for GET /api/compare/benchmarks.

    ``run_ids`` and ``metrics`` are assumed validated by the caller (existing
    runs, allowlisted metrics). Callers also guarantee run order; series are
    keyed by ``str(run_id)`` per the JSON contract.
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
    return {
        "runs": [
            {
                "id": run["id"],
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
            "model_fingerprint_short": short_fingerprint(run_a["model_fingerprint"]),
            "engine": run_a["engine"],
        },
        "run_b": {
            "id": run_b["id"],
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
