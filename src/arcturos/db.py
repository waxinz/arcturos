"""SQLite storage layer for Arcturos.

Append-only by design: the API exposes no update/delete paths, and the schema
installs BEFORE UPDATE / BEFORE DELETE triggers that hard-reject mutations at
the storage layer (defence in depth). Re-running a measurement always appends
a new row; it never overwrites stored data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "arcturos.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    server_url        TEXT    NOT NULL,
    model_fingerprint TEXT    NOT NULL,
    engine            TEXT    NOT NULL,
    context_size      INTEGER NOT NULL,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    context_tokens  INTEGER NOT NULL,
    prefill_tps     REAL,
    decode_tps      REAL,
    ttft_ms         REAL,
    wall_s          REAL,
    output_tokens   INTEGER,
    mtp_draft_n     INTEGER,
    mtp_accepted    INTEGER,
    power_watts     REAL,
    power_host      TEXT,
    power_gpu_index INTEGER,
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_suites (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_id          INTEGER NOT NULL REFERENCES eval_suites(id),
    model_fingerprint TEXT    NOT NULL,
    item_id           TEXT    NOT NULL,
    output            TEXT    NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms        REAL    NOT NULL,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    model_fingerprint TEXT    PRIMARY KEY,
    alias             TEXT,
    engine            TEXT,
    first_seen        TEXT    NOT NULL,
    last_seen         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS baselines (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_fingerprint TEXT    NOT NULL REFERENCES models(model_fingerprint),
    metric_family     TEXT    NOT NULL,
    run_id            INTEGER NOT NULL REFERENCES runs(id),
    created_at        TEXT    NOT NULL,
    UNIQUE (model_fingerprint, metric_family)
);

CREATE TABLE IF NOT EXISTS judgments (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_result_a          INTEGER NOT NULL REFERENCES eval_results(id),
    eval_result_b          INTEGER NOT NULL REFERENCES eval_results(id),
    judge_model            TEXT    NOT NULL,
    judge_template_version TEXT    NOT NULL,
    winner                 TEXT    NOT NULL CHECK (winner IN ('a', 'b', 'tie')),
    confidence             REAL    CHECK (confidence BETWEEN 0 AND 1),
    rationale              TEXT,
    created_at             TEXT    NOT NULL,
    CHECK (eval_result_a <> eval_result_b)
);
"""

# Hard append-only enforcement at the storage layer: any UPDATE or DELETE on a
# stored table aborts with an 'append-only' error, even outside the API.
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS trg_runs_no_update
BEFORE UPDATE ON runs BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on runs');
END;
CREATE TRIGGER IF NOT EXISTS trg_runs_no_delete
BEFORE DELETE ON runs BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on runs');
END;

CREATE TRIGGER IF NOT EXISTS trg_benchmarks_no_update
BEFORE UPDATE ON benchmarks BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on benchmarks');
END;
CREATE TRIGGER IF NOT EXISTS trg_benchmarks_no_delete
BEFORE DELETE ON benchmarks BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on benchmarks');
END;

CREATE TRIGGER IF NOT EXISTS trg_eval_suites_no_update
BEFORE UPDATE ON eval_suites BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on eval_suites');
END;
CREATE TRIGGER IF NOT EXISTS trg_eval_suites_no_delete
BEFORE DELETE ON eval_suites BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on eval_suites');
END;

CREATE TRIGGER IF NOT EXISTS trg_eval_results_no_update
BEFORE UPDATE ON eval_results BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on eval_results');
END;
CREATE TRIGGER IF NOT EXISTS trg_eval_results_no_delete
BEFORE DELETE ON eval_results BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on eval_results');
END;

CREATE TRIGGER IF NOT EXISTS trg_judgments_no_update
BEFORE UPDATE ON judgments BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on judgments');
END;
CREATE TRIGGER IF NOT EXISTS trg_judgments_no_delete
BEFORE DELETE ON judgments BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on judgments');
END;

-- models: the fingerprint, engine, and seen timestamps are immutable once
-- registered; only the alias may change (it is a label, not data — §4.6).
CREATE TRIGGER IF NOT EXISTS trg_models_no_delete
BEFORE DELETE ON models BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden on models');
END;
CREATE TRIGGER IF NOT EXISTS trg_models_alias_only
BEFORE UPDATE ON models BEGIN
    SELECT CASE WHEN
        NEW.model_fingerprint IS NOT OLD.model_fingerprint OR
        NEW.engine IS NOT OLD.engine OR
        NEW.first_seen IS NOT OLD.first_seen
    THEN RAISE(ABORT, 'append-only: only the alias or last_seen may be updated on models')
    END;
END;

-- baselines are reference pointers (§4.6), not stored data: re-pinning a
-- (model, metric-family) pair replaces the pointer, and unpinning removes
-- it. The underlying run rows stay untouched.
CREATE TRIGGER IF NOT EXISTS trg_baselines_replace_pin
BEFORE INSERT ON baselines WHEN
    (SELECT COUNT(*) FROM baselines
     WHERE model_fingerprint = NEW.model_fingerprint
       AND metric_family = NEW.metric_family) > 0
BEGIN
    SELECT RAISE(ABORT, 'baseline-replace');
END;
CREATE TRIGGER IF NOT EXISTS trg_baselines_no_update
BEFORE UPDATE ON baselines BEGIN
    SELECT RAISE(ABORT, 'append-only: re-pin instead of updating a baseline');
END;
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys + WAL enabled.

    WAL journal mode is set once here and persists on the database file, so
    repeated connects are cheap and concurrent reads/writes are safe.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables + append-only triggers. Idempotent (IF NOT EXISTS)."""
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.executescript(TRIGGERS)
    _backfill_registry(conn)
    conn.commit()


def _backfill_registry(conn: sqlite3.Connection) -> None:
    """Register fingerprints that predate the models table (insert-only).

    Runs recorded before /models existed never got a sighting, so the
    registry would show an empty page despite stored data. The backfill is
    pure INSERT OR IGNORE — it never mutates an existing registry row.
    """
    try:
        conn.execute(
            "INSERT OR IGNORE INTO models (model_fingerprint, alias, engine, "
            "first_seen, last_seen) "
            "SELECT DISTINCT model_fingerprint, NULL, engine, "
            "MIN(created_at), MIN(created_at) FROM runs GROUP BY "
            "model_fingerprint"
        )
    except sqlite3.OperationalError:
        # a pre-schema database arriving mid-migration — next init retries
        pass


# Columns added after v0.1: (table, column, DDL type). ALTER TABLE adds them to
# pre-existing databases; fresh databases already have them from SCHEMA (the
# duplicate-column error is swallowed for that case).
_MIGRATIONS = (
    ("benchmarks", "power_host", "TEXT"),
    ("benchmarks", "power_gpu_index", "INTEGER"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add post-v0.1 columns to pre-existing databases (idempotent).

    Schema evolution only — never touches row data, so append-only semantics
    are untouched. ALTER TABLE is metadata DDL, not a data mutation.
    """
    existing: dict[str, set[str]] = {}
    for table, _col, _ddl in _MIGRATIONS:
        if table not in existing:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            existing[table] = cols
        if _col not in existing[table]:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {_col} {_ddl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
