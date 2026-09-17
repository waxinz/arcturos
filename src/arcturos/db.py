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
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    context_tokens INTEGER NOT NULL,
    prefill_tps    REAL,
    decode_tps     REAL,
    ttft_ms        REAL,
    wall_s         REAL,
    output_tokens  INTEGER,
    mtp_draft_n    INTEGER,
    mtp_accepted   INTEGER,
    power_watts    REAL,
    created_at     TEXT    NOT NULL
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
    conn.executescript(TRIGGERS)
    conn.commit()
