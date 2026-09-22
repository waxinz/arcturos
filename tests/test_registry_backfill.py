"""Registry backfill: fingerprints that predate the models table.

init_db must register them insert-only, without mutating any existing
registry row, and idempotently.
"""
import sqlite3

import pytest


def test_backfill_registers_preexisting_runs(tmp_path):
    from arcturos.db import connect, init_db

    db_path = tmp_path / "arcturos.db"
    conn = connect(db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (server_url, model_fingerprint, engine, "
        "context_size, created_at) VALUES (?, ?, ?, ?, ?)",
        ("http://10.10.10.122:8000", "/m/old-fp.gguf", "llama.cpp",
         262144, "2026-09-01T10:00:00.000"),
    )
    conn.commit()
    conn.close()

    # a second init (what every get_db does) must backfill the registry
    conn = connect(db_path)
    init_db(conn)
    rows = conn.execute(
        "SELECT model_fingerprint, alias, engine, first_seen, last_seen "
        "FROM models").fetchall()
    assert len(rows) == 1
    fp, alias, engine, first, last = rows[0]
    assert fp == "/m/old-fp.gguf"
    assert alias is None
    assert engine == "llama.cpp"
    assert first == last == "2026-09-01T10:00:00.000"


def test_backfill_is_idempotent_and_never_mutates(tmp_path):
    from arcturos.db import connect, init_db

    db_path = tmp_path / "arcturos.db"
    conn = connect(db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (server_url, model_fingerprint, engine, "
        "context_size, created_at) VALUES (?, ?, ?, ?, ?)",
        ("http://a:1", "/m/x.gguf", "llama.cpp", 4096,
         "2026-09-01T10:00:00.000"),
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)
    # owner set an alias + a newer run arrived with a different created_at;
    # re-init must keep first_seen and the alias untouched
    conn.execute("UPDATE models SET alias = 'x-250w' WHERE model_fingerprint = '/m/x.gguf'")
    conn.execute(
        "INSERT INTO runs (server_url, model_fingerprint, engine, "
        "context_size, created_at) VALUES (?, ?, ?, ?, ?)",
        ("http://a:1", "/m/x.gguf", "llama.cpp", 8192,
         "2026-09-15T10:00:00.000"),
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)
    row = conn.execute(
        "SELECT alias, first_seen FROM models WHERE model_fingerprint = '/m/x.gguf'"
    ).fetchone()
    assert row["alias"] == "x-250w"
    assert row["first_seen"] == "2026-09-01T10:00:00.000"
    conn.close()


def test_backfill_rejects_update_via_trigger(tmp_path):
    """A registry row from a real sighting must not be rewritten by the
    backfill's INSERT OR IGNORE (conflict -> ignored, not updated)."""
    from arcturos.db import connect, init_db

    db_path = tmp_path / "arcturos.db"
    conn = connect(db_path)
    init_db(conn)
    # register through the normal path with one first_seen...
    conn.execute(
        "INSERT INTO models (model_fingerprint, alias, engine, first_seen, "
        "last_seen) VALUES (?, ?, ?, ?, ?)",
        ("/m/y.gguf", None, "llama.cpp", "2026-09-10T00:00:00.000",
         "2026-09-10T00:00:00.000"),
    )
    # ...then a run with a DIFFERENT (earlier) created_at tries to backfill
    conn.execute(
        "INSERT INTO runs (server_url, model_fingerprint, engine, "
        "context_size, created_at) VALUES (?, ?, ?, ?, ?)",
        ("http://b:2", "/m/y.gguf", "llama.cpp", 4096,
         "2026-09-05T00:00:00.000"),
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)
    row = conn.execute(
        "SELECT first_seen FROM models WHERE model_fingerprint = '/m/y.gguf'"
    ).fetchone()
    assert row["first_seen"] == "2026-09-10T00:00:00.000"
    conn.close()


def test_stale_label_trigger_upgrades_on_legacy_db(tmp_path):
    """A database created before the naming feature carries the OLD blanket
    trg_runs_no_update (any UPDATE aborts). CREATE TRIGGER IF NOT EXISTS is
    a no-op when the name is taken, so init_db must detect the stale
    definition and replace it — otherwise renames 500 forever on live DBs
    (observed on taupo 2026-09-22). Measurement columns must stay locked."""
    import sqlite3
    from arcturos.db import connect, init_db

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_url TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            engine TEXT,
            context_size INTEGER,
            status TEXT NOT NULL DEFAULT 'complete',
            created_at TEXT NOT NULL
        );
        CREATE TRIGGER trg_runs_no_update BEFORE UPDATE ON runs BEGIN
            SELECT RAISE(ABORT, 'append-only: UPDATE forbidden on runs');
        END;
        INSERT INTO runs (server_url, model_fingerprint, engine,
                          context_size, created_at)
        VALUES ('http://x', 'm', 'llama.cpp', 4096, '2026-09-22');
    """)
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)

    # rename now allowed (label whitelisted by the upgraded trigger)
    conn.execute("UPDATE runs SET name = ? WHERE id = ?", ("renamed", 1))
    conn.commit()
    name = conn.execute("SELECT name FROM runs WHERE id = 1").fetchone()[0]
    assert name == "renamed"

    # measurement columns still immutable
    try:
        conn.execute("UPDATE runs SET context_size = 1 WHERE id = 1")
        raise AssertionError("measurement UPDATE should have aborted")
    except sqlite3.IntegrityError:
        pass

    # idempotent: a second init_db neither fails nor downgrades
    init_db(conn)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'trg_runs_no_update'"
    ).fetchone()[0]
    assert "only the name" in sql
    conn.close()
