"""Arcturos dashboard — FastAPI app (JSON API + static index.html).

Run with:  uvicorn arcturos.main:app --host 0.0.0.0 --port 24816

Routes (all data paths are append-only — no UPDATE/DELETE exists):
  GET  /                                -> dashboard HTML
  GET  /health                          -> {"status": "ok"}
  POST /api/runs                        -> create run
  GET  /api/runs                        -> list runs
  GET  /api/runs/{run_id}               -> fetch one run
  POST /api/runs/{run_id}/benchmarks    -> add benchmark row to a run
  GET  /api/runs/{run_id}/benchmarks    -> list benchmarks for a run
  POST /api/eval-suites                 -> create eval suite
  GET  /api/eval-suites                 -> list eval suites
  GET  /api/eval-suites/{suite_id}      -> fetch one suite
  POST /api/eval-suites/{suite_id}/results -> add eval result
  GET  /api/eval-suites/{suite_id}/results -> list results for a suite
  GET  /api/eval-results                -> list all eval results
  POST /api/judgments                   -> add judgment
  GET  /api/judgments                   -> list judgments
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .db import DEFAULT_DB_PATH, connect, init_db
from .schemas import (
    BenchmarkCreate,
    EvalResultCreate,
    EvalSuiteCreate,
    JudgmentCreate,
    RunCreate,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(db_path: str | Path | None = None) -> FastAPI:
    """Build the app against a specific SQLite file (tests use tmp paths)."""
    db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    conn = connect(db_path)
    init_db(conn)
    conn.close()

    app = FastAPI(title="Arcturos", version="0.1.0")
    app.state.db_path = db_path

    def get_db(request: Request):
        conn = connect(request.app.state.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def fetch_one(db: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
        row = db.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def fetch_all(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in db.execute(sql, params).fetchall()]

    def require_parent(db: sqlite3.Connection, table: str, pk: int) -> None:
        if fetch_one(db, f"SELECT 1 AS ok FROM {table} WHERE id = ?", (pk,)) is None:
            raise HTTPException(status_code=404, detail=f"{table} id {pk} not found")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------- runs --
    @app.post("/api/runs", status_code=201)
    def create_run(body: RunCreate, db=Depends(get_db)):
        cur = db.execute(
            "INSERT INTO runs (server_url, model_fingerprint, engine, context_size, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (body.server_url, body.model_fingerprint, body.engine,
             body.context_size, body.created_at),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM runs WHERE id = ?", (cur.lastrowid,))

    @app.get("/api/runs")
    def list_runs(db=Depends(get_db)):
        return fetch_all(db, "SELECT * FROM runs ORDER BY id")

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int, db=Depends(get_db)):
        run = fetch_one(db, "SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise HTTPException(status_code=404, detail=f"run id {run_id} not found")
        return run

    # -------------------------------------------------------- benchmarks --
    @app.post("/api/runs/{run_id}/benchmarks", status_code=201)
    def add_benchmark(run_id: int, body: BenchmarkCreate, db=Depends(get_db)):
        require_parent(db, "runs", run_id)
        cur = db.execute(
            "INSERT INTO benchmarks (run_id, context_tokens, prefill_tps, decode_tps, ttft_ms, "
            "wall_s, output_tokens, mtp_draft_n, mtp_accepted, power_watts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, body.context_tokens, body.prefill_tps, body.decode_tps, body.ttft_ms,
             body.wall_s, body.output_tokens, body.mtp_draft_n, body.mtp_accepted,
             body.power_watts, body.created_at),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT rowid AS id, * FROM benchmarks WHERE rowid = ?",
                         (cur.lastrowid,))

    @app.get("/api/runs/{run_id}/benchmarks")
    def list_benchmarks(run_id: int, db=Depends(get_db)):
        require_parent(db, "runs", run_id)
        return fetch_all(
            db,
            "SELECT rowid AS id, * FROM benchmarks WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )

    # ------------------------------------------------------- eval suites --
    @app.post("/api/eval-suites", status_code=201)
    def create_eval_suite(body: EvalSuiteCreate, db=Depends(get_db)):
        cur = db.execute(
            "INSERT INTO eval_suites (name, version) VALUES (?, ?)",
            (body.name, body.version),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM eval_suites WHERE id = ?", (cur.lastrowid,))

    @app.get("/api/eval-suites")
    def list_eval_suites(db=Depends(get_db)):
        return fetch_all(db, "SELECT * FROM eval_suites ORDER BY id")

    @app.get("/api/eval-suites/{suite_id}")
    def get_eval_suite(suite_id: int, db=Depends(get_db)):
        suite = fetch_one(db, "SELECT * FROM eval_suites WHERE id = ?", (suite_id,))
        if suite is None:
            raise HTTPException(status_code=404, detail=f"eval suite id {suite_id} not found")
        return suite

    # ------------------------------------------------------- eval results --
    @app.post("/api/eval-suites/{suite_id}/results", status_code=201)
    def add_eval_result(suite_id: int, body: EvalResultCreate, db=Depends(get_db)):
        require_parent(db, "eval_suites", suite_id)
        cur = db.execute(
            "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, output, "
            "prompt_tokens, completion_tokens, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (suite_id, body.model_fingerprint, body.item_id, body.output,
             body.prompt_tokens, body.completion_tokens, body.latency_ms, body.created_at),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM eval_results WHERE id = ?", (cur.lastrowid,))

    @app.get("/api/eval-suites/{suite_id}/results")
    def list_eval_results(suite_id: int, db=Depends(get_db)):
        require_parent(db, "eval_suites", suite_id)
        return fetch_all(db, "SELECT * FROM eval_results WHERE suite_id = ? ORDER BY id",
                         (suite_id,))

    @app.get("/api/eval-results")
    def list_all_eval_results(db=Depends(get_db)):
        return fetch_all(db, "SELECT * FROM eval_results ORDER BY id")

    # ---------------------------------------------------------- judgments --
    @app.post("/api/judgments", status_code=201)
    def add_judgment(body: JudgmentCreate, db=Depends(get_db)):
        require_parent(db, "eval_results", body.eval_result_a)
        require_parent(db, "eval_results", body.eval_result_b)
        cur = db.execute(
            "INSERT INTO judgments (eval_result_a, eval_result_b, judge_model, "
            "judge_template_version, winner, confidence, rationale, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (body.eval_result_a, body.eval_result_b, body.judge_model,
             body.judge_template_version, body.winner, body.confidence,
             body.rationale, body.created_at),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM judgments WHERE id = ?", (cur.lastrowid,))

    @app.get("/api/judgments")
    def list_judgments(db=Depends(get_db)):
        return fetch_all(db, "SELECT * FROM judgments ORDER BY id")

    return app


app = create_app()
