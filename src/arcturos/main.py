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
  GET  /api/compare/benchmarks          -> metric series per run (J2)
  GET  /api/compare/run-diff/{a}/{b}    -> per-context deltas between runs (J2)
  GET  /api/reports/suite/{suite_id}    -> J5 report {report, csv} for one suite
  GET  /compare                         -> compare view HTML
  GET  /diff                            -> run-diff view HTML
  GET  /evals                           -> eval suite results view HTML
  GET  /judgments                       -> judgment list view HTML
  GET  /reports                         -> suite report view HTML
  POST /api/ops/bench                   -> kick off a bench sweep against a server (stores run+points)
  POST /api/ops/eval                    -> replay a suite against a target (stores eval_results)
  GET  /api/ops/preflight               -> J6 health checks for a target (kind=bench|eval, model?)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import compare, ops, reports
from .db import DEFAULT_DB_PATH, connect, init_db
from .schemas import (
    BaselineCreate,
    BenchmarkCreate,
    EvalResultCreate,
    EvalSuiteCreate,
    JudgmentCreate,
    ModelAliasUpdate,
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

    # Static assets (charts, JS) + the J2 views, served alongside the API.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

    def register_model(db: sqlite3.Connection, fingerprint: str,
                       engine: str | None = None) -> None:
        """Auto-register a model fingerprint on first sighting (§4.6).

        Insert-only: the registry grows as runs/eval results arrive; the
        alias is the only later edit (PATCH /api/models/{fp}).
        """
        now = _utcnow()
        db.execute(
            "INSERT INTO models (model_fingerprint, alias, engine, "
            "first_seen, last_seen) VALUES (?, NULL, ?, ?, ?) "
            "ON CONFLICT(model_fingerprint) DO UPDATE SET last_seen = ?",
            (fingerprint, engine, now, now, now),
        )

    def _utcnow() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/runs", include_in_schema=False)
    def runs_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/runs/{run_id}", include_in_schema=False)
    def run_detail_view(run_id: int):
        return FileResponse(STATIC_DIR / "run_detail.html")

    @app.get("/compare", include_in_schema=False)
    def compare_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "compare.html")

    @app.get("/diff", include_in_schema=False)
    def diff_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "diff.html")

    @app.get("/evals", include_in_schema=False)
    def evals_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "evals.html")

    @app.get("/judgments", include_in_schema=False)
    def judgments_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "judgments.html")

    @app.get("/reports", include_in_schema=False)
    def reports_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "reports.html")

    @app.get("/models", include_in_schema=False)
    def models_view() -> FileResponse:
        return FileResponse(STATIC_DIR / "models.html")

    @app.get("/baselines", include_in_schema=False)
    def baselines_view() -> FileResponse:
        return FileResponse(STATIC_DIR / "baselines.html")

    @app.get("/create", include_in_schema=False)
    def create_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "create.html")

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
        register_model(db, body.model_fingerprint, body.engine)
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
            "wall_s, output_tokens, mtp_draft_n, mtp_accepted, power_watts, power_host, "
            "power_gpu_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, body.context_tokens, body.prefill_tps, body.decode_tps, body.ttft_ms,
             body.wall_s, body.output_tokens, body.mtp_draft_n, body.mtp_accepted,
             body.power_watts, body.power_host, body.power_gpu_index, body.created_at),
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

    # ------------------------------------------------------------- reports ----
    @app.get("/api/reports/suite/{suite_id}")
    def suite_report_endpoint(suite_id: int, db=Depends(get_db)):
        """J5 report for one suite: win-rate aggregation + CSV export.

        Thin wrapper over reports.suite_report / export_report_csv; unknown
        suite ids surface as 404 (ValueError from the aggregator).
        """
        try:
            report = reports.suite_report(db, suite_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"report": report, "csv": reports.export_report_csv(report)}

    # ----------------------------------------------------------- registry --
    @app.get("/api/models")
    def list_models(db=Depends(get_db)):
        """Registry cards: fingerprint, alias, engine, first/last seen +
        live run count. Fingerprints appear in the order first seen."""
        return fetch_all(
            db,
            "SELECT m.model_fingerprint, m.alias, m.engine, "
            "m.first_seen, m.last_seen, "
            "(SELECT COUNT(*) FROM runs r WHERE r.model_fingerprint "
            " = m.model_fingerprint) AS run_count "
            "FROM models m ORDER BY m.first_seen, m.model_fingerprint",
        )

    @app.patch("/api/models/{fingerprint:path}")
    def update_model_alias(fingerprint: str, body: ModelAliasUpdate,
                           db=Depends(get_db)):
        """Rename the label only. The fingerprint is immutable (§4.6) —
        the DB trigger rejects any other column change."""
        row = fetch_one(db, "SELECT 1 AS ok FROM models WHERE model_fingerprint = ?",
                        (fingerprint,))
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"model fingerprint {fingerprint!r} not registered")
        db.execute("UPDATE models SET alias = ? WHERE model_fingerprint = ?",
                   (body.alias, fingerprint))
        db.commit()
        return fetch_one(
            db,
            "SELECT m.model_fingerprint, m.alias, m.engine, m.first_seen, "
            "m.last_seen, (SELECT COUNT(*) FROM runs r WHERE "
            "r.model_fingerprint = m.model_fingerprint) AS run_count "
            "FROM models m WHERE m.model_fingerprint = ?",
            (fingerprint,),
        )

    # ----------------------------------------------------------- baselines --
    @app.get("/api/baselines")
    def list_baselines(db=Depends(get_db)):
        """Pinned reference runs per (model, metric-family) + the pinned
        run's created_at so the view can show when the reference was taken."""
        return fetch_all(
            db,
            "SELECT b.id, b.model_fingerprint, b.metric_family, b.run_id, "
            "r.created_at AS run_created_at FROM baselines b "
            "JOIN runs r ON r.id = b.run_id ORDER BY b.id",
        )

    @app.post("/api/baselines", status_code=201)
    def pin_baseline(body: BaselineCreate, db=Depends(get_db)):
        """Pin a run as the reference for (model, metric-family).

        Re-pinning the same pair replaces the pointer (the old one is not
        data, it is a reference). The pinned run must exist and must belong
        to the same fingerprint — a baseline of the wrong model would
        silently corrupt every delta column.
        """
        run = fetch_one(db, "SELECT model_fingerprint FROM runs WHERE id = ?",
                        (body.run_id,))
        if run is None:
            raise HTTPException(status_code=404,
                                detail=f"run id {body.run_id} not found")
        if run["model_fingerprint"] != body.model_fingerprint:
            raise HTTPException(
                status_code=422,
                detail=(f"run {body.run_id} has fingerprint "
                        f"{run['model_fingerprint']!r}, not "
                        f"{body.model_fingerprint!r} — a baseline must pin "
                        "a run of the same model"),
            )
        try:
            db.execute(
                "INSERT INTO baselines (model_fingerprint, metric_family, "
                "run_id, created_at) VALUES (?, ?, ?, ?)",
                (body.model_fingerprint, body.metric_family, body.run_id,
                 body.created_at),
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            if "baseline-replace" in str(exc):
                db.execute(
                    "DELETE FROM baselines WHERE model_fingerprint = ? "
                    "AND metric_family = ?",
                    (body.model_fingerprint, body.metric_family),
                )
                db.execute(
                    "INSERT INTO baselines (model_fingerprint, "
                    "metric_family, run_id, created_at) VALUES (?, ?, ?, ?)",
                    (body.model_fingerprint, body.metric_family,
                     body.run_id, body.created_at),
                )
                db.commit()
            else:
                raise HTTPException(status_code=422,
                                    detail=f"baseline pin rejected: {exc}")
        return fetch_one(
            db,
            "SELECT b.id, b.model_fingerprint, b.metric_family, b.run_id, "
            "r.created_at AS run_created_at FROM baselines b "
            "JOIN runs r ON r.id = b.run_id WHERE b.model_fingerprint = ? "
            "AND b.metric_family = ?",
            (body.model_fingerprint, body.metric_family),
        )

    @app.delete("/api/baselines/{baseline_id}")
    def unpin_baseline(baseline_id: int, db=Depends(get_db)):
        """Remove the reference pointer only — stored run rows are never
        touched (unpinning is not a data mutation)."""
        row = fetch_one(db, "SELECT id FROM baselines WHERE id = ?", (baseline_id,))
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"baseline id {baseline_id} not found")
        db.execute("DELETE FROM baselines WHERE id = ?", (baseline_id,))
        db.commit()
        return {"unpinned": baseline_id}

    # ------------------------------------------------------------- ops ----
    @app.post("/api/ops/bench")
    def ops_bench(body: dict, db=Depends(get_db)):
        """Kick off a bench sweep: {server_url, targets: [...], n_predict, api_key?}.

        Synchronous: returns the stored run_id + per-point metrics. The UI
        shows a spinner for the duration; long sweeps use a small target
        list first (targets are context lengths, e.g. [4096, 16384]).
        ``api_key`` authenticates against key-protected servers.
        """
        server_url = body.get("server_url")
        targets = body.get("targets")
        n_predict = body.get("n_predict", 256)
        api_key = body.get("api_key")
        if not isinstance(server_url, str) or not server_url.startswith("http"):
            raise HTTPException(status_code=422, detail="server_url must be an http(s) URL")
        if not isinstance(targets, list) or not all(
                isinstance(t, int) and t >= 1 for t in targets):
            raise HTTPException(status_code=422, detail="targets must be a list of ints >= 1")
        if not isinstance(n_predict, int) or n_predict < 1:
            raise HTTPException(status_code=422, detail="n_predict must be an int >= 1")
        try:
            return ops.dispatch_bench(
                db_path=db_path, server_url=server_url,
                targets=targets, n_predict=n_predict, api_key=api_key)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ops.DispatchFailure as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"bench failed: {exc}")

    @app.post("/api/ops/eval")
    def ops_eval(body: dict):
        """Replay a suite: {suite_id, target, model_fingerprint, suite, api_key?}.

        ``suite`` is the inline JSON payload: {"items": [...]} where each item
        is {"id": ..., "prompt": ...} (single-turn) or
        {"type": "multi-turn", "id": ..., "turns": [...]} (multi-turn). The
        target must be an OpenAI-compatible chat endpoint. Results land in
        eval_results and are returned.
        """
        suite_id = body.get("suite_id")
        target = body.get("target")
        model_fingerprint = body.get("model_fingerprint")
        api_key = body.get("api_key")
        suite = body.get("suite")
        if not isinstance(suite_id, int) or suite_id < 1:
            raise HTTPException(status_code=422, detail="suite_id must be a positive int")
        if not isinstance(target, str) or not target.startswith("http"):
            raise HTTPException(status_code=422, detail="target must be an http(s) URL")
        if not isinstance(model_fingerprint, str) or not model_fingerprint:
            raise HTTPException(status_code=422, detail="model_fingerprint required")
        if not isinstance(suite, dict):
            raise HTTPException(status_code=422, detail="suite must be an inline JSON object with 'items'")
        try:
            return ops.dispatch_eval(
                db_path=db_path, suite_id=suite_id, target=target,
                model_fingerprint=model_fingerprint, suite=suite,
                api_key=api_key)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ops.DispatchFailure as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:  # target unreachable/HTTP error/etc.
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=502, detail=f"eval replay failed: {exc}")

    @app.post("/api/ops/preflight")
    def ops_preflight(body: dict):
        """J6 interactive health preflight: POST JSON
        {target, kind=bench|eval, model?, api_key?} -> checks.

        POST with the key in the body (never in a URL: GET query strings
        leak into access logs, browser history, and proxies — a credential
        field the UI renders password-masked must not leak this way).
        Shorter per-check timeout than the dispatch path (8s vs 20s) so a
        dead target answers in <= ~16s.
        """
        target = body.get("target")
        kind = body.get("kind", "bench")
        model = body.get("model")
        api_key = body.get("api_key")
        if not isinstance(target, str) or not target.startswith("http"):
            raise HTTPException(status_code=422, detail="target must be an http(s) URL")
        if kind not in ("bench", "eval"):
            raise HTTPException(status_code=422, detail="kind must be 'bench' or 'eval'")
        try:
            return ops.preflight.preflight(target, kind=kind, model=model,
                                           api_key=api_key, timeout=8.0)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"preflight failed: {exc}")

    # Kept for keyless GET probes (no credentials travel in URLs when the
    # key is required; keyless probes are fine and keep bookmarks working).
    @app.get("/api/ops/preflight")
    def ops_preflight_get(target: str, kind: str = "bench", model: str | None = None):
        if not target.startswith("http"):
            raise HTTPException(status_code=422, detail="target must be an http(s) URL")
        if kind not in ("bench", "eval"):
            raise HTTPException(status_code=422, detail="kind must be 'bench' or 'eval'")
        try:
            return ops.preflight.preflight(target, kind=kind, model=model,
                                           api_key=None, timeout=8.0)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"preflight failed: {exc}")

    # ------------------------------------------------------------ compare ----
    def parse_run_ids(raw: str) -> list[int]:
        """Comma-separated run ids -> ints, deduped, input order preserved."""
        ids: list[int] = []
        seen: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token.isdigit():
                raise HTTPException(
                    status_code=422, detail=f"invalid run id: {token!r}"
                )
            value = int(token)
            if value not in seen:
                seen.add(value)
                ids.append(value)
        if not ids:
            raise HTTPException(status_code=422, detail="at least one run id required")
        return ids

    def parse_metrics(raw: str | None) -> list[str]:
        """Comma-separated metrics validated against the allowlist.

        Missing/empty param defaults to every allowlisted metric. Unknown
        names are a 422 (allowlist contract, never silently dropped).
        """
        if raw is None or not raw.strip():
            return list(compare.ALLOWED_METRICS)
        metrics: list[str] = []
        seen: set[str] = set()
        for token in raw.split(","):
            token = token.strip()
            if token not in compare.ALLOWED_METRICS:
                raise HTTPException(
                    status_code=422, detail=f"unknown metric: {token!r}"
                )
            if token not in seen:
                seen.add(token)
                metrics.append(token)
        if not metrics:
            raise HTTPException(status_code=422, detail="at least one metric required")
        return metrics

    @app.get("/api/compare/benchmarks")
    def compare_benchmarks(runs: str, metrics: str | None = None,
                           baseline: int | None = None,
                           format: str | None = None, db=Depends(get_db)):
        """Series of every requested metric per run, aligned by context length.

        ``baseline=<run_id>`` overlays a pinned reference: the payload gains
        ``baseline`` metadata + ``series.baseline`` per metric (§4.6), so
        charts draw the dashed reference line and delta tables have a base.
        ``format=csv`` returns the same payload as one CSV document
        (schema-versioned header + fingerprint columns, §4.6 export) with
        Content-Disposition attachment so the browser downloads it.
        """
        run_ids = parse_run_ids(runs)
        metric_list = parse_metrics(metrics)
        for rid in run_ids:
            if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (rid,)) is None:
                raise HTTPException(status_code=404, detail=f"run id {rid} not found")
        if baseline is not None:
            if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (baseline,)) is None:
                raise HTTPException(status_code=404,
                                    detail=f"baseline run id {baseline} not found")
        payload = compare.build_benchmarks_payload(db, run_ids, metric_list,
                                                   baseline_run_id=baseline)
        if format == "csv":
            csv_text = compare.export_benchmarks_csv(payload)
            from fastapi.responses import Response
            return Response(
                csv_text,
                media_type="text/csv",
                headers={"Content-Disposition":
                         "attachment; filename=arcturos-compare.csv"},
            )
        if format is not None:
            raise HTTPException(status_code=422,
                                detail=f"unsupported format {format!r} (try csv)")
        return payload

    @app.get("/api/compare/baseline-deltas")
    def compare_baseline_deltas(runs: str, run: int,
                                metrics: str | None = None,
                                baseline: int | None = None,
                                db=Depends(get_db)):
        """Per-run delta tables against a baseline reference (§4.6).

        Returns ``{run_id: {metric: [rows]}}`` — rows only where BOTH the
        run and the baseline have the metric at the same context_tokens
        (a delta against nothing is not a number; honest nulls beat
        fabricated zeros).
        """
        run_ids = parse_run_ids(runs)
        if run not in run_ids:
            raise HTTPException(status_code=422,
                                detail=f"run {run} must be one of the compared runs")
        metric_list = parse_metrics(metrics)
        base = baseline if baseline is not None else run
        payload = compare.build_benchmarks_payload(
            db, run_ids, metric_list, baseline_run_id=base)
        result = {str(rid): compare.baseline_deltas(payload, rid)
                  for rid in run_ids}
        return result

    # ------------------------------------------------------------- export --
    def _csv_response(rows: list[dict], columns: list[str],
                      filename: str) -> "Response":
        """Schema-versioned CSV (§4.6): every row carries schema_version +
        the self-describing columns the view already shows. Honest nulls:
        missing values are empty cells."""
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        writer = _csv.writer(buf, lineterminator="\n")
        writer.writerow(["schema_version"] + columns)
        for row in rows:
            writer.writerow(["arcturos-v1"] + [row.get(col) for col in columns])
        return Response(
            buf.getvalue(), media_type="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename={filename}"})

    @app.get("/api/export/runs")
    def export_runs(db=Depends(get_db)):
        """CSV of every stored run row (§4.6 per-view export button)."""
        rows = fetch_all(
            db,
            "SELECT id AS run_id, server_url, model_fingerprint, engine, "
            "context_size, created_at FROM runs ORDER BY id",
        )
        return _csv_response(
            rows,
            ["run_id", "server_url", "model_fingerprint", "engine",
             "context_size", "created_at"],
            "arcturos-runs.csv",
        )

    @app.get("/api/export/runs/{run_id}/benchmarks")
    def export_run_benchmarks(run_id: int, db=Depends(get_db)):
        """CSV of one run's benchmark points (§4.6)."""
        if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (run_id,)) is None:
            raise HTTPException(status_code=404, detail=f"run id {run_id} not found")
        rows = fetch_all(
            db,
            "SELECT b.*, r.model_fingerprint, r.engine, r.server_url "
            "FROM benchmarks b JOIN runs r ON r.id = b.run_id "
            "WHERE b.run_id = ? ORDER BY b.context_tokens, b.rowid",
            (run_id,),
        )
        return _csv_response(
            rows,
            ["run_id", "model_fingerprint", "engine", "server_url",
             "context_tokens", "prefill_tps", "decode_tps", "ttft_ms",
             "wall_s", "output_tokens", "mtp_draft_n", "mtp_accepted",
             "power_watts", "power_host", "power_gpu_index", "created_at"],
            f"arcturos-run-{run_id}-benchmarks.csv",
        )

    @app.get("/api/export/eval-results")
    def export_eval_results(suite_id: int | None = None, db=Depends(get_db)):
        """CSV of eval results (optionally one suite's), with judgments
        joined where they exist (§4.6)."""
        if suite_id is not None:
            require_parent(db, "eval_suites", suite_id)
            rows = fetch_all(
                db,
                "SELECT er.*, s.name AS suite_name, s.version AS suite_version "
                "FROM eval_results er JOIN eval_suites s ON s.id = er.suite_id "
                "WHERE er.suite_id = ? ORDER BY er.id",
                (suite_id,),
            )
        else:
            rows = fetch_all(
                db,
                "SELECT er.*, s.name AS suite_name, s.version AS suite_version "
                "FROM eval_results er JOIN eval_suites s ON s.id = er.suite_id "
                "ORDER BY er.id",
            )
        return _csv_response(
            rows,
            ["result_id", "suite_id", "suite_name", "suite_version",
             "model_fingerprint", "item_id", "output", "prompt_tokens",
             "completion_tokens", "latency_ms", "created_at"],
            "arcturos-eval-results.csv",
        )

    @app.get("/api/export/judgments")
    def export_judgments(db=Depends(get_db)):
        """CSV of judgments with both models' fingerprints resolved (§4.6)."""
        rows = fetch_all(
            db,
            "SELECT j.*, ra.model_fingerprint AS fp_a, "
            "rb.model_fingerprint AS fp_b FROM judgments j "
            "JOIN eval_results ra ON ra.id = j.eval_result_a "
            "JOIN eval_results rb ON rb.id = j.eval_result_b "
            "ORDER BY j.id",
        )
        return _csv_response(
            rows,
            ["judgment_id", "eval_result_a", "eval_result_b", "fp_a", "fp_b",
             "judge_model", "judge_template_version", "winner", "confidence",
             "rationale", "created_at"],
            "arcturos-judgments.csv",
        )

    @app.get("/api/compare/run-diff/{run_a}/{run_b}")
    def run_diff(run_a: int, run_b: int, db=Depends(get_db)):
        """Per-context delta table between two runs + engine metadata diff."""
        for rid in (run_a, run_b):
            if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (rid,)) is None:
                raise HTTPException(status_code=404, detail=f"run id {rid} not found")
        return compare.build_run_diff_payload(db, run_a, run_b)

    return app


app = create_app()
