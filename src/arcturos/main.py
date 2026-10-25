"""Arcturos dashboard — FastAPI app (JSON API + static index.html).

Run with:  uvicorn arcturos.main:app --host 0.0.0.0 --port 24816

Routes (all data paths are append-only — no UPDATE/DELETE exists):
  GET  /                                -> dashboard HTML
  GET  /health                          -> {"status": "ok"}
  POST /api/runs                        -> create run
  GET  /api/runs                        -> list runs
  GET  /api/runs/{run_id}               -> fetch one run
  PUT  /api/runs/{run_id}/visibility    -> soft-hide/unhide a run (append-only flag; hidden runs drop out of list views)
  GET  /api/runs/{run_id}/visibility    -> current visibility flag
  POST /api/runs/{run_id}/benchmarks    -> add benchmark row to a run
  GET  /api/runs/{run_id}/benchmarks    -> list benchmarks for a run
  POST /api/eval-suites                 -> create eval suite
  GET  /api/eval-suites                 -> list eval suites
  GET  /api/eval-suites/{suite_id}      -> fetch one suite
  PUT  /api/eval-suites/{suite_id}/definition -> snapshot suite definition JSON
  GET  /api/eval-suites/{suite_id}/definition -> latest definition snapshot
  POST /api/eval-suites/{suite_id}/results -> add eval result
  GET  /api/eval-suites/{suite_id}/results -> list results for a suite
  GET  /api/eval-results                -> list all eval results
  POST /api/judgments                   -> add judgment
  GET  /api/judgments                   -> list judgments
  GET  /api/models                      -> model registry cards
  PATCH /api/models/{fingerprint}       -> rename the alias label only
  GET  /api/baselines                   -> pinned reference runs
  POST /api/baselines                   -> pin a run as baseline
  DELETE /api/baselines/{baseline_id}   -> unpin (reference pointer only)
  GET  /api/compare/benchmarks          -> metric series per run (J2)
  GET  /api/compare/run-diff/{a}/{b}    -> per-context deltas between runs (J2)
  GET  /api/compare/baseline-deltas     -> per-run delta tables vs a baseline reference
  GET  /api/reports/suite/{suite_id}    -> J5 report {report, csv} for one suite
  GET  /compare                         -> compare view HTML
  GET  /diff                            -> run-diff view HTML
  GET  /evals                           -> eval suite results view HTML
  GET  /judgments                       -> judgment list view HTML
  GET  /reports                         -> suite report view HTML
  POST /api/ops/bench                   -> kick off a bench sweep against a server (stores run+points; transport native|openai, model?)
  POST /api/ops/bench/jobs              -> async bench job: returns {job_id} immediately, sweep runs in background
  GET  /api/ops/bench/jobs/{job_id}     -> job status: phase, per-point results, elapsed + ETA
  POST /api/ops/eval                    -> replay a suite against a target (stores eval_results)
  POST /api/ops/eval/jobs               -> async eval replay job: {job_id} immediately, per-item progress
  GET  /api/ops/eval/jobs/{job_id}      -> eval job status: phase, per-item rows, elapsed + ETA
  POST /api/ops/eval/jobs/{id}/abort    -> cooperative stop (completed items stay stored)
  POST /api/ops/judge/jobs              -> async blind A/B judge job (position swap, both orders stored)
  GET  /api/ops/judge/jobs/{job_id}     -> judge job status
  POST /api/ops/judge/jobs/{id}/abort   -> cooperative stop of a judge job
  GET  /evals/run                       -> eval replay job page (async, live progress)
  GET  /api/ops/preflight               -> J6 health checks for a target (kind=bench|eval, model?)
  GET  /api/export/runs                 -> CSV export: all runs
  GET  /api/export/runs/{id}/benchmarks -> CSV export: one run's points
  GET  /api/export/eval-results         -> CSV export: eval results
  GET  /api/export/judgments            -> CSV export: judgments
  GET  /models                          -> model registry view HTML
  GET  /baselines                       -> baselines view HTML
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import compare, ops, reports
from . import config
from . import evals as evals_mod
from . import multiturn as multiturn_mod
from . import judging
from . import pagelib
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


def page_response(name: str) -> Response:
    """Serve a static page with the canonical nav partial injected
    (pagelib.render_page — one source of truth for the menu)."""
    return Response(pagelib.render_page(name), media_type="text/html")


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
                       engine: str | None = None,
                       seen_at: str | None = None) -> None:
        """Auto-register a model fingerprint on first sighting (§4.6).

        Insert-only: the registry grows as runs/eval results arrive; the
        alias is the only later edit (PATCH /api/models/{fp}). ``seen_at``
        pins first_seen/last_seen to the triggering row's own timestamp —
        one clock read, so registry provenance matches the run row exactly
        (a separate _utcnow() read could land a millisecond later and make
        first_seen != created_at).
        """
        now = seen_at or _utcnow()
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
    def index():
        # Merged Compare Runs page (2026-10-11): the former /runs list and
        # /compare are one page now; the root lands on it.
        return RedirectResponse("/compare", status_code=308)

    @app.get("/runs", include_in_schema=False)
    def runs_page():
        # Runs list merged into Compare Runs (2026-10-11): permanent
        # redirect keeps old links and bookmarks working.
        return RedirectResponse("/compare", status_code=308)

    @app.get("/runs/{run_id}", include_in_schema=False)
    def run_detail_view(run_id: int):
        return page_response("run_detail.html")

    @app.get("/compare", include_in_schema=False)
    def compare_page() -> Response:
        return page_response("compare.html")

    # Run Diff integrated into the Compare page (owner call 2026-10-11):
    # the A/B tables render under the charts, so /diff is a permanent
    # redirect — old links and the run-detail deep-link keep working.
    @app.get("/diff", include_in_schema=False)
    def diff_page() -> RedirectResponse:
        return RedirectResponse("/compare", status_code=308)

    # Split pages (2026-09-30 UX round 2): every kick-off form is its own
    # URL so the sidebar can link it directly — no anchor fragments.
    @app.get("/kickoff", include_in_schema=False)
    def kickoff_page() -> Response:
        return page_response("bench-kickoff.html")

    @app.get("/evals/suites/new", include_in_schema=False)
    def eval_suite_new_page() -> Response:
        return page_response("eval-suite-new.html")

    @app.get("/evals", include_in_schema=False)
    def evals_page() -> Response:
        return page_response("evals.html")

    @app.get("/evals/run", include_in_schema=False)
    def eval_run_page() -> Response:
        # v0.2 (2026-10-15): replay moved off /create onto its own async
        # job page (live per-item progress + operator stop).
        return page_response("eval-run.html")

    @app.get("/evals/runs/{run_id}", include_in_schema=False)
    def eval_run_detail_page(run_id: int) -> Response:
        # Drill-down (2026-10-15): one replay run's stored rows. Reached
        # from the Results page's run table; data-page="evalrundetail".
        return page_response("eval-run-detail.html")

    @app.get("/judgments/pair", include_in_schema=False)
    def judgment_pair_page() -> Response:
        # Drill-down (2026-10-15): SxS judgment detail for one suite +
        # model pair (?suite_id=&model_a=&model_b=). data-page="jpair".
        return page_response("judgment-detail.html")

    @app.get("/judgments", include_in_schema=False)
    def judgments_page() -> Response:
        return page_response("judgments.html")

    @app.get("/judgments/{judgment_id}", include_in_schema=False)
    def judgment_by_id_page(judgment_id: int) -> Response:
        # Drill-down (2026-10-25): ONE judgment addressed by its own
        # primary key — /judgments/2903 — replacing the foreign-key
        # query soup (?suite_id=&model_a=&model_b=) as the way a single
        # judgment is opened. data-page="judgment".
        return page_response("judgment.html")

    @app.get("/reports", include_in_schema=False)
    def reports_page() -> RedirectResponse:
        # Reports page dropped as unused (owner call 2026-10-25); the
        # /api/reports/* endpoints stay. Permanent redirect keeps old
        # bookmarks working — same recipe as /runs → /compare.
        return RedirectResponse("/evals/run", status_code=308)

    @app.get("/models", include_in_schema=False)
    def models_view() -> Response:
        return page_response("models.html")

    @app.get("/baselines", include_in_schema=False)
    def baselines_view() -> Response:
        return page_response("baselines.html")

    @app.get("/create", include_in_schema=False)
    def create_page() -> Response:
        # Legacy path (2026-09-30 split): serves the bench start page.
        # 2026-10-11: was serving a frozen create.html copy that drifted
        # from bench-kickoff.html (old headings, old form) — both routes
        # now serve the one canonical page.
        return page_response("bench-kickoff.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------- runs --
    # Hidden-run filter: a run is hidden when its newest run_visibility flag
    # says so (append-only soft delete — the rows stay, list views skip it).
    _HIDDEN_RUN_SQL = (
        "SELECT 1 FROM run_visibility v WHERE v.run_id = runs.id "
        "AND v.hidden = 1 AND v.id = (SELECT MAX(id) FROM run_visibility "
        "WHERE run_id = runs.id)"
    )

    @app.put("/api/runs/{run_id}/visibility", status_code=204)
    def put_run_visibility(run_id: int, body: dict, db=Depends(get_db)):
        """Soft-hide (hidden=true) or unhide (hidden=false) a run.

        Append-only: each call appends a new flag row; the newest flag per
        run wins. Nothing stored is mutated — hidden runs remain fully
        addressable (direct fetch, exports, judgments) and can be
        unhidden at any time.
        """
        require_parent(db, "runs", run_id)
        hidden = body.get("hidden")
        if not isinstance(hidden, bool):
            raise HTTPException(status_code=422,
                                detail="body must be {\"hidden\": true|false}")
        reason = body.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise HTTPException(status_code=422, detail="reason must be a string")
        db.execute(
            "INSERT INTO run_visibility (run_id, hidden, reason, created_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, 1 if hidden else 0, reason, _utcnow()),
        )
        db.commit()
        return Response(status_code=204)

    @app.get("/api/config/ui-defaults", include_in_schema=False)
    def ui_defaults():
        """Non-secret UI boot values (arcturos.toml / local overlay).

        api_key is deliberately NOT exposed — it flows server-side only.
        """
        return {
            "server_url": config.get("bench", "server_url",
                                     "http://localhost:8000"),
            "model": config.get("bench", "model", ""),
            "eval_target": config.get("demo", "judge_url", ""),
        }

    @app.get("/api/runs/{run_id}/visibility")
    def get_run_visibility(run_id: int, db=Depends(get_db)):
        require_parent(db, "runs", run_id)
        row = db.execute(
            "SELECT hidden, reason, created_at FROM run_visibility "
            "WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
        if row is None:
            return {"run_id": run_id, "hidden": False, "reason": None,
                    "created_at": None}
        return {"run_id": run_id, "hidden": bool(row["hidden"]),
                "reason": row["reason"], "created_at": row["created_at"]}

    @app.post("/api/runs", status_code=201)
    def create_run(body: RunCreate, db=Depends(get_db)):
        cur = db.execute(
            "INSERT INTO runs (server_url, model_fingerprint, engine, context_size, name, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body.server_url, body.model_fingerprint, body.engine,
             body.context_size, body.name, body.created_at),
        )
        register_model(db, body.model_fingerprint, body.engine,
                       seen_at=body.created_at)
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM runs WHERE id = ?", (cur.lastrowid,))

    @app.patch("/api/runs/{run_id}/name", status_code=200)
    def rename_run(run_id: int, body: dict, db=Depends(get_db)):
        """Rename a run (2026-09-22 naming feature).

        The name is a LABEL, not measurement data: the append-only trigger
        allows exactly this column to change (mirrors the models.alias
        precedent). Set `{"name": null}` to clear a custom name back to
        the default rendering; `{"name": "..."}` sets it. Measurement
        columns stay immutable — the trigger aborts anything else.
        """
        name = body.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise HTTPException(status_code=422,
                                detail="name must be a non-empty string or null")
        if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (run_id,)) is None:
            raise HTTPException(status_code=404, detail=f"run id {run_id} not found")
        db.execute("UPDATE runs SET name = ? WHERE id = ?",
                   (name.strip() if isinstance(name, str) else None, run_id))
        db.commit()
        run = fetch_one(db, "SELECT * FROM runs WHERE id = ?", (run_id,))
        run["display_name"] = run.get("name") or _unnamed_run_label(run)
        return run

    @app.get("/api/runs")
    def list_runs(db=Depends(get_db), include_hidden: bool = False,
                  status: str | None = None):
        # status filter (2026-09-22 partial-run support): 'partial' returns
        # only partial runs; 'complete' only complete ones; omitted = all.
        where = ["NOT EXISTS (" + _HIDDEN_RUN_SQL + ")"]
        params: list = []
        if status is not None:
            if status not in ("complete", "partial"):
                raise HTTPException(
                    status_code=422,
                    detail="status must be 'complete' or 'partial'")
            where.append("status = ?")
            params.append(status)
        if include_hidden:
            base = "1=1"
        else:
            base = "NOT EXISTS (" + _HIDDEN_RUN_SQL + ")"
        where = [w for w in where if not w.startswith("NOT EXISTS")]
        where.insert(0, base)
        sql = "SELECT * FROM runs WHERE " + " AND ".join(where) + " ORDER BY id"
        rows = fetch_all(db, sql, tuple(params))
        # Display name (2026-09-22 naming UX round): the Runs page must
        # show a usable label for EVERY run, including legacy rows that
        # predate the name column. A stored name is authoritative; an
        # unnamed run falls back to 'host · model-short' computed at read
        # time (no data mutation — append-only rows stay untouched).
        for row in rows:
            row["display_name"] = row.get("name") or _unnamed_run_label(row)
        return rows

    def _unnamed_run_label(row: dict) -> str:
        host = compare.host_label(row["server_url"])
        model = compare.short_fingerprint(row["model_fingerprint"])
        return f"{host} · {model}"

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int, db=Depends(get_db)):
        run = fetch_one(db, "SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise HTTPException(status_code=404, detail=f"run id {run_id} not found")
        run["display_name"] = run.get("name") or _unnamed_run_label(run)
        return run

    # -------------------------------------------------------- benchmarks --
    @app.post("/api/runs/{run_id}/benchmarks", status_code=201)
    def add_benchmark(run_id: int, body: BenchmarkCreate, db=Depends(get_db)):
        require_parent(db, "runs", run_id)
        cur = db.execute(
            "INSERT INTO benchmarks (run_id, context_tokens, prefill_tps, decode_tps, ttft_ms, "
            "wall_s, output_tokens, mtp_draft_n, mtp_accepted, power_watts, power_host, "
            "power_gpu_index, streams, decode_tps_combined, prefill_tps_combined, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, body.context_tokens, body.prefill_tps, body.decode_tps, body.ttft_ms,
             body.wall_s, body.output_tokens, body.mtp_draft_n, body.mtp_accepted,
             body.power_watts, body.power_host, body.power_gpu_index, body.streams,
             body.decode_tps_combined, body.prefill_tps_combined, body.created_at),
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
        # Most recent first — the run/judge pickers surface fresh suites.
        return fetch_all(db, "SELECT * FROM eval_suites ORDER BY id DESC")

    @app.get("/api/eval-suites/{suite_id}")
    def get_eval_suite(suite_id: int, db=Depends(get_db)):
        suite = fetch_one(db, "SELECT * FROM eval_suites WHERE id = ?", (suite_id,))
        if suite is None:
            raise HTTPException(status_code=404, detail=f"eval suite id {suite_id} not found")
        return suite

    @app.put("/api/eval-suites/{suite_id}/definition", status_code=204)
    def put_suite_definition(suite_id: int, body: dict, db=Depends(get_db)):
        """Snapshot a suite definition (items, categories, prompts).

        Append-only snapshots: each PUT appends a new row; report-time
        category joins read the latest one. Re-seeding a suite is a new
        snapshot, never an in-place edit.
        """
        require_parent(db, "eval_suites", suite_id)
        if not isinstance(body, dict) or not body.get("items"):
            raise HTTPException(status_code=422,
                                detail="definition must be an object with a non-empty 'items' array")
        db.execute(
            "INSERT INTO suite_definitions (suite_id, payload, created_at) "
            "VALUES (?, ?, ?)",
            (suite_id, json.dumps(body), _utcnow()),
        )
        db.commit()
        return Response(status_code=204)

    @app.get("/api/eval-suites/{suite_id}/definition")
    def get_suite_definition(suite_id: int, db=Depends(get_db)):
        """Latest definition snapshot for a suite (404 if never snapshotted)."""
        require_parent(db, "eval_suites", suite_id)
        row = db.execute(
            "SELECT payload FROM suite_definitions WHERE suite_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (suite_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"no definition snapshot for suite {suite_id}")
        return json.loads(row["payload"])

    @app.get("/api/eval-suites/{suite_id}/image")
    def get_suite_item_image(suite_id: int, item: str, db=Depends(get_db)):
        """Data URL for one item's suite image (prompt expander on the
        judgment-details page).

        The image ref MUST appear as an item's ``image`` in the suite's
        latest definition snapshot — the endpoint never accepts raw paths
        from the query string, so there is no arbitrary filesystem read
        (evals.load_image_part's local-path candidates are dispatch-time
        only). http(s) refs pass through as ``url`` (the browser fetches
        them directly); local refs are inlined as base64 ``data_url`` —
        same resolution rules as replay (SUITE_ASSETS_DIR, then repo
        root). 404 is honest and actionable at every miss.
        """
        require_parent(db, "eval_suites", suite_id)
        row = db.execute(
            "SELECT payload FROM suite_definitions WHERE suite_id = ? "
            "ORDER BY id DESC LIMIT 1", (suite_id,)).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"no definition snapshot for suite {suite_id}")
        definition = json.loads(row["payload"])
        item_row = next(
            (i for i in (definition.get("items") or [])
             if isinstance(i, dict) and i.get("id") == item),
            None)
        if item_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"item {item!r} not in suite {suite_id}'s definition")
        ref = item_row.get("image")
        if not ref or not isinstance(ref, str):
            raise HTTPException(
                status_code=404,
                detail=f"item {item!r} has no image in its definition")
        if ref.startswith(("http://", "https://")):
            return {"url": ref, "data_url": None}
        try:
            data_url = evals_mod.image_data_url(ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — decode/IO failures → 502
            raise HTTPException(status_code=502,
                                detail=f"image load failed: {exc}")
        return {"url": None, "data_url": data_url}

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

    # --------------------------------------------------------- eval runs --
    @app.get("/api/eval-runs")
    def list_eval_runs(db=Depends(get_db)):
        """Replay runs, newest first — the Results page's primary table.

        Each row: run id, suite, the run's model fingerprint, run label,
        lifecycle status, stored result count, created_at. A run id links
        to /evals/runs/{id}; selecting two runs judges them against each
        other (eval_run_ids-scoped judge job).
        """
        return fetch_all(db, """
            SELECT r.id, r.suite_id, r.model_fingerprint, r.name, r.status,
                   r.created_at, s.name AS suite_name, s.version AS suite_version,
                   (SELECT COUNT(*) FROM eval_results er
                     WHERE er.eval_run_id = r.id) AS result_count
            FROM eval_runs r JOIN eval_suites s ON s.id = r.suite_id
            ORDER BY r.id DESC
        """)

    @app.get("/api/eval-runs/{run_id}")
    def get_eval_run(run_id: int, db=Depends(get_db)):
        """One replay run + its stored result rows (run detail page)."""
        run = fetch_one(db, """
            SELECT r.*, s.name AS suite_name, s.version AS suite_version
            FROM eval_runs r JOIN eval_suites s ON s.id = r.suite_id
            WHERE r.id = ?""", (run_id,))
        if run is None:
            raise HTTPException(status_code=404,
                                detail=f"eval run {run_id} not found")
        results = fetch_all(db, """
            SELECT id, item_id, output, prompt_tokens, completion_tokens,
                   latency_ms, created_at, eval_run_id
            FROM eval_results WHERE eval_run_id = ? ORDER BY id""", (run_id,))
        return {"run": run, "results": results}

    # ------------------------------------------------- judgment SxS view --
    @app.get("/api/judgments/pair")
    def judgments_pair(suite_id: int, model_a: str, model_b: str,
                       db=Depends(get_db)):
        """Side-by-side judgment rows for one suite + model pair (any order).

        Each row joins the judgment with BOTH stored responses so the
        detail page renders the SxS diff. `fingerprint_a` names whose
        response is 'a' in that row — winner 'a' always means
        fingerprint_a's response won, whatever order the caller passed.
        """
        require_parent(db, "eval_suites", suite_id)
        return fetch_all(db, """
            SELECT j.id, j.judge_model,
                   j.judge_template_version AS template,
                   j.winner, j.verdict_detail, j.confidence, j.rationale,
                   j.presentation, j.raw_reply, j.role, j.created_at,
                   ra.item_id AS item_id,
                   ra.model_fingerprint AS fingerprint_a,
                   rb.model_fingerprint AS fingerprint_b,
                   ra.output AS output_a, rb.output AS output_b,
                   ra.id AS result_a_id, rb.id AS result_b_id
            FROM judgments j
            JOIN eval_results ra ON ra.id = j.eval_result_a
            JOIN eval_results rb ON rb.id = j.eval_result_b
            WHERE ra.suite_id = ?
              AND ((ra.model_fingerprint = ? AND rb.model_fingerprint = ?)
                OR (ra.model_fingerprint = ? AND rb.model_fingerprint = ?))
            ORDER BY ra.item_id, j.id
        """, (suite_id, model_a, model_b, model_b, model_a))

    @app.get("/api/judgments/pair/summary")
    def judgments_pair_summary(suite_id: int, model_a: str, model_b: str,
                               judge_model: str | None = None,
                               judge_template: str | None = None,
                               db=Depends(get_db)):
        """Per-dimension category summary for one suite + model pair.

        Feeds the pair page's headline card (details stay in the SxS list
        below). Winner orientation is normalized to the CALLER's order:
        'a' in the response always means the caller's ``model_a`` won,
        whatever canonical presentation order the judgment rows were
        stored in — swapping model_a/model_b mirrors the a/b columns
        without changing n, ties, or the row set. ``judge_model`` /
        ``judge_template`` scope the aggregate to one judge generation.
        """
        require_parent(db, "eval_suites", suite_id)
        return reports.pair_summary(db, suite_id, model_a, model_b,
                                    judge_model=judge_model,
                                    judge_template=judge_template)

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
            "judge_template_version, winner, verdict_detail, confidence, "
            "rationale, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (body.eval_result_a, body.eval_result_b, body.judge_model,
             body.judge_template_version, body.winner, body.verdict_detail,
             body.confidence, body.rationale, body.created_at),
        )
        db.commit()  # durable before the response is sent (teardown runs after send)
        return fetch_one(db, "SELECT * FROM judgments WHERE id = ?", (cur.lastrowid,))

    @app.get("/api/judgments")
    def list_judgments(db=Depends(get_db)):
        return fetch_all(db, "SELECT * FROM judgments ORDER BY id")

    @app.get("/api/judgments/{judgment_id}")
    def get_judgment(judgment_id: int, db=Depends(get_db)):
        """One judgment by its own primary key — the drill-down surface.

        The judgment's id is the only address: the response joins BOTH
        stored results (outputs, fingerprints, item, suite, run ids) so
        the detail page renders the full SxS without the caller knowing
        any foreign key. 404 for unknown ids.
        """
        row = fetch_one(db, """
            SELECT j.id, j.judge_model,
                   j.judge_template_version AS template,
                   j.winner, j.verdict_detail, j.confidence, j.rationale,
                   j.presentation, j.raw_reply, j.role, j.created_at,
                   j.eval_result_a AS result_a_id,
                   j.eval_result_b AS result_b_id,
                   ra.suite_id AS suite_id,
                   ra.item_id AS item_id,
                   ra.model_fingerprint AS fingerprint_a,
                   rb.model_fingerprint AS fingerprint_b,
                   ra.output AS output_a, rb.output AS output_b,
                   ra.eval_run_id AS run_a_id, rb.eval_run_id AS run_b_id
            FROM judgments j
            JOIN eval_results ra ON ra.id = j.eval_result_a
            JOIN eval_results rb ON rb.id = j.eval_result_b
            WHERE j.id = ?
        """, (judgment_id,))
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"judgment {judgment_id} not found")
        return row

    # ------------------------------------------------------------- reports ----
    @app.get("/api/reports/suite/{suite_id}")
    def suite_report_endpoint(suite_id: int,
                              judge_model: str | None = None,
                              judge_template: str | None = None,
                              db=Depends(get_db)):
        """J5 report for one suite: win-rate aggregation + CSV export.

        Thin wrapper over reports.suite_report / export_report_csv; unknown
        suite ids surface as 404 (ValueError from the aggregator).
        Categories/dimensions: the DB stores item_id only, so the maps are
        rebuilt from the suite definition JSON the operator seeded (if
        present). ``judge_model``/``judge_template`` scope every aggregate
        to ONE judge generation (v0.2, ADR 007 §6); both None aggregates
        all judgments.
        """
        try:
            report = reports.suite_report(
                db, suite_id,
                category_map=reports.category_map_for(db, suite_id),
                dimension_map=reports.dimension_map_for(db, suite_id),
                judge_model=judge_model, judge_template=judge_template)
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
        """Kick off a bench sweep: {server_url, targets: [...], n_predict,
        api_key?, transport?, model?, engine?, name?}.

        Synchronous: returns the stored run_id + per-point metrics. The UI
        shows a spinner for the duration; long sweeps use a small target
        list first (targets are context lengths, e.g. [4096, 16384]).
        ``api_key`` authenticates against key-protected servers.

        ``transport`` selects the wire protocol: ``'native'`` (default,
        llama.cpp /tokenize + /completion) or ``'openai'`` (any
        /v1/chat/completions server — tabbyAPI, vLLM, litellm). The
        ``openai`` transport runs preflight as reachable + chat (these
        servers have no /tokenize) and streams each point for client-side
        TTFT + decode; it requires ``model`` (the chat model name).
        ``model`` is ignored for the native transport. The UI no longer
        sends ``transport`` (2026-10-11): it autodetects server-side —
        llama.cpp /tokenize first, OpenAI chat fallback — but the
        parameter stays valid for API users.

        ``engine`` (owner request 2026-10-11) is the operator's freeform
        engine tag stored on the run row ('llama.cpp', 'vLLM',
        'tabbyAPI'…); when set AND the run is unnamed, the default run
        name becomes 'Engine / Model' (falling back to
        'host · model · targets' when the engine or fingerprint is
        unknown).
        """
        server_url = body.get("server_url")
        targets = body.get("targets")
        n_predict = body.get("n_predict", 1024)
        api_key = body.get("api_key")
        transport = body.get("transport", "native")
        model = body.get("model")
        engine = body.get("engine")
        streams = body.get("streams", 1)
        power_host = body.get("power_host")
        power_gpu_index = body.get("power_gpu_index", 0)
        name = body.get("name")
        if not isinstance(server_url, str) or not server_url.startswith("http"):
            raise HTTPException(status_code=422, detail="server_url must be an http(s) URL")
        if not isinstance(targets, list) or not all(
                isinstance(t, int) and t >= 1 for t in targets):
            raise HTTPException(status_code=422, detail="targets must be a list of ints >= 1")
        if not isinstance(n_predict, int) or n_predict < 1:
            raise HTTPException(status_code=422, detail="n_predict must be an int >= 1")
        if not isinstance(transport, str) or transport not in ("native", "openai", "auto"):
            raise HTTPException(status_code=422,
                                detail="transport must be 'native', 'openai' or 'auto'")
        if model is not None and not isinstance(model, str):
            raise HTTPException(status_code=422, detail="model must be a string")
        if engine is not None and (not isinstance(engine, str)
                                   or not engine.strip()):
            raise HTTPException(status_code=422,
                                detail="engine must be a non-empty string when set")
        if not isinstance(streams, int) or isinstance(streams, bool) \
                or not (1 <= streams <= 16):
            raise HTTPException(status_code=422,
                                detail="streams must be an int between 1 and 16")
        if power_host is not None and (not isinstance(power_host, str)
                                       or not power_host.strip()):
            raise HTTPException(status_code=422,
                                detail="power_host must be a non-empty string when set")
        if not isinstance(power_gpu_index, int) or isinstance(power_gpu_index, bool) \
                or power_gpu_index < 0:
            raise HTTPException(status_code=422,
                                detail="power_gpu_index must be an int >= 0")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise HTTPException(status_code=422,
                                detail="name must be a non-empty string when set")
        try:
            return ops.dispatch_bench(
                db_path=db_path, server_url=server_url,
                targets=targets, n_predict=n_predict, api_key=api_key,
                transport_name=transport, model=model,
                engine=engine.strip() if isinstance(engine, str) else None,
                streams=streams,
                power_host=power_host, power_gpu_index=power_gpu_index,
                name=name.strip() if isinstance(name, str) else None)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ops.DispatchFailure as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"bench failed: {exc}")

    @app.post("/api/ops/bench/jobs")
    def ops_bench_job(body: dict):
        """Async bench sweep: same body as POST /api/ops/bench.

        Returns {job_id} immediately after synchronous validation +
        preflight-free snapshot; the sweep (preflight, points, storage)
        runs in a background thread. Poll GET /api/ops/bench/jobs/{id}.
        Input errors are 422 here; everything inside the thread lands in
        the job record instead.
        """
        server_url = body.get("server_url")
        targets = body.get("targets")
        n_predict = body.get("n_predict", 1024)
        api_key = body.get("api_key")
        transport = body.get("transport", "native")
        model = body.get("model")
        engine = body.get("engine")
        streams = body.get("streams", 1)
        power_host = body.get("power_host")
        power_gpu_index = body.get("power_gpu_index", 0)
        name = body.get("name")
        if not isinstance(server_url, str) or not server_url.startswith("http"):
            raise HTTPException(status_code=422, detail="server_url must be an http(s) URL")
        if not isinstance(targets, list) or not all(
                isinstance(t, int) and t >= 1 for t in targets):
            raise HTTPException(status_code=422, detail="targets must be a list of ints >= 1")
        if not isinstance(n_predict, int) or n_predict < 1:
            raise HTTPException(status_code=422, detail="n_predict must be an int >= 1")
        if not isinstance(transport, str) or transport not in ("native", "openai", "auto"):
            raise HTTPException(status_code=422,
                                detail="transport must be 'native', 'openai' or 'auto'")
        if model is not None and not isinstance(model, str):
            raise HTTPException(status_code=422, detail="model must be a string")
        if engine is not None and (not isinstance(engine, str)
                                   or not engine.strip()):
            raise HTTPException(status_code=422,
                                detail="engine must be a non-empty string when set")
        if not isinstance(streams, int) or isinstance(streams, bool) \
                or not (1 <= streams <= 16):
            raise HTTPException(status_code=422,
                                detail="streams must be an int between 1 and 16")
        if power_host is not None and (not isinstance(power_host, str)
                                       or not power_host.strip()):
            raise HTTPException(status_code=422,
                                detail="power_host must be a non-empty string when set")
        if not isinstance(power_gpu_index, int) or isinstance(power_gpu_index, bool) \
                or power_gpu_index < 0:
            raise HTTPException(status_code=422,
                                detail="power_gpu_index must be an int >= 0")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise HTTPException(status_code=422,
                                detail="name must be a non-empty string when set")
        try:
            job_id = ops.run_bench_job(
                db_path=db_path, server_url=server_url, targets=targets,
                n_predict=n_predict, api_key=api_key,
                transport_name=transport, model=model,
                engine=engine.strip() if isinstance(engine, str) else None,
                streams=streams,
                power_host=power_host, power_gpu_index=power_gpu_index,
                name=name.strip() if isinstance(name, str) else None)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"job_id": job_id}

    @app.get("/api/ops/bench/jobs/{job_id}")
    def ops_bench_job_status(job_id: str):
        rec = ops.bench_job_status(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"bench job {job_id} not found")
        if rec["finished_at"] is None:
            rec["elapsed_s"] = round(time.monotonic() - rec["started_at"], 1)
        else:
            rec["elapsed_s"] = round(rec["finished_at"] - rec["started_at"], 1)
        rec.pop("started_at", None)
        rec.pop("finished_at", None)
        return rec

    @app.post("/api/ops/bench/jobs/{job_id}/abort")
    def ops_bench_job_abort(job_id: str):
        """Operator stop for a faulty/ runaway bench job (ADR 006).

        Cooperative: the sweep checks the flag between points and inside
        streaming loops; completed points land as a partial run exactly
        like any mid-sweep failure. 404 on unknown job; 200 otherwise —
        aborting a finished job is an honest no-op.
        """
        result = ops.abort_bench_job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"bench job {job_id} not found")
        return result

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
        name = body.get("name")
        reasoning_effort = body.get("reasoning_effort")
        reasoning_budget = body.get("reasoning_budget")
        if not isinstance(suite_id, int) or suite_id < 1:
            raise HTTPException(status_code=422, detail="suite_id must be a positive int")
        if not isinstance(target, str) or not target.startswith("http"):
            raise HTTPException(status_code=422, detail="target must be an http(s) URL")
        if not isinstance(model_fingerprint, str) or not model_fingerprint:
            raise HTTPException(status_code=422, detail="model_fingerprint required")
        if not isinstance(suite, dict):
            raise HTTPException(status_code=422, detail="suite must be an inline JSON object with 'items'")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise HTTPException(status_code=422,
                                detail="name must be a non-empty string when set")
        try:
            multiturn_mod.validate_reasoning(reasoning_effort, reasoning_budget)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            return ops.dispatch_eval(
                db_path=db_path, suite_id=suite_id, target=target,
                model_fingerprint=model_fingerprint, suite=suite,
                api_key=api_key,
                name=name.strip() if isinstance(name, str) else None,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ops.DispatchFailure as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:  # target unreachable/HTTP error/etc.
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=502, detail=f"eval replay failed: {exc}")

    @app.post("/api/ops/eval/jobs")
    def ops_eval_job(body: dict):
        """Async eval replay (v0.2): same body as POST /api/ops/eval plus
        max_tokens/temperature/retries/concurrency.

        Returns {job_id} immediately after synchronous validation; the
        replay (preflight, per-item chat calls, per-item commits) runs in
        a background thread. Poll GET /api/ops/eval/jobs/{id}; stop with
        POST .../abort. Input errors are 422 here; everything inside the
        thread lands in the job record.
        """
        suite_id = body.get("suite_id")
        target = body.get("target")
        model_fingerprint = body.get("model_fingerprint")
        api_key = body.get("api_key")
        suite = body.get("suite")
        name = body.get("name")
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.0)
        retries = body.get("retries", 0)
        concurrency = body.get("concurrency", 1)
        reasoning_effort = body.get("reasoning_effort")
        reasoning_budget = body.get("reasoning_budget")
        if not isinstance(suite_id, int) or suite_id < 1:
            raise HTTPException(status_code=422, detail="suite_id must be a positive int")
        if not isinstance(target, str) or not target.startswith("http"):
            raise HTTPException(status_code=422, detail="target must be an http(s) URL")
        if not isinstance(model_fingerprint, str) or not model_fingerprint:
            raise HTTPException(status_code=422, detail="model_fingerprint required")
        if not isinstance(suite, dict):
            raise HTTPException(status_code=422, detail="suite must be an inline JSON object with 'items'")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise HTTPException(status_code=422,
                                detail="name must be a non-empty string when set")
        for field, value in (("max_tokens", max_tokens), ("retries", retries),
                             ("concurrency", concurrency)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise HTTPException(status_code=422, detail=f"{field} must be an int")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise HTTPException(status_code=422, detail="temperature must be a number")
        try:
            job_id = ops.run_eval_job(
                db_path=db_path, suite_id=suite_id, target=target,
                model_fingerprint=model_fingerprint, suite=suite,
                api_key=api_key,
                name=name.strip() if isinstance(name, str) else None,
                max_tokens=max_tokens, temperature=temperature,
                retries=retries, concurrency=concurrency,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget)
        except ops.DispatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except (ValueError, evals_mod.ItemValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"job_id": job_id}

    @app.get("/api/ops/eval/jobs/{job_id}")
    def ops_eval_job_status(job_id: str):
        result = ops.eval_job_status(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"eval job {job_id} not found")
        return result

    @app.post("/api/ops/eval/jobs/{job_id}/abort")
    def ops_eval_job_abort(job_id: str):
        result = ops.abort_eval_job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"eval job {job_id} not found")
        return result

    # -------------------------------------------------- judge jobs (J4) ----
    @app.post("/api/ops/judge/jobs")
    def ops_judge_job(body: dict):
        """Async blind A/B judge run (v0.2, ADR 007 §3).

        Body: {suite_id, judge_model?, judge_url?, model_a?, model_b?,
        retries?, allow_contestant_judge?, judge_api_key?, eval_run_ids?,
        target_a?, target_b?, eval_run_b?, eval_max_tokens?,
        eval_temperature?, eval_concurrency?, eval_retries?,
        tiebreak_flips?}.
        target_a/target_b may each also carry optional per-model
        reasoning_effort ('minimal'|'low'|'medium'|'high'|'xhigh'|'max') and
        reasoning_budget (max thinking tokens) — applied to that side's
        replay only.

        Modes. (1) Judge stored results (model_a/model_b or
        eval_run_ids scope). (2) Combined kick-off (2026-10-15): give
        target_a AND target_b ({url, model, api_key?}) and the job first
        replays the suite's latest definition snapshot on both models,
        then judges the two fresh runs — the Judgements page as the one
        kick-off surface for evals + judging. (3) Mixed kick-off: give
        target_a plus eval_run_b (a stored eval run id on the same
        suite) — model A replays fresh, model B is NOT replayed and the
        stored run is judged as the baseline instead; mirrored (2026-10-23):
        eval_run_a plus target_b replays model B fresh against the stored
        model-A run. Every judgeable
        item is judged in BOTH presentation orders — position swap —
        and both raw judgments stored. judge_url falls back to the
        config layer ([eval] judge_url, then legacy [demo] judge_url).
        Returns {job_id}; poll GET /api/ops/judge/jobs/{id}; stop with
        POST .../abort.
        """
        suite_id = body.get("suite_id")
        judge_model = body.get("judge_model")
        judge_url = body.get("judge_url")
        model_a = body.get("model_a")
        model_b = body.get("model_b")
        retries = body.get("retries", 1)
        allow_contestant = body.get("allow_contestant_judge", False)
        judge_api_key = body.get("judge_api_key") or body.get("api_key")
        eval_run_ids = body.get("eval_run_ids")
        target_a = body.get("target_a")
        target_b = body.get("target_b")
        eval_run_b = body.get("eval_run_b")
        eval_run_a = body.get("eval_run_a")
        eval_max_tokens = body.get("eval_max_tokens", 1024)
        eval_temperature = body.get("eval_temperature", 0.0)
        eval_concurrency = body.get("eval_concurrency", 4)
        eval_retries = body.get("eval_retries", 0)
        tiebreak_flips = body.get("tiebreak_flips", True)
        if not isinstance(suite_id, int) or suite_id < 1:
            raise HTTPException(status_code=422, detail="suite_id must be a positive int")
        if eval_run_ids is not None:
            # Results-page multi-select: scope judging to the selected
            # replay runs (needs two runs on the same suite to pair).
            if (not isinstance(eval_run_ids, list)
                    or not all(isinstance(r, int) and not isinstance(r, bool)
                               and r >= 1 for r in eval_run_ids)
                    or len(eval_run_ids) < 2):
                raise HTTPException(
                    status_code=422,
                    detail="eval_run_ids must be a list of >= 2 positive ints")
        if eval_run_a is not None and eval_run_b is not None:
            raise HTTPException(
                status_code=422,
                detail="eval_run_a and eval_run_b are mutually exclusive "
                       "— to re-judge two stored runs, pass "
                       "eval_run_ids=[a, b] instead")
        if eval_run_a is not None:
            if (not isinstance(eval_run_a, int)
                    or isinstance(eval_run_a, bool) or eval_run_a < 1):
                raise HTTPException(
                    status_code=422,
                    detail="eval_run_a must be a positive int when set")
            if target_a is not None:
                raise HTTPException(
                    status_code=422,
                    detail="target_a and eval_run_a are mutually exclusive "
                           "— model A is either a fresh replay or a stored "
                           "run")
            if target_b is None:
                raise HTTPException(
                    status_code=422,
                    detail="eval_run_a requires target_b (model B must "
                           "replay fresh)")
        if eval_run_b is not None:
            if (not isinstance(eval_run_b, int)
                    or isinstance(eval_run_b, bool) or eval_run_b < 1):
                raise HTTPException(
                    status_code=422,
                    detail="eval_run_b must be a positive int when set")
            if target_b is not None:
                raise HTTPException(
                    status_code=422,
                    detail="target_b and eval_run_b are mutually exclusive "
                           "— model B is either a fresh replay or a stored "
                           "run")
            if target_a is None:
                raise HTTPException(
                    status_code=422,
                    detail="eval_run_b requires target_a (model A must "
                           "replay fresh)")
        if (target_a is None) != (target_b is None):
            # One target alone is legal ONLY in a mixed kick-off, paired
            # with the OTHER side's stored run (target_a + eval_run_b, or
            # target_b + eval_run_a); anything else needs both targets
            # together.
            mixed_ok = ((target_a is not None and eval_run_b is not None
                         and target_b is None)
                        or (target_b is not None and eval_run_a is not None
                            and target_a is None))
            if not mixed_ok:
                raise HTTPException(
                    status_code=422,
                    detail="target_a and target_b must be given together "
                           "— or pair one target with the other side's "
                           "stored run (target_a + eval_run_b, or "
                           "target_b + eval_run_a)")
        for side, tgt in (("a", target_a), ("b", target_b)):
            if tgt is None:
                continue
            if not isinstance(tgt, dict):
                raise HTTPException(status_code=422,
                                    detail=f"target_{side} must be an object")
            if not isinstance(tgt.get("url"), str) or not tgt["url"].startswith("http"):
                raise HTTPException(status_code=422,
                                    detail=f"target_{side}.url must be an http(s) URL")
            if not isinstance(tgt.get("model"), str) or not tgt["model"].strip():
                raise HTTPException(status_code=422,
                                    detail=f"target_{side}.model must be a non-empty string")
            key = tgt.get("api_key")
            if key is not None and not isinstance(key, str):
                raise HTTPException(status_code=422,
                                    detail=f"target_{side}.api_key must be a string when set")
        if eval_run_ids is not None and (target_a is not None
                                         or eval_run_b is not None
                                         or eval_run_a is not None):
            raise HTTPException(status_code=422,
                                detail="eval_run_ids is mutually exclusive with target_a/target_b/eval_run_b/eval_run_a")
        for field, value in (("eval_max_tokens", eval_max_tokens),
                             ("eval_concurrency", eval_concurrency),
                             ("eval_retries", eval_retries)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1 and field != "eval_retries":
                raise HTTPException(status_code=422, detail=f"{field} must be a positive int")
        if not isinstance(eval_retries, int) or isinstance(eval_retries, bool) or eval_retries < 0:
            raise HTTPException(status_code=422, detail="eval_retries must be a non-negative int")
        if not isinstance(eval_temperature, (int, float)) or isinstance(eval_temperature, bool):
            raise HTTPException(status_code=422, detail="eval_temperature must be a number")
        if judge_model is not None and (not isinstance(judge_model, str)
                                        or not judge_model.strip()):
            raise HTTPException(status_code=422,
                                detail="judge_model must be a non-empty string when set")
        if judge_url is not None and (not isinstance(judge_url, str)
                                      or not judge_url.startswith("http")):
            raise HTTPException(status_code=422, detail="judge_url must be an http(s) URL")
        if not isinstance(retries, int) or isinstance(retries, bool) or not (0 <= retries <= 3):
            raise HTTPException(status_code=422, detail="retries must be an int between 0 and 3")
        if (model_a is None) != (model_b is None):
            raise HTTPException(status_code=422,
                                detail="model_a and model_b must be given together")
        if model_a is not None and (not isinstance(model_a, str)
                                    or not isinstance(model_b, str)
                                    or not model_a.strip() or not model_b.strip()):
            raise HTTPException(status_code=422,
                                detail="model_a/model_b must be non-empty strings when set")
        if not isinstance(allow_contestant, bool):
            raise HTTPException(status_code=422,
                                detail="allow_contestant_judge must be a boolean")
        try:
            job_id = judging.run_judge_job(
                db_path, suite_id,
                judge_url=judge_url,
                judge_model=judge_model.strip() if isinstance(judge_model, str) else None,
                model_a=model_a, model_b=model_b, retries=retries,
                allow_contestant_judge=allow_contestant,
                judge_api_key=judge_api_key,
                eval_run_ids=eval_run_ids,
                target_a=target_a, target_b=target_b,
                eval_run_b=eval_run_b,
                eval_run_a=eval_run_a,
                eval_max_tokens=eval_max_tokens,
                eval_temperature=eval_temperature,
                eval_concurrency=eval_concurrency,
                eval_retries=eval_retries,
                tiebreak_flips=tiebreak_flips)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"job_id": job_id}

    @app.get("/api/ops/judge/jobs/{job_id}")
    def ops_judge_job_status(job_id: str):
        result = judging.judge_job_status(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"judge job {job_id} not found")
        return result

    @app.post("/api/ops/judge/jobs/{job_id}/abort")
    def ops_judge_job_abort(job_id: str):
        result = judging.abort_judge_job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"judge job {job_id} not found")
        return result

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
        if kind not in ("bench", "eval", "auto"):
            raise HTTPException(status_code=422,
                                detail="kind must be 'bench', 'eval' or 'auto'")
        if kind == "auto":
            # UI autodetect (2026-10-11): the Start page no longer picks a
            # transport, so the interactive check probes openai first (the
            # usual case) and falls back to native. On total failure the
            # openai result is returned — its check details are the ones
            # the operator can act on. Worst case is 2 probes (~32s) on a
            # dead target; the button is explicitly a "check" action.
            pf_eval = ops.preflight.preflight(target, kind="eval", model=model,
                                              api_key=api_key, timeout=8.0)
            if pf_eval.ok():
                return pf_eval
            pf_bench = ops.preflight.preflight(target, kind="bench", model=model,
                                               api_key=api_key, timeout=8.0)
            return pf_bench if pf_bench.ok() else pf_eval
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
    def require_visible_run(db: sqlite3.Connection, run_id: int,
                            label: str = "run id") -> None:
        """404 unless the run exists AND is not hidden.

        One shared gate for every compare surface (series, run-diff,
        baseline-deltas, the baseline= reference param): the append-only
        soft hide removes runs from list/compare views, and every
        rejection names the unhide call so the 404 is actionable.
        """
        if fetch_one(db, "SELECT 1 AS ok FROM runs WHERE id = ?", (run_id,)) is None:
            raise HTTPException(status_code=404, detail=f"{label} {run_id} not found")
        if fetch_one(db, f"SELECT 1 AS ok FROM runs WHERE id = ? AND NOT EXISTS ({_HIDDEN_RUN_SQL})", (run_id,)) is None:
            raise HTTPException(status_code=404,
                                detail=f"{label} {run_id} is hidden (unhide via PUT /api/runs/{run_id}/visibility)")

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
            require_visible_run(db, rid)
        if baseline is not None:
            require_visible_run(db, baseline, label="baseline run id")
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
        for rid in run_ids:
            require_visible_run(db, rid)
        base = baseline if baseline is not None else run
        require_visible_run(db, base)
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
            "SELECT id AS run_id, name, server_url, model_fingerprint, engine, "
            "context_size, status, created_at FROM runs ORDER BY id",
        )
        return _csv_response(
            rows,
            ["run_id", "name", "server_url", "model_fingerprint", "engine",
             "context_size", "status", "created_at"],
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
             "context_tokens", "prefill_tps", "decode_tps",
             "prefill_tps_combined", "decode_tps_combined", "ttft_ms",
             "wall_s", "output_tokens", "mtp_draft_n", "mtp_accepted",
             "power_watts", "power_host", "power_gpu_index", "streams",
             "created_at"],
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
                "SELECT er.id AS result_id, er.*, s.name AS suite_name, s.version AS suite_version "
                "FROM eval_results er JOIN eval_suites s ON s.id = er.suite_id "
                "WHERE er.suite_id = ? ORDER BY er.id",
                (suite_id,),
            )
        else:
            rows = fetch_all(
                db,
                "SELECT er.id AS result_id, er.*, s.name AS suite_name, s.version AS suite_version "
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
            "SELECT j.id AS judgment_id, j.*, ra.model_fingerprint AS fp_a, "
            "rb.model_fingerprint AS fp_b FROM judgments j "
            "JOIN eval_results ra ON ra.id = j.eval_result_a "
            "JOIN eval_results rb ON rb.id = j.eval_result_b "
            "ORDER BY j.id",
        )
        return _csv_response(
            rows,
            ["judgment_id", "eval_result_a", "eval_result_b", "fp_a", "fp_b",
             "judge_model", "judge_template_version", "winner",
             "verdict_detail", "confidence",
             "rationale", "created_at", "role", "presentation",
             "raw_reply"],
            "arcturos-judgments.csv",
        )

    @app.get("/api/compare/run-diff/{run_a}/{run_b}")
    def run_diff(run_a: int, run_b: int, db=Depends(get_db)):
        """Per-context delta table between two runs + engine metadata diff."""
        for rid in (run_a, run_b):
            require_visible_run(db, rid)
        return compare.build_run_diff_payload(db, run_a, run_b)

    return app


app = create_app()
