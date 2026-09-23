"""API + storage tests for the Arcturos dashboard skeleton.

Covers: schema creation, inserts, full API round trips per endpoint group
(runs/benchmarks, eval suites/results, judgments), append-only enforcement
(no update/delete API paths + storage-layer triggers), and Pydantic input
validation.
"""

import sqlite3

import pytest

EXPECTED_COLUMNS = {
    "runs": {"id", "server_url", "model_fingerprint", "engine", "context_size", "status", "name", "created_at"},
    "benchmarks": {
        "run_id", "context_tokens", "prefill_tps", "decode_tps", "ttft_ms",
        "wall_s", "output_tokens", "mtp_draft_n", "mtp_accepted", "power_watts",
        "power_host", "power_gpu_index", "streams", "decode_tps_combined",
        "prefill_tps_combined", "created_at",
    },
    "eval_suites": {"id", "name", "version"},
    "eval_results": {
        "id", "suite_id", "model_fingerprint", "item_id", "output", "prompt_tokens",
        "completion_tokens", "latency_ms", "eval_run_id", "created_at",
    },
    "eval_runs": {"id", "suite_id", "model_fingerprint", "name", "created_at"},
    "judgments": {
        "id", "eval_result_a", "eval_result_b", "judge_model", "judge_template_version",
        "winner", "confidence", "rationale", "created_at",
    },
    "suite_definitions": {"id", "suite_id", "payload", "created_at"},
    "run_visibility": {"id", "run_id", "hidden", "reason", "created_at"},
}

RUN = {
    "server_url": "http://10.10.10.222:8000/v1",
    "model_fingerprint": "GLM-5.3-Flash-UD-IQ2_XXS",
    "engine": "llama.cpp",
    "context_size": 262144,
}


# ------------------------------------------------------------ schema -------


def test_schema_creation(client):
    db = sqlite3.connect(client.app.state.db_path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(EXPECTED_COLUMNS) <= tables
    for table, cols in EXPECTED_COLUMNS.items():
        actual = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        assert actual == cols, f"{table}: expected {cols}, got {actual}"
    db.close()


def test_append_only_triggers_installed(client):
    db = sqlite3.connect(client.app.state.db_path)
    triggers = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    for table in EXPECTED_COLUMNS:
        assert f"trg_{table}_no_update" in triggers
        assert f"trg_{table}_no_delete" in triggers
    db.close()


# ------------------------------------------------------------ runs ---------


def test_runs_benchmarks_roundtrip(client):
    created = client.post("/api/runs", json=RUN)
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["id"] >= 1
    assert run["server_url"] == RUN["server_url"]
    assert run["engine"] == "llama.cpp"

    bench = client.post(
        f"/api/runs/{run['id']}/benchmarks",
        json={
            "context_tokens": 4096,
            "prefill_tps": 174.2,
            "decode_tps": 20.1,
            "ttft_ms": 512.0,
            "wall_s": 12.4,
            "output_tokens": 250,
            "mtp_draft_n": 3,
            "mtp_accepted": 2,
            "power_watts": 250.0,
            "power_host": "alexei@10.10.10.122",
            "power_gpu_index": 0,
        },
    )
    assert bench.status_code == 201, bench.text
    row = bench.json()
    assert row["run_id"] == run["id"]
    assert row["decode_tps"] == 20.1
    assert row["mtp_accepted"] == 2

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert any(r["id"] == run["id"] for r in runs.json())

    assert client.get(f"/api/runs/{run['id']}").json()["id"] == run["id"]

    benchs = client.get(f"/api/runs/{run['id']}/benchmarks")
    assert benchs.status_code == 200
    assert len(benchs.json()) == 1

    # Same context point re-run appends a new row, never overwrites (PRD J1).
    again = client.post(
        f"/api/runs/{run['id']}/benchmarks",
        json={"context_tokens": 4096, "decode_tps": 19.8},
    )
    assert again.status_code == 201
    assert len(client.get(f"/api/runs/{run['id']}/benchmarks").json()) == 2


def test_benchmark_requires_existing_run(client):
    r = client.post("/api/runs/999/benchmarks", json={"context_tokens": 4096})
    assert r.status_code == 404


def test_benchmark_mtp_validation(client):
    run = client.post("/api/runs", json=RUN).json()
    r = client.post(
        f"/api/runs/{run['id']}/benchmarks",
        json={"context_tokens": 4096, "mtp_draft_n": 3, "mtp_accepted": 5},
    )
    assert r.status_code == 422


# ------------------------------------------------------ eval suites ---------


def test_eval_suites_results_roundtrip(client):
    suite = client.post("/api/eval-suites", json={"name": "reasoning", "version": "1.0.0"})
    assert suite.status_code == 201, suite.text
    suite_id = suite.json()["id"]

    res = client.post(
        f"/api/eval-suites/{suite_id}/results",
        json={
            "model_fingerprint": "GLM-5.3-Flash-UD-IQ2_XXS",
            "item_id": "math-001",
            "output": "42",
            "prompt_tokens": 120,
            "completion_tokens": 3,
            "latency_ms": 240.0,
        },
    )
    assert res.status_code == 201, res.text
    assert res.json()["suite_id"] == suite_id

    suites = client.get("/api/eval-suites")
    assert suites.status_code == 200
    assert len(suites.json()) == 1

    assert client.get(f"/api/eval-suites/{suite_id}").json()["name"] == "reasoning"

    results = client.get(f"/api/eval-suites/{suite_id}/results")
    assert results.status_code == 200
    assert len(results.json()) == 1

    assert len(client.get("/api/eval-results").json()) == 1


def test_eval_result_requires_existing_suite(client):
    r = client.post(
        "/api/eval-suites/999/results",
        json={
            "model_fingerprint": "x", "item_id": "i", "output": "o",
            "prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 1.0,
        },
    )
    assert r.status_code == 404


# ------------------------------------------------------- run visibility -----


def test_run_visibility_hide_unhide_roundtrip(client):
    """Soft hide: newest flag wins; hidden runs drop out of list views but
    stay directly addressable; unhide appends a visible flag."""
    run = client.post("/api/runs", json=RUN).json()
    rid = run["id"]

    # default: visible, no flag rows
    vis = client.get(f"/api/runs/{rid}/visibility")
    assert vis.status_code == 200
    assert vis.json() == {"run_id": rid, "hidden": False, "reason": None,
                          "created_at": None}
    assert len(client.get("/api/runs").json()) == 1

    # hide -> drops from the list, still directly fetchable
    put = client.put(f"/api/runs/{rid}/visibility",
                     json={"hidden": True, "reason": "accidental duplicate"})
    assert put.status_code == 204
    assert client.get("/api/runs").json() == []
    assert client.get(f"/api/runs/{rid}").json()["id"] == rid
    assert client.get(f"/api/runs/{rid}/visibility").json()["hidden"] is True
    assert client.get(f"/api/runs/{rid}/visibility").json()["reason"] == "accidental duplicate"

    # include_hidden=true surfaces it again (operator escape hatch)
    assert len(client.get("/api/runs?include_hidden=true").json()) == 1

    # unhide -> back in the list
    assert client.put(f"/api/runs/{rid}/visibility",
                      json={"hidden": False}).status_code == 204
    assert len(client.get("/api/runs").json()) == 1
    assert client.get(f"/api/runs/{rid}/visibility").json()["hidden"] is False


def test_run_visibility_validation_and_404(client):
    run = client.post("/api/runs", json=RUN).json()
    assert client.put(f"/api/runs/{run['id']}/visibility",
                      json={"hidden": "yes"}).status_code == 422
    assert client.put(f"/api/runs/{run['id']}/visibility",
                      json={}).status_code == 422
    assert client.put("/api/runs/999/visibility",
                      json={"hidden": True}).status_code == 404
    assert client.get("/api/runs/999/visibility").status_code == 404


def test_run_visibility_survives_append_only_triggers(client):
    """The flag table is append-only at the storage layer too."""
    import sqlite3 as _sq
    run = client.post("/api/runs", json=RUN).json()
    client.put(f"/api/runs/{run['id']}/visibility", json={"hidden": True})
    db = _sq.connect(client.app.state.db_path)
    # The triggers RAISE(ABORT), which surfaces as IntegrityError.
    with pytest.raises(_sq.DatabaseError):
        db.execute("UPDATE run_visibility SET hidden = 0").fetchall()
    with pytest.raises(_sq.DatabaseError):
        db.execute("DELETE FROM run_visibility").fetchall()
    db.close()


def test_hidden_run_excluded_from_compare(client):
    """Compare rejects a hidden run with an actionable 404; visible runs pass."""
    run = client.post("/api/runs", json=RUN).json()
    client.put(f"/api/runs/{run['id']}/visibility", json={"hidden": True})
    r = client.get(f"/api/compare/benchmarks?runs={run['id']}&metrics=decode_tps")
    assert r.status_code == 404
    assert "hidden" in r.json()["detail"]
    client.put(f"/api/runs/{run['id']}/visibility", json={"hidden": False})
    r2 = client.get(f"/api/compare/benchmarks?runs={run['id']}&metrics=decode_tps")
    assert r2.status_code == 200


# ---------------------------------------------- suite definition snapshots --


def test_suite_definition_snapshot_roundtrip(client):
    """PUT snapshots the definition; GET returns the latest snapshot."""
    suite = client.post("/api/eval-suites", json={"name": "s", "version": "1"}).json()
    sid = suite["id"]

    missing = client.get(f"/api/eval-suites/{sid}/definition")
    assert missing.status_code == 404

    definition = {
        "suite": "s", "version": "1",
        "items": [
            {"id": "sr-001", "category": "logic", "prompt": "p1"},
            {"id": "sr-002", "category": "counting", "prompt": "p2"},
        ],
    }
    put = client.put(f"/api/eval-suites/{sid}/definition", json=definition)
    assert put.status_code == 204, put.text

    got = client.get(f"/api/eval-suites/{sid}/definition")
    assert got.status_code == 200
    assert got.json()["items"][0]["category"] == "logic"

    # Re-seeding appends a new snapshot; the newest one wins on read.
    v2 = {**definition, "items": [{"id": "sr-001", "category": "revised", "prompt": "p1"}]}
    assert client.put(f"/api/eval-suites/{sid}/definition", json=v2).status_code == 204
    assert client.get(f"/api/eval-suites/{sid}/definition").json()["items"][0]["category"] == "revised"


def test_suite_definition_requires_existing_suite_and_items(client):
    assert client.put("/api/eval-suites/999/definition",
                      json={"items": [{"id": "x"}]}).status_code == 404
    suite = client.post("/api/eval-suites", json={"name": "s", "version": "1"}).json()
    assert client.put(f"/api/eval-suites/{suite['id']}/definition",
                      json={"no_items": True}).status_code == 422


# --------------------------------------------------------- judgments -------


def _seed_results(client, n=2):
    suite = client.post("/api/eval-suites", json={"name": "s", "version": "1"}).json()
    ids = []
    for i in range(n):
        r = client.post(
            f"/api/eval-suites/{suite['id']}/results",
            json={
                "model_fingerprint": f"model-{i}",
                "item_id": "q1",
                "output": f"answer {i}",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "latency_ms": 50.0,
            },
        )
        assert r.status_code == 201
        ids.append(r.json()["id"])
    return ids


def test_judgments_roundtrip(client):
    a, b = _seed_results(client)
    j = client.post(
        "/api/judgments",
        json={
            "eval_result_a": a,
            "eval_result_b": b,
            "judge_model": "qwen-3.6-27b",
            "judge_template_version": "blind-v1",
            "winner": "a",
            "confidence": 0.8,
            "rationale": "concise and correct",
        },
    )
    assert j.status_code == 201, j.text
    assert j.json()["winner"] == "a"

    got = client.get("/api/judgments")
    assert got.status_code == 200
    assert len(got.json()) == 1
    assert got.json()[0]["judge_model"] == "qwen-3.6-27b"


def test_judgment_requires_existing_results(client):
    a, _ = _seed_results(client)
    r = client.post(
        "/api/judgments",
        json={
            "eval_result_a": a,
            "eval_result_b": 9999,
            "judge_model": "qwen-3.6-27b",
            "judge_template_version": "blind-v1",
            "winner": "b",
        },
    )
    assert r.status_code == 404


def test_judgment_self_comparison_rejected(client):
    a, _ = _seed_results(client)
    r = client.post(
        "/api/judgments",
        json={
            "eval_result_a": a,
            "eval_result_b": a,
            "judge_model": "qwen-3.6-27b",
            "judge_template_version": "blind-v1",
            "winner": "tie",
        },
    )
    assert r.status_code == 422


# -------------------------------------------------------- append-only --------


def test_api_has_no_update_or_delete_paths(client):
    # PUT is allowed on exactly one append-only snapshot surface: the suite
    # definition (each PUT appends a new snapshot row; nothing stored is
    # ever mutated). PATCH and DELETE are allowed on exactly two non-data
    # surfaces (§4.6): the model alias (a label) and baseline unpin (a
    # reference pointer). None of these touch stored measurement rows.
    methods = {m for r in client.app.routes for m in getattr(r, "methods", set())}
    assert "PUT" in methods  # snapshot/flag surfaces only — see below
    put_paths = {
        r.path for r in client.app.routes
        if getattr(r, "methods", set()) & {"PUT"}}
    assert put_paths == {"/api/eval-suites/{suite_id}/definition",
                         "/api/runs/{run_id}/visibility"}
    editable = {
        r.path for r in client.app.routes
        if getattr(r, "methods", set()) & {"PATCH", "DELETE"}}
    assert editable == {"/api/models/{fingerprint:path}",
                        "/api/baselines/{baseline_id}",
                        # 2026-09-22: run names are LABELS (alias precedent)
                        "/api/runs/{run_id}/name"}


def test_db_rejects_update_and_delete(client):
    # Seed at least one row in every table: SQLite triggers are row-level and
    # only fire when the statement actually matches rows.
    run = client.post("/api/runs", json=RUN).json()
    client.post(f"/api/runs/{run['id']}/benchmarks", json={"context_tokens": 4096})
    suite = client.post("/api/eval-suites", json={"name": "s", "version": "1"}).json()
    ids = []
    for i in range(2):
        r = client.post(
            f"/api/eval-suites/{suite['id']}/results",
            json={
                "model_fingerprint": f"model-{i}", "item_id": "q1", "output": "x",
                "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 50.0,
            },
        )
        assert r.status_code == 201
        ids.append(r.json()["id"])
    client.post("/api/judgments", json={
        "eval_result_a": ids[0], "eval_result_b": ids[1], "judge_model": "j",
        "judge_template_version": "v1", "winner": "a",
    })

    db = sqlite3.connect(client.app.state.db_path)
    cases = [
        ("UPDATE runs SET engine = 'hacked' WHERE id = ?", (run["id"],)),
        ("DELETE FROM runs WHERE id = ?", (run["id"],)),
        ("UPDATE benchmarks SET decode_tps = 999", ()),
        ("DELETE FROM benchmarks", ()),
        ("UPDATE eval_suites SET name = 'hacked'", ()),
        ("DELETE FROM eval_suites", ()),
        ("UPDATE eval_results SET output = 'hacked'", ()),
        ("DELETE FROM eval_results", ()),
        ("UPDATE judgments SET winner = 'b'", ()),
        ("DELETE FROM judgments", ()),
    ]
    for sql, params in cases:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute(sql, params)
    db.close()

    # Everything survived every attempted mutation.
    assert len(client.get("/api/runs").json()) == 1
    assert len(client.get("/api/eval-suites").json()) == 1
    assert len(client.get("/api/judgments").json()) == 1


# -------------------------------------------------------- validation --------


def test_input_validation(client):
    assert client.post("/api/runs", json={}).status_code == 422
    assert client.post("/api/runs", json={**RUN, "server_url": "not-a-url"}).status_code == 422
    assert client.post("/api/runs", json={**RUN, "context_size": -5}).status_code == 422
    assert client.post("/api/eval-suites", json={"name": "", "version": "1"}).status_code == 422
    assert client.post("/api/judgments", json={
        "eval_result_a": 1, "eval_result_b": 2, "judge_model": "j",
        "judge_template_version": "v1", "winner": "invalid",
    }).status_code == 422


# -------------------------------------------------------- frontend ----------


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Arcturos" in r.text


def test_run_detail_page_has_combined_throughput_columns(client):
    """Run detail renders Σ combined-throughput columns (2026-09-22) —
    hidden behind streams > 1 for single-stream points (honest nulls)."""
    html = client.get("/runs/1").text
    assert "Σ prefill t/s" in html
    assert "Σ decode t/s" in html
    assert "prefill_tps_combined" in html
    assert "decode_tps_combined" in html
    assert "multi-stream points only" in html


# ----------------------------------------------------- partial runs --------


def test_runs_status_filter_partial_and_complete(client):
    """2026-09-22 partial-run support: /api/runs?status= filters by run
    outcome; omitted returns everything (hidden still filtered)."""
    import sqlite3
    db = sqlite3.connect(client.app.state.db_path)
    db.execute(
        "INSERT INTO runs (server_url, model_fingerprint, engine, "
        "context_size, status, created_at) VALUES (?, ?, ?, ?, 'partial', ?)",
        ("http://fake:8000", "/models/partial.gguf", "llama.cpp", 4096,
         "2026-09-22T00:00:00Z"))
    db.commit()
    db.close()

    all_runs = client.get("/api/runs").json()
    assert any(r["status"] == "partial" for r in all_runs)

    partial = client.get("/api/runs?status=partial").json()
    assert len(partial) == 1
    assert partial[0]["status"] == "partial"

    complete = client.get("/api/runs?status=complete").json()
    assert all(r["status"] == "complete" for r in complete)
    assert len(complete) == len(all_runs) - 1


def test_runs_status_filter_validation(client):
    res = client.get("/api/runs?status=bogus")
    assert res.status_code == 422
    assert "complete" in res.json()["detail"]


def test_runs_list_page_shows_partial_badge(client):
    """/runs marks partial runs with a visible badge — never hidden."""
    html = client.get("/runs").text
    assert "⚠ partial" in html
    assert "completed points preserved" in html


def test_run_detail_page_shows_partial_marker(client):
    html = client.get("/runs/1").text
    assert "partial — sweep failed mid-way" in html


def test_create_page_links_partial_run_on_failure(client):
    """The live job panel links to the partial run when a sweep fails
    mid-way instead of dead-ending."""
    html = client.get("/create").text
    assert "(partial)" in html
    assert "completed point(s) kept" in html


def test_rename_run_sets_and_clears_name(client):
    """PATCH /api/runs/{id}/name sets the label; null clears it back to
    the default rendering; measurement columns stay untouched."""
    run_id = client.post("/api/runs", json=RUN).json()["id"]
    r = client.patch(f"/api/runs/{run_id}/name", json={"name": "evening sweep"})
    assert r.status_code == 200
    assert r.json()["name"] == "evening sweep"
    # clear
    r = client.patch(f"/api/runs/{run_id}/name", json={"name": None})
    assert r.status_code == 200
    assert r.json()["name"] is None
    # validation: empty string rejected
    r = client.patch(f"/api/runs/{run_id}/name", json={"name": "  "})
    assert r.status_code == 422
    # unknown run
    r = client.patch("/api/runs/999999/name", json={"name": "x"})
    assert r.status_code == 404


def test_rename_run_cannot_touch_measurements(client):
    """The trigger whitelists ONLY the name column — any other UPDATE
    aborts (append-only integrity, ADR 003)."""
    import sqlite3
    run_id = client.post("/api/runs", json=RUN).json()["id"]
    conn = sqlite3.connect(client.app.state.db_path)
    try:
        conn.execute("UPDATE runs SET context_size = 1 WHERE id = ?", (run_id,))
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    finally:
        conn.close()
    assert raised


def test_runs_list_page_shows_name_column(client):
    """/runs second column is the run name (2026-09-22 naming feature);
    unset names render the honest `—` placeholder."""
    html = client.get("/runs").text
    assert "<th>Name</th>" in html
    assert "r.name || '—'" in html          # honest fallback, never blank


def test_run_detail_page_shows_name_row(client):
    """Run-detail meta shows Name first; rename control present."""
    run_id = client.post("/api/runs", json=RUN).json()["id"]
    html = client.get(f"/runs/{run_id}").text
    assert "['Name', run.name || '—']" in html


def test_run_detail_page_rename_control(client):
    """Run-detail page carries the ✎ Rename control wired to the rename
    API: prompt pre-filled with the current name, empty submit clears
    back to the default rendering (null), cancel is a no-op."""
    run_id = client.post("/api/runs", json=RUN).json()["id"]
    html = client.get(f"/runs/{run_id}").text
    assert "✎ Rename" in html
    assert "'/api/runs/' + run.id + '/name'" in html
    assert "method: 'PATCH'" in html
    assert "name: next.trim() || null" in html      # empty -> clear (null)
    assert "next === null" in html                   # cancel is a no-op
    assert "Run name (empty = clear back to default)" in html
