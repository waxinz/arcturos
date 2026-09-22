# Arcturos QA Test Plan v0.1

Owner: Mantis (b) · Status: active · Scope: dashboard + bench + evals (J1–J5)

## 1. Strategy

Three layers, all runnable from the repo root:

| Layer | Tooling | Target |
|---|---|---|
| Unit / API | pytest + FastAPI TestClient | every endpoint, validation boundary, append-only behavior |
| Storage | pytest against temp SQLite | triggers, FK constraints, schema shape |
| Live smoke | curl / real requests against taupo :24816 | deployed service health, real benchmark round trip |

Rules:
- Every acceptance criterion in PRD.md §J1–J5 maps to at least one test here.
- Append-only is tested at THREE levels: API (no update/delete routes), storage
  (triggers reject), and behavioral (re-run creates a new row, never overwrites).
- A bug that escapes this plan gets a regression test in the same commit as its fix.

## 2. Coverage map

### J1 — Benchmarks
- [x] `tests/test_api.py::test_create_run` — run creation, 201
- [x] `tests/test_api.py::test_add_benchmark*` — metrics per context point; 422 on invalid (e.g. mtp_accepted > mtp_draft_n)
- [x] `tests/test_api.py::test_append_not_overwrite` — re-POST same point → 2 rows
- [ ] `qa/suites/test_j1_integration.py` — bench module → storage round trip (added with bench core)
- [ ] Real smoke: bench ruapehu at 512 tokens, verify row appears via API

### J2 — Comparison (future endpoints/views)
- [ ] ≥2 models × ≥1 metric × N points in one query
- [ ] run-vs-run delta computation
- [ ] chart data endpoint returns ordered, complete series (missing points = explicit null, not dropped)

### J3 — Evals
- [x] suite creation, result storage, 404 on missing suite
- [ ] fixture suites in `qa/suites/` load via API (this commit)
- [ ] multi-turn item replay produces one result per turn
- [ ] suite version tag recorded on every result

### J4 — Blind judging
- [x] judgment creation with winner enum + confidence bounds
- [x] same-result A/B rejected (self-judging guard)
- [ ] order-randomization audit: stored judgment payload must not leak which model is A/B
- [ ] re-judging same pair with different template version appends a second row

### J5 — Reports
- [ ] win-rate aggregation: counts + percentages per suite × pair
- [ ] category breakdown
- [ ] CSV/JSON export matches stored rows exactly (no silent rounding)

### J7 — CRUD + dispatch (added 2026-09-17)
- [x] `tests/test_ops_dispatch.py` — bench dispatch happy path + storage round trip via export; validation (empty/zero targets); eval dispatch single-turn + multi-turn; unknown suite rejected BEFORE hitting the target; empty suite; missing prompt; transactional rollback (failed batch stores nothing)
- [x] `tests/test_ops_api.py` — `/api/ops/bench` + `/api/ops/eval` validation (422 shapes), full bench round trip through the API (run + benchmark rows visible in /api/runs), eval round trip (results visible in suite results API), suite-not-found → 422 with actionable detail, `/create` view served as HTML
- [x] API key auth (J6): `tests/test_ops_dispatch.py` authed-mock (llama.cpp `--api-key` behavior) — keyless request → preflight 401 → DispatchError; with key → sweep succeeds end-to-end. Preflight module + endpoint accept `api_key` for auth probes.
- [x] `tests/test_api.py` — power provenance: `power_watts` without `power_host` → 422 (ADR 002 enforced at the boundary); watts + host accepted and stored
- [x] `tests/test_bench_stream_power.py` + `tests/test_ops_dispatch.py` — power sampler mean/None semantics preserved
- [ ] Live smoke: dispatch a 1-point bench from `/create` against ruapehu during a quiet window; verify row lands in /compare
- [ ] Live smoke: eval replay from `/create` against qwen rotation; verify rows land in /evals

### Async bench jobs + progress indicator (added 2026-09-21)
- [x] `tests/test_ops_dispatch.py` — openai transport dispatch: happy path stores points with no power fields (ADR 002), preflight failure surfaces actionable DispatchError, missing model → DispatchError, invalid transport name → 422; all network mocked
- [x] Job API live-validated on taupo: job poll mid-run returns running + per-point metrics + self-correcting ETA; done state carries run_id; unknown job id → 404; openai job without model → 422 before any thread starts
- [x] Job-status 500 regression (missing `time` import crashed running-job polls; fixed a7cba82) — poller retries transient failures client-side
- [ ] Load: two concurrent jobs against different targets do not interleave stored rows

### Run visibility — soft hide (added 2026-09-21)
- [x] `tests/test_api.py::test_run_visibility_hide_unhide_roundtrip` — hide drops the run from /api/runs but direct fetch still works; unhide restores; `include_hidden=true` escape hatch; newest-flag-wins with reason preserved
- [x] `tests/test_api.py::test_run_visibility_validation_and_404` — non-bool hidden → 422, missing key → 422, unknown run → 404 (both endpoints)
- [x] `tests/test_api.py::test_run_visibility_survives_append_only_triggers` — UPDATE/DELETE on run_visibility rejected at the storage layer
- [x] `tests/test_api.py::test_hidden_run_excluded_from_compare` — compare rejects a hidden run with actionable 404 naming the unhide call; unhidden run compares again
- [x] Live-validated on taupo: hide run 9 → gone from /api/runs (reason stored), unhide → back; 144 tests green

### Cross-cutting
- [x] /health returns ok; index served
- [ ] concurrent POSTs don't race (commit-before-respond fix regression suite)
- [ ] DB survives restart (WAL checkpoint recovery)

## Review round — regressions (2026-09-22)

| # | Test | File | Status |
|---|------|------|--------|
| R1 | Export CSVs carry non-empty `result_id` / `judgment_id` on every row (writers previously asked for aliased columns the SELECTs never provided) | `tests/test_export.py::test_export_eval_results_ids_populated`, `::test_export_judgments_ids_populated` | ✅ |
| R2 | Bench-job status snapshot is isolated: mutating a fetched snapshot's `points` cannot corrupt the live job record (worker appends concurrently) | `tests/test_ops_dispatch.py::test_bench_job_status_snapshot_is_isolated` | ✅ |
| R3 | Hidden run rejected (404, actionable detail) by `/api/compare/baseline-deltas` — parity with the series + run-diff gates | `tests/test_ops_api.py::test_hidden_run_rejected_by_baseline_deltas` | ✅ |
| R4 | Hidden run rejected (404) as the `baseline=` reference on `/api/compare/benchmarks` | `tests/test_ops_api.py::test_hidden_run_rejected_as_baseline_reference` | ✅ |
| R5 | Registry provenance: `first_seen`/`last_seen` derive from the triggering row's own timestamp (single clock read) — deterministic, no ms-boundary flake | `tests/test_registry.py::test_second_run_updates_last_seen_not_first` (now stable across consecutive full-suite runs) | ✅ |
| R6 | `run_visibility` covered by the schema assertion map (`EXPECTED_COLUMNS`) | `tests/test_api.py::test_schema_creation` | ✅ |

## 2b. Review round 2 — run-17 decode fix

| # | Test | File | Status |
|---|------|------|--------|
| D1 | Server `completion_tokens_per_sec` wins over client chunk rate | `tests/test_bench_openai.py::test_happy_path_extracts_ttft_and_usage` | done |
| D2 | `completion_time` fallback (30 tok / 0.75 s = 40 t/s) when per-sec absent | `tests/test_bench_openai.py::test_completion_time_fallback_without_per_sec` | done |
| D3 | Reasoning-only stream (zero content chunks) still yields server decode — run-17 point-2 class (missing 128k value) | `tests/test_bench_openai.py::test_reasoning_only_stream_gets_server_decode` | done |
| D4 | Burst-flushed span (<0.25 s) rejected: no client rate, no artifact — run-17 point-1 class (15054 t/s) | `tests/test_bench_openai.py::test_burst_span_rejected_without_server_timing` | done |
| D5 | Real client span (>0.25 s) with no server timing still computes chunk rate | `tests/test_bench_openai.py::test_client_rate_used_when_span_real_and_no_server_timing` | done |
| D6 | Dispatch stores the resolved decode rate | `tests/test_ops_dispatch.py::test_dispatch_bench_openai_happy_stores_points_no_power` | done |

## 2c. Parallel streams (2026-09-22)

| # | Test | File | Status |
|---|------|------|--------|
| S1 | streams=2 dispatch stores aggregated point with streams=2 (mean rates, deterministic mock) | `tests/test_ops_dispatch.py::test_dispatch_bench_streams_two_stores_stream_count` | done |
| S2 | streams bounds: 0 and 17 rejected with DispatchError; bool rejected | `tests/test_ops_dispatch.py::test_dispatch_bench_streams_validation` | done |
| S3 | default streams=1 stored when the field is omitted | `tests/test_ops_dispatch.py::test_dispatch_bench_default_streams_is_one` | done |
| S4 | API 422 on streams out of bounds (sync endpoint) | `tests/test_ops_api.py::test_ops_bench_streams_out_of_bounds_is_422` | done |
| S5 | benchmarks.streams column present (schema assertion incl. migration) | `tests/test_api.py` EXPECTED_COLUMNS | done |
| S6 | compare allowlist exposes streams as a neutral metric | `tests/test_compare.py::test_compare_metrics_default_to_all_when_omitted` | done |

## 3. Fixture suites (`qa/suites/`)

Seeded demo data + reference eval suites, format documented in each file's
header. Used by: integration tests, dashboard demo mode, and as the first real
eval payloads for J3/J4 implementation.

## 4. Exit criteria per milestone

A milestone is "done" when: all mapped tests pass on taupo, the live smoke
layer exercises the feature against :24816, and AGENTS.md §6 records it.
