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
- [x] `tests/test_api.py` — power provenance: `power_watts` without `power_host` → 422 (ADR 002 enforced at the boundary); watts + host accepted and stored
- [x] `tests/test_bench_stream_power.py` + `tests/test_ops_dispatch.py` — power sampler mean/None semantics preserved
- [ ] Live smoke: dispatch a 1-point bench from `/create` against ruapehu during a quiet window; verify row lands in /compare
- [ ] Live smoke: eval replay from `/create` against qwen rotation; verify rows land in /evals

### Cross-cutting
- [x] /health returns ok; index served
- [ ] concurrent POSTs don't race (commit-before-respond fix regression suite)
- [ ] DB survives restart (WAL checkpoint recovery)

## 3. Fixture suites (`qa/suites/`)

Seeded demo data + reference eval suites, format documented in each file's
header. Used by: integration tests, dashboard demo mode, and as the first real
eval payloads for J3/J4 implementation.

## 4. Exit criteria per milestone

A milestone is "done" when: all mapped tests pass on taupo, the live smoke
layer exercises the feature against :24816, and AGENTS.md §6 records it.
