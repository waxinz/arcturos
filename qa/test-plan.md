# Arcturos QA Test Plan v0.1

Owner: Mantis (b) · Status: active · Scope: dashboard + bench + evals (J1–J5)

## 1. Strategy

Three layers, all runnable from the repo root:

| Layer | Tooling | Target |
|---|---|---|
| Unit / API | pytest + FastAPI TestClient | every endpoint, validation boundary, append-only behavior |
| Storage | pytest against temp SQLite | triggers, FK constraints, schema shape |
| Live smoke | curl / real requests against the deploy host :24816 | deployed service health, real benchmark round trip |

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
- [ ] Real smoke: bench demo host A at 512 tokens, verify row appears via API

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

### J4 — Both-bad judging (reference-guided, added 2026-10-23)
- [x] `tests/test_judging.py::test_extract_verdict_both_bad_variants` — VERDICT: both_bad / both-bad / "both bad" / case-insensitive / bare-token fallback; mid-sentence stray mention stays None (truncation guard)
- [x] `tests/test_judging.py::test_render_reference_kinds` — reference rendered for all five check kinds; missing value / unknown kind / non-dict -> None (plain v3 template, never a fabricated reference)
- [x] `tests/test_judging.py::test_judge_once_uses_ref_template_only_with_reference` — reference-guided prompt (blind-pair-v4-ref) only when a reference renders; template_version recorded per judgment
- [x] `tests/test_judging.py::test_judge_item_stores_both_bad_as_qualified_tie` / `..._plain_tie_keeps_no_detail` — both-bad stores winner='tie' + verdict_detail='both_bad' (CHECK untouched, append-only schema); plain tie keeps NULL detail
- [x] `tests/test_judging.py::test_dispatch_judge_persists_verdict_detail` — verdict_detail persisted through the dispatch INSERT for both presentation orders
- [x] `tests/test_judging.py::test_migration_adds_verdict_detail_to_legacy_db` — legacy DB migrates on init_db (idempotent ALTER); legacy tie rows keep NULL detail; new-style insert works
- [x] `tests/test_reports.py::test_both_bad_counts_as_own_bucket_and_half_draw` — both_bad bucket in overall/pair counts, both_bad_rate headline, 0.5-draw score math; plain ties unaffected

### J4 — Audit round (2026-10-23)
- [x] `tests/test_eval_jobs.py::test_eval_job_counter_counts_rows_not_items` — progress denominator = stored-row count (multi-turn items store one row per assistant turn): 1 single + 1 multi (2 assistant turns) → 3/3, final line "3 results from 2 items"; fixes live "553/480" on smart-agent-101 (480 items → 587 rows)
- [x] `tests/test_api.py::test_judgment_api_accepts_verdict_detail` — POST /api/judgments accepts a qualified both-bad draw (winner='tie' + verdict_detail='both_bad'); 422 when detail rides a non-tie winner
- [x] `tests/test_judging.py::test_both_bad_end_to_end_dispatch_report_csv` — glue: mocked both-bad judge → dispatch (both orders) → suite_report buckets/both_bad_rate → pair_summary draw math → CSV both_bad/pct_both_bad parsed and pinned
- [x] release-gate PII sweep over the FULL tracked tree (all 11 operator strings) — CLEAN after fixing the sticky-inputs.js first-name comment; qa/suites-private verified untracked + gitignored

### J4 — Combined kick-off (evals + judging, added 2026-10-15)
- [x] `tests/test_eval_jobs.py::test_combined_job_end_to_end` — target_a/target_b replays both models from the suite's latest definition snapshot, then judges the two fresh runs run-scoped; two judgments stored (both orders); run names carry the kick-off provenance; API keys never echo on the job record (targets expose url+model only)
- [x] `tests/test_eval_jobs.py::test_combined_job_validation` — 422s: one target without the other, non-http url, empty model, identical A/B models, judge-is-contestant guard (UI discriminator substring), eval_run_ids+targets mutually exclusive, suite without a definition snapshot
- [x] `tests/test_eval_jobs.py::test_combined_job_replay_failure_is_honest` — a replay that stores zero results hard-stops the job as error before judging ("nothing to judge"), zero judgments written
- [x] staged job record: `combined`/`stage` (replay-a → replay-b → judging) drive the UI stage chip; one Stop flag flows through all three stages
- [x] Live smoke: combined kick-off from /judgments against two real models on the smoke suite (2026-10-15, deploy host: guard 422 → override accepted → staged job replay A/B complete on the suite's 6-item snapshot → 12 judgments stored (6 items × both orders), runs #4/#5, status_line honest; winners a=7/tie=5 from the family judge)

### J4 — Pair page — category summary (added 2026-10-15)
- [x] `tests/test_reports.py::test_pair_summary_per_dimension_math` — exact per-dimension counts/percent/score (wins + ½ ties 0–100) + overall; Wilson 95% present on A's win share; rows sorted n desc then name
- [x] `tests/test_reports.py::test_pair_summary_orientation_invariant` — calling with model_a/model_b swapped mirrors the a/b columns (counts/percent/score); n, ties, row set, judges, templates unchanged
- [x] `tests/test_reports.py::test_pair_summary_judge_filter_scoping` — judge_model / judge_template scope counts AND the judges/templates lists (either field alone filters; unknown judge → honest empty)
- [x] `tests/test_reports.py::test_pair_summary_multiturn_rows_fold_into_base_dimension` — '#t2' turn rows land in the base item's dimension; no '#'-suffixed rows leak
- [x] `tests/test_reports.py::test_pair_summary_uncategorized_fallback` — no definition snapshot → honest 'uncategorized' bucket
- [x] `tests/test_reports.py::test_pair_summary_empty_pair_honest` — unknown pair → rows [], overall n=0, null percents, zero scores, no wilson
- [x] `tests/test_evalviews.py::test_pair_summary_endpoint_caller_orientation` — GET /api/judgments/pair/summary: caller-orientation through HTTP ('a' = caller's model_a won), 404-free shape with judges/templates lists
- [x] `tests/test_evalviews.py::test_pair_summary_endpoint_404_and_empty_pair` — unknown suite → 404 (require_parent); unjudged pair → 200 with honest empty summary
- [x] Page contract: `tests/test_evalviews.py::test_pair_page_has_category_summary` + `tests/test_eval_runs_api.py::test_drilldown_pages_carry_shell_identity` pin `id="jsum"` + 'Category summary' on /judgments/pair
- [x] Column order honors the kick-off's run naming (live check): the page reads `/api/eval-runs`, maps each URL fingerprint to its newest `A:`/`B:`-prefixed run on that suite, and displays A on the LEFT (URL order as fallback) — applied to the summary table, the meta win counts, and the detail SxS cards below

### J4 — Judge template v3 — mandatory one-sentence reason (added 2026-10-15)
- [x] `tests/test_eval_jobs.py::test_extract_reason_cases` — REASON parse: present / absent / whitespace collapse / 300-char cap with ellipsis
- [x] `tests/test_eval_jobs.py::test_extract_rationale_bare_verdict_is_none` — a bare `VERDICT: X` is NOT a rationale (None), chatter after the verdict still kept, REASON preferred when present
- [x] `tests/test_eval_jobs.py::test_judge_prompt_demands_reason` — v3 prompt carries the REASON spec; template version bumped
- [x] `tests/test_eval_jobs.py::test_judge_item_followup_captures_reason` — verdict-only reply → exactly one supplemental call per order; sentence stored as rationale on both judgments
- [x] `tests/test_eval_jobs.py::test_judge_item_followup_failure_stays_valid` — follow-up exploding never invalidates a paid-for judgment (rationale NULL, no error)
- [x] `tests/test_judging.py::test_judge_item_both_orders_stored_with_remap` — extended: REASON present → no follow-up (exactly 2 calls); rationale captured verbatim
- [x] `tests/test_judging.py::test_judge_item_tie_and_invalid_verdict_retry` — retry loop unaffected; nudge terminates cleanly on a bare tie
- [x] `tests/test_eval_jobs.py::test_extract_verdict_truncation_guard` — truncated deliberation (no VERDICT line, token mid-sentence) → None → retry; final-mention convention preserved when the token ends the reply
- [x] `tests/test_eval_jobs.py::test_judge_once_default_max_tokens_bumped` — judge call budget 2048 (kimi chatter overran 1024 on ~9% of the first real run)

### J4 — Judge presentation trims serving-stack wrappers (added 2026-10-23)
- [x] `tests/test_judging.py::test_select_pairs_trims_serving_preamble` — the judge worklist trims leading/trailing whitespace from out_a/out_b (multiturn history included); the store stays verbatim (append-only doctrine, same as the reference scorer). Audit trigger: evals-101 run #8 vs #9 stored 27 judgments on byte-identical-after-trim answers, every one decided against the side carrying qwen's leading blank-line preamble (e101-vision-003 judgment #480 the owner's example)

### J4 — Mirrored mixed kick-off — reuse a run on either side (added 2026-10-23)
- [x] `tests/test_eval_jobs.py::test_mirrored_mixed_job_end_to_end` — `eval_run_a` + `target_b`: only model B replays (spy asserts replay list), the stored A run serves its outputs verbatim into judging, 2 judgments stored (both orders), job record echoes the stored side as model-fingerprint-only (no URL), fresh B side url+model, api keys never echo
- [x] `tests/test_eval_jobs.py::test_mirrored_mixed_validation_422` — `eval_run_a`+`target_a` mutually exclusive; stored A alone → "requires target_b"; `eval_run_a`+`eval_run_b` → steered to `eval_run_ids=[a, b]`; `eval_run_ids` stays exclusive with mirrored mode; unknown run id → "not found"
- [x] Validation order: side-specific exclusivity fires before the generic asymmetric-target shape check (specific 422 detail wins)
- [x] UI: "reuse a recent run" picker on BOTH model cards, moved to the top of each card; picker disables that side's endpoint/reasoning fields; runs-on-both-sides → run-scoped re-judge payload (`eval_run_ids`), same-run-twice rejected client-side; kick button label mirrors the mode; `tests/test_static_wiring.py` pins both pickers + payload keys

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
- [ ] Live smoke: dispatch a 1-point bench from `/create` against demo host A during a quiet window; verify row lands in /compare
- [ ] Live smoke: eval replay from `/create` against qwen rotation; verify rows land in /evals

### Async bench jobs + progress indicator (added 2026-09-21)
- [x] `tests/test_ops_dispatch.py` — openai transport dispatch: happy path stores points with no power fields (ADR 002), preflight failure surfaces actionable DispatchError, missing model → DispatchError, invalid transport name → 422; all network mocked
- [x] Job API live-validated on the deploy host: job poll mid-run returns running + per-point metrics + self-correcting ETA; done state carries run_id; unknown job id → 404; openai job without model → 422 before any thread starts
- [x] Job-status 500 regression (missing `time` import crashed running-job polls; fixed a7cba82) — poller retries transient failures client-side
- [ ] Load: two concurrent jobs against different targets do not interleave stored rows

### Run visibility — soft hide (added 2026-09-21)
- [x] `tests/test_api.py::test_run_visibility_hide_unhide_roundtrip` — hide drops the run from /api/runs but direct fetch still works; unhide restores; `include_hidden=true` escape hatch; newest-flag-wins with reason preserved
- [x] `tests/test_api.py::test_run_visibility_validation_and_404` — non-bool hidden → 422, missing key → 422, unknown run → 404 (both endpoints)
- [x] `tests/test_api.py::test_run_visibility_survives_append_only_triggers` — UPDATE/DELETE on run_visibility rejected at the storage layer
- [x] `tests/test_api.py::test_hidden_run_excluded_from_compare` — compare rejects a hidden run with actionable 404 naming the unhide call; unhidden run compares again
- [x] Live-validated on the deploy host: hide run 9 → gone from /api/runs (reason stored), unhide → back; 144 tests green

### Cross-cutting
- [x] Top-bar chrome is page-independent (added 2026-10-23): `tests/test_static_wiring.py::test_no_page_level_nav_or_header_css` — no page `<style>` may style the injected chrome (`nav a` / `header` rules are dead); `tests/test_static_wiring.py::test_nav_css_owns_tab_decoration` — nav.css `.nav-section-head` kills the UA link underline. Seven pages shipped rail-era `nav a { margin-right: 1rem }` + `header { border-bottom … }` blocks that hit the injected `.topnav`: per-page spacing + underline differences (create/kickoff/compare underlined tabs + tighter spacing vs the rest). Live-checked post-deploy: identical pill/tab geometry on every route
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
| S7 | combined throughput = SUM across streams (capacity), mean still tracked | `tests/test_bench.py::test_aggregate_combined_is_sum_of_stream_rates` | done |
| S8 | combined sum skips null-rate streams (no fabricated zeros) | `tests/test_bench.py::test_aggregate_combined_skips_null_streams` | done |
| S9 | streams=1: combined equals the single stream's rate | `tests/test_bench.py::test_aggregate_single_stream_combined_equals_rate` | done |
| S10 | dispatch payload carries combined fields (stored + returned) | `tests/test_ops_dispatch.py::test_dispatch_bench_streams_two_stores_stream_count` | done |
| S11 | combined metrics selectable on /compare and plot stored values | `tests/test_compare.py::test_compare_combined_throughput_series` | done |
| S12 | combined series honest-null for single-stream points | `tests/test_compare.py::test_compare_combined_honest_null_when_not_multi_stream` | done |
| S13 | run-diff exposes combined metrics as higher-is-better | `tests/test_compare.py::test_run_diff_combined_metrics_direction_and_deltas` | done |
| S14 | /compare + run-detail pages render combined columns/checkboxes | `tests/test_compare.py::test_compare_page_lists_combined_metric_checkboxes`, `tests/test_api.py::test_run_detail_page_has_combined_throughput_columns` | done |
| S15 | live job table stays compact (7 columns) for single-stream jobs | `tests/test_ops_api.py::test_create_view_live_table_single_stream_stays_compact` | done |
| S16 | live job table adapts: Σ prefill/decode pair for multi-stream jobs | `tests/test_ops_api.py::test_create_view_live_table_adapts_for_multi_stream_jobs` | done |
| S17 | UI floats render at 2dp across all views (presentation-only) | `tests/test_compare.py::test_ui_floats_render_at_2dp` | done |
| S18 | openai transport captures MTP acceptance from usage details | `tests/test_bench_openai.py::test_mtp_acceptance_extracted_from_usage_details`, `tests/test_ops_dispatch.py::test_dispatch_bench_openai_mtp_captured_from_usage_details` | done |
| S19 | MTP honest-null when usage carries no details | `tests/test_bench_openai.py::test_mtp_fields_null_when_usage_details_absent` | done |
| S20 | power sampling opt-in per dispatch: sampled + stored with provenance | `tests/test_ops_dispatch.py::test_dispatch_bench_power_host_opt_in_samples_and_stores` | done |
| S21 | power_host validation (blank/non-string rejected) | `tests/test_ops_dispatch.py::test_dispatch_bench_power_host_validation` | done |
| S22 | mid-sweep failure stores completed points as a partial run | `tests/test_ops_dispatch.py::test_dispatch_bench_mid_sweep_failure_stores_partial_run` | done |
| S23 | failure with zero completed points: no run, 502 unchanged | `tests/test_ops_dispatch.py::test_dispatch_bench_failure_with_zero_points_still_502` | done |
| S24 | success payload carries status='complete' | `tests/test_ops_dispatch.py::test_dispatch_bench_success_reports_complete_status` | done |
| S25 | partial preservation on the openai transport | `tests/test_ops_dispatch.py::test_dispatch_bench_openai_mid_sweep_failure_stores_partial_run` | done |
| S26 | /api/runs?status= filter (partial/complete/validation) | `tests/test_api.py::test_runs_status_filter_partial_and_complete`, `tests/test_api.py::test_runs_status_filter_validation` | done |
| S27 | partial-run UI: /runs badge, run-detail marker, live-job link | `tests/test_api.py::test_runs_list_page_shows_partial_badge`, `tests/test_api.py::test_run_detail_page_shows_partial_marker`, `tests/test_api.py::test_create_page_links_partial_run_on_failure` | done |

## 3. Fixture suites (`qa/suites/`)

Seeded demo data + reference eval suites, format documented in each file's
header. Used by: integration tests, dashboard demo mode, and as the first real
eval payloads for J3/J4 implementation.

## 4. Exit criteria per milestone

A milestone is "done" when: all mapped tests pass on the deploy host, the live smoke
layer exercises the feature against :24816, and AGENTS.md §6 records it.
| S28 | default run name is humanized `host · model · targets` | `tests/test_ops_dispatch.py::test_default_run_name_humanized` | done |
| S29 | custom name flows through dispatch (native + openai) into the stored run | `tests/test_ops_dispatch.py::test_dispatch_bench_custom_name_stored`, `tests/test_ops_dispatch.py::test_dispatch_bench_openai_name_stored` | done |
| S30 | unnamed dispatch stores the computed default name | `tests/test_ops_dispatch.py::test_dispatch_bench_default_name_when_unnamed` | done |
| S31 | eval replay creates an eval_runs row; items stamp eval_run_id; name stored (or NULL) | `tests/test_ops_dispatch.py::test_dispatch_eval_creates_named_eval_run`, `tests/test_ops_dispatch.py::test_dispatch_eval_unnamed_eval_run_stores_null_name` | done |
| S32 | name is the only mutable column on runs (trigger-guarded label) | `tests/test_api.py` rename route contract + db trigger checks | done |
| S33 | /runs second column shows the name (`—` when unset) | `tests/test_api.py::test_runs_list_page_shows_name_column` | done |
| S34 | compare legends + 📌 baseline label + diff headers prefer the name | `tests/test_compare.py` payload assertions + static-source checks | done |
| S35 | TTFT renders in seconds (ms÷1000) across run detail, live table, compare | `tests/test_ops_api.py::test_create_view_ttft_column_shows_seconds` + static-source checks | done |
| S36 | live-job detail table scrolls horizontally instead of clipping | `tests/test_ops_api.py::test_create_view_detail_table_scroll_container` | done |

## Coverage validation (2026-09-22 final check-in)

Measured with pytest-cov (`pytest --cov=src/arcturos`):

| Module | Stmts | Cover |
|---|---|---|
| schemas.py | 74 | 99% |
| multiturn.py | 59 | 98% |
| bench_openai.py | 110 | 97% |
| power.py | 45 | 100% |
| preflight.py | 64 | 94% |
| bench_stream.py | 65 | 94% |
| compare.py | 121 | 93% |
| reports.py | 77 | 92% |
| bench.py | 226 | 91% |
| main.py | 493 | 91% |
| db.py | 63 | 89% |
| ops.py | 247 | 84% |
| **TOTAL** | **1645** | **92%** |

232 tests, all passing. The remaining uncovered lines are dominated by
interactive/live paths that require a real inference server (live bench
sweeps against dead targets, thread-timing edges in the async job
registry, preflight network error branches) — these are exercised by
the live-validation passes recorded in AGENTS.md instead of unit tests.
Coverage tooling: pytest-cov added to the venv (dev-only, not a runtime
dependency).
