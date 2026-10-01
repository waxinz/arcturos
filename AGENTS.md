# AGENTS.md — Arcturos

**Project:** Arcturos — local-first LLM benchmarking & evaluation suite
**Interface:** local web dashboard, `0.0.0.0:24816` (owner decision 2026-09-17)
**Repo host:** taupo (`alexei@192.168.122.1`), path `~/arcturos`
**Owner:** Alexei
**Created:** 2026-09-17
**Status:** Phase 1 (foundation) — see [Current Status](#current-status)

> **Read this file first.** It is the single source of truth for project
> progress, conventions, and how any agent (human or AI) picks up work at any
> point. Update it **in the same commit** as any meaningful change.

---

## 1. Mission

Arcturos benchmarks and evaluates LLM inference servers and stores every result
for future comparison. It answers two questions, repeatably:

1. **How fast is a model?** — decode tok/s, prefill tok/s, TTFT, wall time,
   MTP/speculative acceptance, power draw — at multiple context lengths.
2. **How good is a model?** — prompt-based evals (single- and multi-turn)
   scored by external LLM judges, including blind pairwise A/B preference.

All data is local-first (SQLite), reproducible, and exportable.

## 2. Key user journeys (from the PRD)

| # | Journey | PRD section |
|---|---------|-------------|
| 1 | Benchmark a model at multiple context lengths, store metrics | PRD §J1 |
| 2 | Visual comparison across models and runs | PRD §J2 |
| 3 | Administer/run prompt evals (single & multi-turn) | PRD §J3 |
| 4 | Blind A/B scoring of two models' outputs by an external judge | PRD §J4 |
| 5 | Reports on eval suite performance (win rates, coverage) | PRD §J5 |
| 6 | Supporting features (model registry, env snapshots, export) | PRD §J6 |

## 3. Repository layout

```
arcturos/
├── AGENTS.md            ← this file (progress + conventions)
├── PRD.md               ← product requirements (journeys, acceptance criteria)
├── docs/
│   ├── ux-design.md     ← UX design (Phase 3)
│   └── decisions/       ← architecture decision records (ADR-style, one file per decision)
├── qa/
│   ├── test-plan.md     ← QA strategy (Phase 4)
│   └── suites/          ← concrete test suites / fixtures
├── src/arcturos/        ← implementation (Phase 5+; language: Python 3.10+)
├── scripts/             ← operational helpers (bench runners, seed data)
└── data/                ← NOT committed. Local SQLite + result artifacts only.
```

## 4. Conventions

- **Commits:** imperative mood, reference journey/phase, e.g.
  `bench: capture MTP acceptance per draft step (J1)`.
- **Every phase ends with:** tests passing, AGENTS.md status updated, PRD
  checkboxes ticked for completed acceptance criteria.
- **Data model changes** require an entry in `docs/decisions/` and a schema
  migration note in AGENTS.md §Current Status.
- **Metrics are append-only.** Never mutate stored run data; supersede runs.
- **Bench methodology:** cold-cache measurement (fresh UUID prompt prefix,
  server restart between context-size tests where applicable). See
  `docs/decisions/` for the full methodology record.
- Language: Python 3.10+ (taupo has 3.12.3). SQLite stdlib. Dashboard: local
  web app bound `0.0.0.0:24816` (LAN-accessible; no auth beyond LAN boundary
  for v1). Charts rendered client-side (Chart.js or similar); API = JSON/REST.

## 5. Environment facts (taupo)

- Ubuntu 24.04, git 2.43.0, Python 3.12.3, `/` has 400+ GB free.
- GPU inference servers available for benchmarking targets (see
  home-server inventory skill): `ruapehu` 10.10.10.122 and
  `pakuranga-inf` 10.10.10.222 — 6× RTX 3090 each, llama.cpp `llama serve`
  on port 8000 (DS4-Flash, 262k ctx). `aotea` 10.10.10.14:8080 is another
  OpenAI-compatible target. `langfuse` on taupo (:3001) is an alternative
  eval-tracing backend but Arcturos is local-first by design.
- SSH as `alexei` is passwordless to all boxes.

## 6. Current Status

**Phase:** 2 of 6 — PRD signed off; UX design + implementation underway.
**Next:** J1 bench core hardening + live-sweep soak; coverage 92% (232 tests).

- [x] Repo initialized on taupo (`~/arcturos`, branch `main`)
- [x] AGENTS.md created (this file)
- [x] PRD.md drafted from key user journeys
- [x] PRD reviewed and signed off by Alexei
- [x] UX design doc (journey walkthroughs, wireframe-level flows)
- [x] QA test plan + suites (coverage-validated 2026-09-22: 92% total, 232 tests)
- [x] J1 bench core: bench.py (tokenize sizing, cold UUID prompts, timings parser, storage, export) — commit f7c2fee
- [x] J1 validated end-to-end: real bench of ruapehu DeepSeek-V4-Flash (512+4096 tok) stored via API, visible at :24816/api/runs/1/benchmarks
- [x] M3 QA: test plan + fixture eval suites + demo seed script + J3/J4 contract tests — commit 4e0cb3b
- [x] API key auth end-to-end (J6): optional `api_key` on every outbound request — bench (tokenize/completion/props), bench_stream, eval dispatch, preflight probes, `/api/ops/bench` + `/api/ops/eval` body field, `/create` UI password fields (bench + eval); keys never persisted to the store. Authed-mock tests added (98 green).
- [x] J2 comparison UI: compare/diff endpoints + chart views at /compare /diff (ef887c5)
- [x] J4 blind A/B judge runner: scripts/ab_judge.py (e958d32+fixes 69d6c21, 785bc1f, 2708523, b73ea6f)
- [x] J5 report aggregation: src/arcturos/reports.py (5ee2768)
- [x] J3 multi-turn replay: src/arcturos/multiturn.py + live validation (807f24d)
- [x] J4 multi-turn judging: per-turn blind judgments (turn-suffixed item ids)
- [x] Eval/judgment/reports views: /evals /judgments /reports + /api/reports/suite/{id} (42218eb + tests 5c08e31)
- [x] TTFT streaming variant: bench_stream.py (llama.cpp) + bench_openai.py (litellm/vLLM) with reasoning-model TTFT semantics
- [x] Power draw: power.py validated (134W on ruapehu during bench) — dedicated hosts only per ADR 002
- [x] Power provenance wired into store path: benchmarks.power_host + power_gpu_index columns, db migration for pre-existing DBs, API enforces watts-require-host (422)
- [x] J7 CRUD + dispatch: ops.py (dispatch_bench/dispatch_eval), /api/ops/bench + /api/ops/eval, /create view (6 form cards: bench dispatch, suite create, suite item editor, eval replay, manual bench point, judgment record); PRD J7 + ux-design §J7 + test-plan J7 sections written; 86 tests green
- [x] M9 hardening remainder: multi-host compare metadata (server_url in compare payload), health preflight before dispatch (J6), seed power into seed_demo path
- [x] First real benchmark captured via /create dispatch (live smoke, quiet window)

### Log (newest first)

- 2026-09-30 — Grouped navigation UX round: the flat nine-link nav became
  three labelled sections — Benchmarks (Kick off / Runs / Compare / Diff),
  Evals (Evals / Judgments / Suite create / Suite items / Replay kick-off),
  Settings (Models / Baselines) — implemented as .nav-groups blocks with
  hairline dividers + uppercase labels in theme.css, wired on all ten
  pages; the per-page highlighter (inlined under the theme.js script tag)
  is now hash-aware. The eval kick-off cards moved conceptually under the
  Evals group: nav links /create#eval-suite-create, /create#eval-suite-items,
  /create#eval-replay-kick anchor-scroll to (and flash) the matching card —
  card ids + a click/lochash router added. /create's stale numerals
  ("1 · Benchmark run" etc.) dropped; cross-card hints name cards instead
  of numbers. The unused card 5 (manual benchmark point) and card 6
  (judgment record) were REMOVED from /create — API surfaces stay (POST
  /api/runs/{id}/benchmarks, /api/judgments); PRD J7 + ux-design amended.
  Operator item set delivered in the same round: /create's model field
  defaults to GLM-5.3-Flash (openai transport), the Power host / GPU index
  pair was removed from the bench card (power stays an API-only body field
  per ADR 002), the Compare picker table gained a Name column in the
  second position (display_name, same fallback rule as /runs), and the
  Diff page's A/B selects label entries '#id — run name' (display_name →
  name → fingerprint chain). test_evalviews nav assertions updated for
  the grouped shape. 229 passed.

- 2026-09-22 — Final check-in: naming UX round + theme + local-time
  datetimes + coverage validation. (1) Run-detail: the rename button
  was appended to #run-actions BEFORE that container was wiped by the
  header-actions block — the button existed in source but never
  rendered (cbed345); the name now has its own line under the title
  with an inline editor (input + Save/Cancel/Escape, empty submit
  clears to default, in-place update without reload — d64ae8f).
  (2) Runs page: off-by-one in the row builder dropped the name cell
  and shifted every column one left (Name column showed server_url —
  66a03c4); display_name = stored name or computed 'host · model'
  fallback at read time so legacy unnamed runs show a usable label
  (d64ae8f). (3) Theme: light/dark toggle (☾/☀︎) on every page —
  dark palette is Dracula; persisted in localStorage, defaults to the
  system preference, Chart.js re-themes live (15691f5); dark-mode
  form controls + visited-link hue + button colors fixed after live
  feedback (9f79a4b). (4) Datetimes render in the viewer's timezone
  (Pacific fallback) via app-time.js; '(UTC)' header claims dropped;
  storage/exports stay UTC ISO (15691f5). (5) Coverage validation:
  pytest-cov added (dev-only) — 92% total, every module >= 84%;
  +16 tests around the gaps (ops endpoint validation, eval dispatch
  validation/rollback, power sampler, multiturn post/validate);
  qa/test-plan.md gains the coverage table. 232 passed. Live on
  taupo throughout (service restarted per deploy, verified 200s).

- 2026-09-22 — Deploy fix: stale append-only trigger upgrade (a6c2862).
  The live restart exposed that CREATE TRIGGER IF NOT EXISTS never
  replaces an existing trigger — taupo's pre-naming database kept the
  old blanket trg_runs_no_update ('any UPDATE aborts') even after the
  runs.name migration, so renames 500'd forever. init_db now compares
  stored trigger SQL against the shipped definition for label-bearing
  triggers (runs, eval_runs) and drops + re-creates on mismatch
  (DDL-only, data untouched, idempotent). Regression test seeds a
  legacy blanket-trigger DB: rename works, measurement UPDATE still
  aborts, second init_db is a no-op. Live-verified on taupo: rename
  API 200, two real dispatches (named + default-name
  '10.10.10.222 · GLM-5.3-Flash · 512/1k') stored correctly against
  pakuranga. 204 passed.

- 2026-09-22 — Run naming + eval-run entity (ADR 003): every benchmark
  run and eval run carries an optional human `name`. Default when
  unnamed: `host · model · targets` computed at dispatch from engine
  metadata (32768 → `32k`, non-kibibyte stays raw), written at store so
  /runs shows real labels. The name is a LABEL, not measurement data:
  `runs.name` is trigger-whitelisted as the ONLY mutable column
  (mirrors models.alias; every other UPDATE aborts) — renames via
  `PATCH /api/runs/{id}/name` (set + clear-to-null + 422/404 contract).
  Eval runs gain a real entity: `eval_runs` table (one replay = one
  row) + `eval_results.eval_run_id` stamping (idempotent migration,
  historical rows NULL) — replay batches are now addressable and
  nameable. Surfaces: /runs second column (honest `—` fallback),
  run-detail meta + rename control, /create name field, compare
  legends + 📌 baseline label, diff headers, CSV exports (runs list +
  compare payload). Docs: PRD J7 run-naming block, ux-design §7
  time-units + wide-tables rules, test-plan S28–S36, ADR 003
  (labels-vs-measurements + eval-run entity). Also: TTFT now DISPLAYS
  in seconds (ms÷1000 at render; storage keeps ms) across run detail,
  live job table, compare charts — labels say `TTFT (s)` / `ttft (s)`;
  and the live-job detail table scrolls horizontally instead of
  clipping under the next card (.table-scroll wrapper). +16 tests —
  203 passed.

- 2026-09-22 — Partial-run preservation: a bench sweep that fails
  mid-way no longer discards the points it completed. New append-only
  `runs.status` column ('complete' | 'partial', NOT NULL DEFAULT
  'complete', CHECK-constrained, idempotent migration — historical rows
  grandfathered as complete). dispatch_bench collects completed points
  via the on_point hook; on mid-sweep failure it stores them as a
  partial run and returns {status: 'partial', run_id, error, points}
  instead of raising — a failure with zero completed points still
  raises DispatchFailure (502 contract unchanged, no empty run rows).
  Also fixed a pre-existing bug the work surfaced: connection-level
  failures (httpx.HTTPError) mid-sweep escaped dispatch's exception net
  as unhandled 500s — now mapped to DispatchFailure/502 like other
  runtime failures. Async jobs surface partial runs in the job record
  (status 'partial', run_id set, actionable status_line); /create's
  live panel links '⚠ run #N (partial) — K completed point(s) kept'.
  /runs badges partial runs ⚠, run-detail marks the title line,
  /api/runs?status=partial|complete filters (422 on other values).
  Docs: PRD J7 amendment, ux-design partial-runs section, test-plan
  S22-S27. +9 tests — 187 passed.

- 2026-09-22 — MTP acceptance + opt-in power on the openai transport
  (runs 29/30 investigation): the openai path ignored tabbyAPI's
  speculative-decoding counters — `completion_tokens_details.
  accepted_prediction_tokens / rejected_prediction_tokens` in the final
  usage chunk now map onto mtp_accepted / mtp_draft_n (accepted +
  rejected; live-verified against pakuranga: 200-token gen → 109/73).
  Power sampling becomes opt-in per dispatch for BOTH transports:
  `power_host` (+ optional `power_gpu_index`) on /api/ops/bench,
  /api/ops/bench/jobs, and dispatch_bench/run_bench_job runs an SSH
  nvidia-smi sampler alongside each point and stores mean W with
  provenance — wired through bench.run_benchmark (native) AND
  _run_openai_bench_points (openai); absent = no sampling (ADR 002
  unchanged — host must be named explicitly, never inferred). /create
  bench card gains Power host + GPU index fields. Dispatch payload now
  carries power fields. +6 tests — 178 passed.

- 2026-09-22 — 2dp display precision everywhere: all numeric rendering
  paths round floats to 2 decimals (run 30 showed
  30.105000000000004 — float-aggregation noise leaking into the DOM).
  run_detail cellText, create.html live-table cell(), compare.html
  fmtNum + Chart.js tooltip label callbacks now toFixed(2); diff.html
  already rounded via fmt(). Integers (ctx, streams, token counts) and
  strings pass through; presentation-only — storage, API, and CSV
  exports keep full fidelity. ux-design §7 gains a numeric-precision
  rule. +1 regression test — 172 passed.

- 2026-09-22 — Combined throughput surfaces in the UI: run detail's
  benchmark table gains Σ prefill/decode t/s columns (values shown only
  for multi-stream points, honest `—` + reason tooltip otherwise —
  setCell gained a per-cell reason override); /compare gains
  decode_tps_combined / prefill_tps_combined as their own checkboxes
  (labels 'Σ decode tok/s (combined)' / 'Σ prefill tok/s (combined)')
  so combined throughput is plottable like any other metric; compare
  baseline-delta coloring treats combined as higher-is-better; diff.html
  legend text names combined t/s; /create's live job detail table adapts
  to the job — single-stream jobs keep the compact 7-column table,
  multi-stream jobs add a Σ prefill/decode pair (decided at kick-off
  from the streams field so columns never flicker mid-run). +7 tests
  (compare series, honest-null single-stream class, run-diff direction +
  deltas, page assertions for /compare checkboxes, run-detail columns,
  and the adaptive live table) — 171 passed.

- 2026-09-22 — Parallel streams: bench dispatch accepts `streams`
  (default 1, max 16, bools rejected); each context length runs that many
  identical concurrent workstreams (ThreadPoolExecutor per target, targets
  stay sequential); per-point aggregation = mean rates/TTFT/token counts,
  max wall_s; the stream count is stored per benchmark row (schema +
  idempotent migration), surfaced in the dispatch payload, job status
  rows, run-detail table, live job detail table, compare metrics
  (neutral), and CSV exports. +4 tests — 160 passed.
- 2026-09-22 — Combined throughput: `decode_tps_combined` /
  `prefill_tps_combined` stored per benchmark row (SUM across all parallel
  streams = true server capacity under concurrency; per-stream means stay
  alongside). Schema + migration, manual-point API, dispatch payload, job
  rows, run-detail + live-job tables, compare metrics (higher-is-better),
  per-run CSV export columns. +3 tests — 163 passed.

- 2026-09-22 — Run-17 decode fix (burst-flush artifact). User reported run 17
  (ruapehu, openai transport) showing 15054.43 t/s decode at 64k and no decode
  at 128k. Root cause verified live: tabbyAPI burst-flushes SSE chunks under
  long-prefill load, collapsing the client-side content-chunk span (point 1:
  256 chunks / 17 ms -> 15054 t/s; point 2: span 0 -> None). The final usage
  chunk carries server-accounted completion_time / completion_tokens_per_sec
  (~72 t/s live) which the bench ignored. bench_openai now resolves decode as
  server per-sec > server tokens/time > client chunk-rate, and the client
  fallback rejects spans below a 0.25 s burst floor (socket drain, not decode).
  ops maps the resolved rate into BenchPoint.decode_tps. 4 new tests: server
  rate wins, completion_time fallback, reasoning-only stream (run-17 point-2
  class), burst span rejected (run-17 point-1 class). 153 passed.

- 2026-09-22 — full-project review round (docs, code, tests): fixed export CSVs shipping empty id columns (eval-results asked `result_id` from `er.*`, judgments asked `judgment_id` from `j.*` — both SELECTs now alias `id AS result_id` / `id AS judgment_id`); made bench-job status snapshots thread-safe (`bench_job_status` copies `points` + rows under `_JOBS_LOCK` so a poll can never serialize a torn row or corrupt the live record); create.html poller now retries network-level poll failures with the same bounded budget as HTTP failures and resets its retry counter per job; root-caused the intermittent `test_second_run_updates_last_seen_not_first` flake — `first_seen` came from a second `_utcnow()` read that could cross a millisecond boundary vs the run's `created_at`; `register_model`/`store_benchmark_run`/`_store_eval_result` now pin registry provenance to the triggering row's own timestamp (one clock read); hidden-run gating unified into `require_visible_run` and applied to `/api/compare/baseline-deltas` + the `baseline=` reference param (previously only series + run-diff checked); diff.html dynamic cells DOM-built (string-concat XSS sink removed, parity with run_detail.html); dead code removed (reports.py placeholder line, ab_judge.py unused contested-set/id vars, mid-file import moved to top); falsy-zero TTFT guards replaced with `is not None` in bench_openai/bench_stream (honest-nulls); compare.html no longer colors power_watts deltas (API direction map says neutral); create.html stale "dispatches are synchronous" hint corrected; test_api EXPECTED_COLUMNS now covers run_visibility; +5 regression tests (export id columns, snapshot isolation, hidden-run baseline-deltas/baseline-ref 404s) — 149 passed.

- 2026-09-21 — Run visibility (soft hide) + docs pass: append-only
  run_visibility flag table (newest-wins, triggers enforced); PUT/GET
  /api/runs/{id}/visibility; hidden runs drop out of /api/runs
  (?include_hidden=true escape hatch) and compare/diff reject them with
  an actionable 404 naming the unhide call; direct fetch + exports stay
  addressable. UI: Hide button on /runs rows, Hide/Unhide toggle on the
  run-detail header (confirm dialog names the append-only semantics).
  Docs: PRD J7 amended (async jobs, transport choice, visibility) + 3
  acceptance boxes ticked; ux-design J7 progress-indicator + visibility
  sections; test-plan J7 async-jobs + visibility sections. Live-validated:
  hide run 9 -> gone from /api/runs with reason stored, unhide -> back.
  144 tests green.

- 2026-09-21 — Fix: bench job status 500 on RUNNING jobs (a7cba82) —
  the elapsed computation called time.monotonic() but main.py never
  imported time; finished-job polls skipped that branch, so the bug only
  bit live sweeps (500 -> UI 'Unexpected token I' JSON parse error).
  Import added; poller now retries transient poll failures (2s backoff,
  20 tries) instead of killing the indicator. 140 tests green.

- 2026-09-21 — Async bench jobs + live progress indicator (803ae7b):
  POST /api/ops/bench/jobs validates synchronously then runs the sweep in
  a daemon thread (in-process job registry, TTL-pruned; storage still
  append-only via the normal dispatch path); GET /api/ops/bench/jobs/{id}
  returns phase (preflight|running|done|error), per-point metrics, elapsed
  and a self-correcting ETA (elapsed/done*remaining, recomputed per point).
  /create bench card: animated progress bar (indeterminate during
  preflight), status line, elapsed/remaining clock, zoomable detail
  (per-point table + live events). Synchronous POST /api/ops/bench
  unchanged for API users. Live-validated: job df1b0acd118e -> run 12
  (2 points, 16.5s) vs pakuranga tabbyAPI; 140 tests green.

- 2026-09-21 — Transport-aware bench card (be64bf6): /create bench card
  gains a transport select (native|openai) + model field wired through to
  dispatch; Check target now runs the preflight kind matching the selected
  transport (fixes the misleading tokenize-404 failure on OpenAI-compatible
  targets); default context targets 65535, 131072, 196608 per operator.

- 2026-09-21 — OpenAI-compatible bench transport (26474d8): /api/ops/bench
  gains transport=native|openai + model; openai runs eval-kind preflight
  (no /tokenize on tabbyAPI/vLLM), cold UUID prompts sized client-side,
  streamed points with client-side TTFT/decode, engine='openai' stored,
  power fields stay None per ADR 002. First real benchmark captured via
  dispatch: run 9 vs pakuranga tabbyAPI GLM-5.3-Flash (512 ctx, 64 tok,
  prefill 386.3 tok/s, TTFT 1258.0 ms, wall 2.847 s). 140 tests green.

- 2026-09-21 — Suite definition snapshots + per-category report breakdown
  (1d0b37e + 34fa9c4): the DB stores item_id only, so /reports showed
  everything as "uncategorized" despite the ux-design category-breakdown
  spec. New append-only suite_definitions table (snapshot per PUT, newest
  wins) + PUT/GET /api/eval-suites/{id}/definition; reports.category_map_for()
  rebuilds the item->category map at report time and suite_report joins it.
  PUT allowlist contract test updated (one allowed PUT surface: definition
  snapshots). Live on taupo: /api/reports/suite/1 now reports 6 categories
  (logic 0/4/0, writing 0/4/0, counting 0/1/3, coding 1/2/1, reasoning 1/0/0,
  instruction-following 1/0/0 — a/b/tie/total). 136 tests green.

- 2026-09-21 — M9 hardening remainder closed (288b1ed + 1b847ee):
  seed demo benches carry power_host/gpu_index per ADR 002
  (demo-seed.json); tabbyAPI returns usage:null — resp.get('usage',
  {}) default never fires on explicit None, so extract_output
  crashed mid-dispatch with AttributeError; fixed with 'or {}' in
  multiturn.py extract_output + scripts/seed_demo.py; regression
  test test_extract_output_null_usage added; live re-verified: eval
  dispatch vs pakuranga tabbyAPI GLM-5.3-Flash stored eval_result
  id 38, latency 794.1ms, 133 tests green.

- 2026-09-17 — §4.6 backlog finished (bcd1cb8): compare baseline overlay
  (dashed 📌 reference dataset + per-run delta tables with direction-aware
  good/bad colouring; deltas only where BOTH sides have the metric at the
  same ctx length — honest gaps, never fabricated zeros; pinned and ad-hoc
  baselines both selectable); CSV export everywhere — compare payload
  (schema_version arcturos-compare-v1, fingerprint + host columns on every
  row, optional baseline reference row), /runs, per-run benchmarks, eval
  results, judgments (arcturos-v1 rows; missing metric = empty cell);
  export buttons on /runs /evals /judgments /compare and run detail;
  eval drill-down keyboard parity (Tab focus, Enter/Space toggle,
  aria-expanded). 129 tests green; live-verified on taupo (real delta
  rows + both CSVs download with attachment headers). §4.6 is now fully
  implemented; UX review findings all closed.

- 2026-09-17 — Model registry + baseline tracking (§4.6 J6, commits
  192dea7 + 7774f0d): /models view (cards per fingerprint: alias-only
  edit trigger-enforced, engine, first/last seen, run count) and
  /baselines view (pin/re-pin/unpin per model × metric-family, wrong-
  model pin rejected 422). Auto-registration on first sighting in bench
  dispatch, eval replay, and manual run create; insert-only registry
  backfill in init_db so pre-existing taupo runs appear without touching
  stored rows. Run-detail gained "Pin as baseline" (confirm dialog);
  compare legends show registry aliases as 'alias (short-fp)'. API
  contract test amended: PATCH /api/models/{fp} + DELETE
  /api/baselines/{id} are the two allowed non-data surfaces (PUT still
  forbidden). 116 tests green; live-verified on taupo (pin, re-pin,
  alias edit, unpin, backfill of 4 historical fingerprints).

- 2026-09-17 — UX review round (both reviewer agents finally completed
  after gateway 429/timeout retries; 11 findings deduped) + fixes
  (6233984): DispatchFailure/DispatchError split (runtime target failures
  now 502 with 'check the server' guidance, input/preflight stay 422);
  preflight moved to POST — api_key travels in the body, never a URL (key
  leak into access logs/history flagged by both reviewers); actionable
  preflight failure detail (names checks + target + next step); UI renders
  Pydantic 422 detail arrays as 'field: msg' instead of '[object Object]';
  bench form warns on dropped non-numeric targets; honest nulls in
  run-detail ('—' + AA-contrast + tooltip, no falsy-zero '?'); run-detail
  header actions (compare/diff/copy-JSON) + engine-env <details> block;
  compare view: busy-disable, 'select ≥2 runs' hint, decode_tps default
  only, accessible select-column header; eval drill-down keyboard note
  deferred. BONUS: fixed pre-existing test bug — _pass_preflight()
  permanently replaced the preflight module attribute, leaking ok:stub
  checks into later tests; now a restoring fixture. 103 tests green,
  live-verified on taupo.

- 2026-09-17 — UX backlog landed (1b2dcaa): confirm dialogs on
  judgment-record + manual-bench-point append posts (ux principle 5);
  eval output click-to-expand drill-down in /evals. 99 tests green,
  live on taupo. Backlog now: multi-host compare metadata (M9).
- 2026-09-17 — UX review and fixes (862ef4e): run drill-down view
  /runs/{id} (meta + per-point metrics, honest nulls) + clickable run ids
  on /runs; interactive preflight 8s per-check timeout (dead target
  answers <=16s, was 40s). Eval dispatch verified to consume api_key
  end-to-end (single + multi-turn + UI). 99 tests green, live-verified
  on taupo. Remaining UX backlog: confirm dialogs for re-judge,
  eval-results drill-down view, multi-host compare metadata (M9).
- 2026-09-17 — API key auth end-to-end (14edfb2 + 5733999): optional
  `api_key` threaded through the whole outbound path — bench.py
  (_client headers), bench_stream.py, ops.dispatch_bench/dispatch_eval,
  preflight module + `/api/ops/preflight?api_key=`, `/api/ops/bench` +
  `/api/ops/eval` body fields, `/create` UI password fields (bench card +
  eval card + preflight "Check target" buttons pass them). Keys are
  never persisted (store/log/export). Authed-mock tests (llama.cpp
  `--api-key` behavior) added; 98 tests green. Docs updated: PRD J6/J7,
  ux-design §1/§3, test-plan J7.
- 2026-09-17 — UX design doc written: `docs/ux-design.md` — journey
  walkthroughs J1–J6, CLI surface spec, dashboard view specs (M2).
- 2026-09-17 — J7 CRUD + dispatch landed (8cfd214 + f979b12): ops.py dispatch layer, /api/ops/bench + /api/ops/eval endpoints, /create view (6 form cards), power provenance (benchmarks.power_host + power_gpu_index with db migration; API rejects watts without host per ADR 002), bench.py nounlets→nounits fix (second instance of the same typo bug). PRD J7 + ux-design §6/§8 + qa/test-plan J7 written to match. 86 tests green. Dispatch validation contract: bad shapes → 422, server failure → 502, unknown suite checked BEFORE calling the target, eval batches transactional (rollback on failure). Live-validated on taupo: suite create 201 → eval replay vs qwen rotation 200, result stored. Follow-up fixes: model_fingerprint passed as chat model name (litellm 400 root cause — hardcoded "test" was rejected, d77ff9d); single-turn wall latency real (was stub 0.0, 5820b6d).
- 2026-09-17 — M9 surfaces complete: all 6 dashboard views live, 71 tests green on taupo. ADR 002: power sampling only on dedicated inference hosts (litellm is a proxy; qwen serving hosts elsewhere); never bench shared servers with unacknowledged queueing (ruapehu TTFT artifacts 32s-254s were self-inflicted; true cold TTFT 1.5-9s). Reasoning-model bench semantics: TTFT = first chunk of ANY payload; decode = content chunks only. qwen rotation validated: TTFT ~1.9-2.2s, prefill est ~373-450 tok/s, decode ~82-101 tok/s content.
- 2026-09-17 — J3 multi-turn complete: sr-101/sr-102 replayed vs both models, per-turn records stored (item_id #t-suffix), per-turn blind judging working. Final smoke-reasoning v1 report (n=18 judgments, 2 judges): overall a/b/tie 3/11/4 — DeepSeek 16.7% / GLM 61.1% / tie 22.2%; per-judge: qwen 0/6/2, laguna 3/5/2. All 6 journeys J1-J6 now have working implementations with real data.
- 2026-09-17 — J2/J4/J5 landed and validated with real data: smoke-reasoning suite replayed live vs DeepSeek-V4-Flash (ruapehu) + GLM-5.3-Flash (pakuranga-inf); blind judging via qwen rotation + laguna2.1-s judges on litellm (localhost:4000/v1, key sk-1234); report: overall 1/8/3 (a/b/tie, n=12), GLM 66.7% wins, per-judge + per-category breakdowns working. Judge is NOT a contestant. Lessons: reasoning models (DeepSeek-V4-Flash, qwen3.8) return empty content — read reasoning_content, max_tokens>=512; model fingerprints vary by server suffix — normalize identity in ab_judge; litellm /v1 suffix must be stripped.
- 2026-09-17 — J1 bench core landed (f7c2fee) + validated live; QA suite landed (4e0cb3b). 33 tests green on taupo. Bench targets: ruapehu .122:8000 (DeepSeek-V4-Flash) + pakuranga-inf .222:8000 (GLM-5.3-Flash) confirmed live; aotea .14:8080 down.
- 2026-09-17 — Dashboard skeleton live: FastAPI+SQLite append-only schema, 14 tests passing on taupo, systemd user service at 0.0.0.0:24816 (commit 789eaf3). UX design doc committed (f03ef37).
- 2026-09-17 — PRD signed off; dashboard decided: web UI on 0.0.0.0:24816.
- 2026-09-17 — Repo created on taupo; AGENTS.md + PRD.md committed (Phase 1).

## 7. How to pick up work (any agent)

1. `git -C ~/arcturos pull` (or just read if local) and read this file.
2. Check §6 Current Status: what phase, what's next.
3. Read `PRD.md` for the journey you're implementing; check `docs/decisions/`
   for constraints already decided.
4. Work, test, commit, then **update §6 (status + log)** in the same commit.
5. Never commit `data/` contents or credentials.
