# Arcturos — PRD v0.2

Local-first LLM benchmarking & evaluation suite.
Owner: the operator. Status: v0.2 — eval-system revision (2026-10-15).

---

## 1. Vision

A repeatable, local-first tool that measures **how fast** an LLM serves tokens
and **how well** it answers prompts, stores every measurement, and makes
comparisons across models and runs effortless.

## 2. Personas

- **P1 — Benchmark operator (the operator):** technical, wants cold-cache numbers,
  power draw, MTP acceptance, and to compare quantization/engine changes over time.
- **P2 — Eval operator:** same person, different hat; wants eval suites run
  against models and judged by an external LLM rater.

## 3. Key user journeys

### J1 — Benchmark performance at multiple context lengths
**Actor:** P1 · **Goal:** capture performance metrics for a model across
several context sizes and store them for future reference.

- The operator selects a target server (OpenAI-compatible endpoint), a model,
  and a set of context-length test points (e.g. 4k / 32k / 128k / 250k).
- The harness sends prompts sized to each context point (tokenized via the
  server's `/tokenize`, not char-estimated), each with a **fresh random UUID
  prefix** so no run is contaminated by KV/prefix cache.
- For each point, the harness records per run:
  - `prefill_tps` (prompt eval tok/s), `decode_tps` (generation tok/s),
  - `ttft` (time to first token), `wall_time`, `output_tokens`,
  - `mtp_draft_n`, `mtp_accepted`, derived acceptance rate (where engine
    reports it; vLLM via `/metrics` diff, llama.cpp via inline `timings`),
  - power draw (W) sampled during the run where `nvidia-smi` is reachable,
  - engine metadata: server build, flags/cmdline, model file/repo, quant,
    KV cache type, context size, power limit.
- Metrics land in a local SQLite DB; the raw request/response artifacts are
  optionally stored alongside (JSON).
- A run is **append-only**; re-running supersedes, never overwrites.

**Acceptance criteria**
- [ ] One command benchmarks a model at N context lengths and writes a run ID.
- [ ] Every listed metric present per context point; missing ones explicitly `null` with a reason.
- [ ] Same-run repeated at the same context point produces a new record, not an overwrite.
- [ ] Cold-cache methodology enforced (UUID prefix; restart documented where required).

### J2 — Visual comparison across models and runs
**Actor:** P1 · **Goal:** quickly see how models compare on stored benchmarks.

- Operator picks models/runs and metrics; the tool renders comparison views:
  - line/bar charts of decode & prefill tok/s vs context length,
  - TTFT curves, acceptance-rate curves, wall-time bars,
  - side-by-side run diff (flags + metrics table with deltas).
- Export to PNG/SVG and shareable JSON.

**Acceptance criteria**
- [ ] Dashboard served at `0.0.0.0:24816` (owner decision 2026-09-17).
- [ ] Compare ≥2 models × ≥1 metric × N context points in one view.
- [ ] Compare two runs of the same model with deltas highlighted.
- [ ] Charts exportable to PNG or SVG.

### J3 — Administer and run prompt-based evals
**Actor:** P2 · **Goal:** run prompt eval suites against a model and record
outputs + performance metrics, single-turn and multi-turn, text and image.

*(v0.2 amendment, 2026-10-15: dimensions, image items, deterministic
reference checks, async replay jobs. The benchmarking side spent a month
learning these patterns — async jobs with live progress, cooperative stop,
partial preservation, preflight, honest nulls, configurable request params.
The eval system inherits them wholesale instead of re-learning.)*

- Eval suites are versioned collections of items (JSON), each item:
  - `id`, `type` (`single-turn` | `multi-turn`), `dimension` — the quality
    axis the item measures (`reasoning`, `knowledge`,
    `instruction-following`, `writing`, `coding`, `vision`, `safety`),
    `category` (freeform sub-category), optional `system` prompt, tags.
  - Single-turn: `prompt` (+ optional `image` — http(s) URL or a path
    relative to the suite assets dir for vision items).
  - Multi-turn: ordered `turns` with assistant placeholders (unchanged).
  - Optional `reference` — a deterministic check spec (`exact`, `numeric`,
    `contains`, `one_of`, `regex`). Reference items are scored objectively
    at report time with no judge involved; the rest fall to blind pairwise
    preference (J4). Objective where possible, judged where not.
- The runner replays suites against a target model (any OpenAI-compatible
  chat endpoint) and stores every output with token counts and latency.
  Image items are sent as OpenAI `image_url` content parts (local files
  inlined as data URIs — the suite stays self-contained).
- Replay runs as an **async job** (the bench pattern): validate + preflight
  synchronously, replay in a background thread, live per-item progress,
  self-correcting ETA, cooperative operator stop — items completed before
  the stop are kept (partial preservation, `eval_runs.status`), failures
  recorded per item, never swallowed.
- Retry, concurrency, `max_tokens`, `temperature` configurable per dispatch
  (defaults: retries 0, concurrency 1, max_tokens 1024, temperature 0.0).
  Lesson from the smoke-reasoning round: reasoning models burn token budget
  thinking before answering — 512 shipped too small and truncated answers.

**Acceptance criteria**
- [x] Suite definition format supports single-turn and multi-turn items.
- [x] One command runs a suite against a model; all outputs + timings stored.
- [x] Re-run produces a new result set; results traceable to suite version.
- [x] Items carry a dimension (7-value enum); reference items score WITHOUT a judge at report time (reports.reference_scores).
- [x] Vision items (image + prompt) replay against multimodal targets (image → image_url content part).
- [x] Async replay jobs: live per-item progress, ETA, operator stop, partial preservation (`POST /api/ops/eval/jobs`, /evals/run page).
- [x] Retry/concurrency/max_tokens/temperature configurable; single-turn `system` honored.

### J4 — Blind A/B preference scoring with an external judge
**Actor:** P2 · **Goal:** have an external LLM rater blindly prefer between
two models' answers for the same eval prompt.

*(v0.2 amendment, 2026-10-15: the judge moves in-process with the bench
job pattern; position-swap double judging; dimension rubrics; generation
filters. Learnings: LLM judges carry position bias — the first-presented
response wins more often than chance — so every item is judged TWICE, once
per presentation order, both raw judgments stored; a judge that flips with
order nets out as a tie instead of a coin flip. Dimension rubrics focus the
judge on the axis the item measures. Judge-isolation is enforced: the judge
model must not be a contestant.)*

- For each eval item where models A and B both produced outputs, the judge
  runner submits a **blind** pairwise judgment to a judge model (any
  OpenAI-compatible endpoint, default from the config layer): identities
  anonymized (Response 1 / Response 2), presentation order **randomized per
  judgment**. What is SENT to the judge is leakage-free; the stored judgment
  links eval item + both model result IDs + judge template version + judge
  model (identity mapping is internal-only, by design).
- **Position swap:** every item is judged in BOTH orders and both judgment
  rows are stored (append-only, no derived rows). Position bias nets out in
  aggregate; per-item order-disagreement is visible in reports as the
  flip count.
- **Dimension rubrics:** the judge prompt template is versioned and
  dimension-aware — the item's dimension selects a rubric paragraph
  (correctness-first for reasoning/knowledge/coding, instruction-fidelity
  for instruction-following, prose quality for writing, visual grounding
  for vision, calibration for safety). Template version + rubric are stored
  with every judgment.
- Judging runs **in-process** (`POST /api/ops/judge/jobs`, async job with
  live per-item progress + operator stop). Invalid judge outputs are
  retried per policy (default 1 retry), then recorded in the job summary —
  counted, not silently dropped.
- Multi-turn items are judged per assistant turn with the conversation
  serialized up to that turn (context-preserving, per-turn granularity —
  the store already keys turn records).
- Vision items include the image in the judge payload (multimodal judge
  required for those items; a text-only judge fails them with an honest
  error, never a fabricated verdict).
- Operator can re-judge with a different template/judge model; all
  judgments retained; reports scope to a **generation** (judge model +
  template version), defaulting to newest.
- **Combined kick-off (v0.3, 2026-10-15):** the Judgements page is the
  one kick-off surface for evals + judging. Model A and Model B each carry
  an endpoint URL, an optional API key, and a model name; with both URLs
  filled, one action replays the suite's latest definition snapshot on
  both models, then blind-judges the two fresh runs (staged job: replay A
  → replay B → judging, live per-stage progress, one Stop that preserves
  completed work). Leaving the URLs blank keeps the stored-results
  judging path (model names only). API keys never echo on job records.
- **Pair page category summary (v0.3, 2026-10-15):** the judgment detail
  page (`/judgments/pair`) leads with a per-dimension category summary
  table — Arena-style blind pairwise score = wins + ½ ties scaled 0–100,
  win/tie %, n, and a Wilson 95% CI on A's win share — served by
  `GET /api/judgments/pair/summary`; winner orientation is normalized to
  the caller's model_a/model_b order. Column order follows the kick-off's
  `A:`/`B:` run naming (A displays on the left; URL order is the
  fallback). The per-item SxS rows stay below
  as the detail view.
- **Judge template v3 — always capture a reason (v0.3, 2026-10-15):** the
  v2 prompt demanded "EXACTLY one line … nothing else", so 91% of real
  judgments stored a bare `VERDICT: X` with no rationale. v3 requires
  `VERDICT: X` + `REASON: <one short sentence>`; a verdict-only reply
  triggers exactly one cheap follow-up asking for the sentence (failure
  leaves rationale NULL — honest, never fatal), and a bare verdict line
  no longer counts as a rationale. `JUDGE_TEMPLATE_VERSION` bumped to
  `blind-pair-v3` so reports can scope generations.

**Acceptance criteria**
- [x] Blind randomized pairwise judging (v0.1, scripts/ab_judge.py — API still valid).
- [x] Judgment records link eval item + both model result IDs + judge template version + judge model.
- [x] Every item judged in both response orders; both judgments stored (position swap; the swap changes what the judge SEES).
- [x] Dimension rubrics in the judge template; rubric selection driven by item dimension (blind-pair-v2; v3 adds a mandatory one-sentence REASON with a follow-up nudge for verdict-only replies).
- [x] In-process judge jobs: live progress, operator stop, invalid-verdict retry policy (`POST /api/ops/judge/jobs`).
- [x] Judge-not-contestant enforced at the dispatch boundary (normalized fingerprints; substring match trips).
- [x] Combined kick-off (`target_a`/`target_b`): one staged job replays both models from the definition snapshot, then judges the two fresh runs; staged progress + operator stop; API keys stay off the job record (v0.3).
- [x] Vision items carry the image into the judge payload (image_url content part).
- [x] Pair page leads with a per-dimension category summary (Arena-style score = wins + ½ ties 0–100, win/tie %, n, Wilson 95% CI on A win share); per-item details below (v0.3).

### J5 — Eval performance reports
**Actor:** P2 · **Goal:** report how eval suites performed on models, e.g.
"two models answered 100 prompts in suite X; judge preferred A 62% / B 34% /
tie 4%".

*(v0.2 amendment, 2026-10-15: the headline lens is the DIMENSION breakdown —
"which model is better, on which axes" — alongside deterministic reference
scores that need no judge, Wilson confidence intervals that keep small-n
honest, and judge-generation scoping so re-judges don't pollute aggregates.)*

- Aggregate views: win-rate matrix per suite × model pair, **per-dimension
  breakdown** (the primary lens), per-category breakdown, item-level
  drill-down to the raw outputs and judgments.
- **Deterministic scores:** items with reference checks report
  pass-rate-per-model per dimension alongside judge win rates — objective
  signal, no judge required (scored at report time from the definition
  snapshot; improving a reference spec applies retroactively without
  touching stored rows).
- **Statistical honesty:** every win-rate carries its n and a Wilson 95%
  score interval; n < 20 renders low-confidence tinted (existing UX rule).
  Judge-generation filter (judge model + template version) scopes every
  aggregate; default = newest generation.
- Reports exportable (JSON/CSV/HTML).

**Acceptance criteria**
- [x] Suite-level A/B win-rate summary with counts and percentages.
- [x] Category-level breakdown; drill-down to item level.
- [x] Export to at least one portable format (CSV or JSON).
- [x] Per-dimension breakdown as the headline view.
- [x] Deterministic pass rates per model per dimension (scored at report time from the definition snapshot — retroactive).
- [x] Wilson 95% intervals + n on every rate; judge-generation filter (judge_model + judge_template query params, UI selects).

### J6 — Supporting features (proposed)
- **API keys everywhere:** every outbound request (bench tokenize/
  completion/props, streaming bench, eval chat calls, health preflight
  probes) optionally carries an `Authorization: Bearer <api_key>` header.
  Open servers ignore it; key-protected servers (llama.cpp `--api-key`,
  litellm proxy, vLLM) require it. Keys are entered per-dispatch (UI
  password field / CLI flag / API body field) and are **never persisted**
  to the store, logs, or exports.
- **Model registry:** every benchmark/eval result references a model
  fingerprint (file hash / repo id / quant / engine build) so comparisons
  never rely on a label alone.
- **Environment snapshot:** capture server cmdline, flags, GPU/driver
  versions, power limit per run.
- **Data export:** JSONL/CSV dump of any run or eval for external analysis.
- **Baseline tracking:** pin "reference runs" to diff future runs against.
- **Health preflight:** verify target server health + model loaded before
  launching long benchmarks/evals (slots idle, context size discovery).

### J7 — Create, run, and administer everything from the dashboard
**Actor:** P1/P2 · **Goal:** define and kick off new benchmark or eval runs,
create eval suites and populate them, add benchmark points manually, and
record judgments — all from the web UI (added 2026-09-17 from UI feedback;
extends the original "glance-first UI" principle: the dashboard is now also
the launch surface, not only the reading surface).

- **Create & Run view (`/create`)** with four cards (2026-09-30 amendment:
  the manual benchmark-point and judgment-record cards were removed — both
  unused in practice; the API endpoints remain for scripted use. 2026-10-15
  amendment: the eval replay card moved OFF /create onto a dedicated
  **Run evals page (`/evals/run`)** — replay is an async job there, with
  live per-item progress and an operator stop; the sync API surface
  (`POST /api/ops/eval`) is unchanged for scripted use:
  1. **Bench dispatch** — pick server URL, context targets, n_predict,
     optional API key; the API replays a real cold-cache sweep (tokenize →
     UUID-prefixed prompts → native `/completion`), stores run + benchmark
     points, returns metrics.
  2. **Eval suite create** — name + version; the API echoes the new suite id
     straight into the replay form.
  3. **Suite item editor** — JSON textarea, client-side validated
     (non-empty items, ids present, prompts present; multi-turn shape checked).
  4. ~~Eval replay dispatch~~ — moved to `/evals/run` (2026-10-15).
- **Grouped navigation (added 2026-09-30; amended 2026-10-15):** the
  dashboard nav organises every view into three labelled sections —
  **Benchmarks** (Start, Compare), **Evals** (Create, Run, Results,
  Judgments, Reports), and **Settings** (models registry, baselines).
- **API dispatch endpoints** (thin wrappers over ops.py): `POST /api/ops/bench`,
  `POST /api/ops/eval`. Both accept an optional `api_key` body field passed
  through to every outbound request. Synchronous, bounded by timeouts;
  validation errors are 422, server-side failures 502.
- **Async bench jobs + progress (added 2026-09-21):** `POST /api/ops/bench/jobs`
  validates synchronously, then runs the sweep in a background thread and
  returns `{job_id}` immediately; `GET /api/ops/bench/jobs/{id}` reports the
  phase (preflight/running/done/error), per-point metrics as they land,
  elapsed time and a self-correcting ETA (elapsed ÷ done × remaining,
  recomputed per completed point). The UI renders an animated progress bar,
  a status line, an elapsed/remaining clock, and a zoomable per-point
  detail table. The synchronous endpoint is unchanged for scripted use.
- **Transport choice (added 2026-09-21):** bench dispatch accepts
  `transport: native|openai` (+ `model` for openai) — native targets
  llama.cpp `/completion`; openai targets any OpenAI-compatible chat server
  (tabbyAPI, vLLM, litellm) with reachable+chat preflight and streamed
  client-side TTFT/decode. Engine stored as `openai`; power fields stay
  null per ADR 002.
- **Parallel streams (added 2026-09-22):** bench dispatch accepts
  `streams` (default 1, max 16) — each context length runs that many
  identical concurrent workstreams against the target; per-point metrics
  aggregate across streams (mean rates/TTFT, max wall) and the stored row
  records the stream count, visible in run detail, compare metrics,
  exports, and the live job detail table. Combined throughput
  (`decode_tps_combined` / `prefill_tps_combined` — the SUM across all
  parallel streams, i.e. true server capacity under concurrency) is
  stored per point and tracked in compare reports alongside the
  per-stream means.
- **Multi-stream correctness (added 2026-10-08, ADR 004; decode amended
  2026-10-11, ADR 005):** the openai transport builds a fresh cold prompt
  PER stream (the shared prompt made streams 2..N prefix-cache hits whose
  TTFT was queue time). Combined throughput: `prefill_tps_combined` =
  Σ prompt tokens / max TTFT (the window in which every stream's prefill
  completed); `decode_tps_combined` = **sum of the per-stream decode
  rates** (each rate is measured over its own generation span). ADR 004's
  original shared-window decode formula (Σ generated / (max wall − min
  TTFT)) credited the denominator with OTHER streams' serialized prefill
  time and produced impossible rows like run #129 (Σ decode 4.22 t/s
  below one stream's own 9.28 t/s) — ADR 005 supersedes it. Historical
  caveats: pre-2026-10-08 openai combined values are upper bounds (run
  #103); ADR-004-window decode values (2026-10-08..2026-10-11) are
  prefill-contaminated lower bounds (run #129).
- **Stop active benchmark run (added 2026-10-11, ADR 006):** a running
  bench job can be stopped by the operator — the Kick off page shows a
  Stop button while a sweep is in flight, and
  `POST /api/ops/bench/jobs/{id}/abort` sets a cooperative flag the sweep
  checks between points and inside streaming decode loops. Completed
  points land as a partial run (status `aborted` on the job record,
  `partial` on the run row); points after the flag never start. No
  thread killing, no orphaned measurements — the stop is a first-class
  failure path through the same partial-preservation net.
- **Merged Compare Runs page (added 2026-10-11):** the Runs list and the
  Compare picker are ONE page at `/compare` (nav label "Compare runs").
  The runs picker renders latest-first and keeps the Runs page's
  capabilities (CSV export, per-row + bulk Hide); the charts render at
  the TOP of the page; ttft/wall/prefill/decode checkboxes are pre-checked
  so the page opens comparison-ready. `/` and `/runs` 308-redirect to
  `/compare`; the old index.html is retired.
- **Run visibility — soft hide (added 2026-09-21):** `PUT
  /api/runs/{id}/visibility` `{hidden: true|false, reason?}` appends a flag
  row (append-only, newest wins). Hidden runs drop out of `/api/runs` and
  are rejected by compare/diff with an actionable 404, but stay directly
  addressable and can be unhidden at any time. Nothing is ever deleted.
- **Partial-run preservation (added 2026-09-22):** a bench sweep that
  fails mid-way still stores the points it completed before failing,
  marked `status: 'partial'` on the run row (append-only column, default
  `'complete'`). `/api/runs?status=partial|complete` filters; the /runs
  list and run-detail pages badge partial runs; the live job panel links
  to the partial run instead of dead-ending. Nothing measured is ever
  discarded; failed points are simply absent.
- **Run naming (added 2026-09-22):** every benchmark run and eval run
  carries an optional human `name`. Default (unset): `host · model ·
  targets` computed at dispatch time from engine metadata, context
  targets humanized (32768 → `32k`; non-kibibyte values stay raw). The
  name is a LABEL, not measurement data: stored on the run row
  (`runs.name`, trigger-guarded so ONLY this column is mutable),
  editable before kick-off (`name` field on /create + dispatch APIs) or
  afterwards on the run-detail page (`PATCH /api/runs/{id}/name`),
  rename history is not kept — the column holds the current label, the
  measurements around it stay immutable. Eval runs gain a real entity:
  `eval_runs` (one replay = one row) and `eval_results.eval_run_id`
  stamping, so a replay batch is addressable and nameable. Surfaces:
  /runs second column, run-detail meta, compare chart legends +
  📌 baseline label, diff headers, CSV exports. Honest fallback: unset
  names render `—` in tables; legends/headers fall back to the model
  fingerprint.
- **Judgements mixed kick-off — baseline reuse (added 2026-10-15):** the
  Model B card on /judgments gains a "reuse a recent run on this suite"
  dropdown listing the suite's recent eval runs NEWEST FIRST (`#id ·
  fingerprint · date · result count`, from `/api/eval-runs`). Selecting a
  run disables B's endpoint fields and submits the MIXED kick-off —
  `POST /api/ops/judge/jobs` with `target_a` (fresh replay) +
  `eval_run_b` (stored run id): model A replays, B is NOT replayed, and
  the stored run is judged as the baseline via the same run-scoped
  pairing as `eval_run_ids`. Validation is honest at the boundary: 422
  when `eval_run_b` lacks `target_a`, collides with `target_b` or
  `eval_run_ids`, names an unknown run, a run from another suite, or a
  run with zero stored results. Choosing the dropdown's default option
  (or typing in B's URL) returns to the fresh-replay flow.
- **Recent judgment runs list (added 2026-10-15):** the per-item
  work-detail table under the judging progress bar is REMOVED (the bar +
  status + Stop remain). In its place the Judgements page shows a
  "Recent judgment runs" list: one card per judged A/B run pair × judge
  model (`Run #6 vs #7 · model-a vs model-b · judged by Kimi · N
  judgments · date`), MOST RECENT FIRST, each card deep-linking to the
  existing SxS pair page (`/judgments/pair?suite_id=…&model_a=…&model_b=…`).
  The list is mobile-friendly (stacks under 640px) and refreshes when a
  job finishes.
- **Sticky inputs (added 2026-10-15, replaces config defaults in the
  UI):** every text box on every page (text/url/number/password/email/
  search/tel inputs + textareas, keyed by field id) remembers the last
  typed value in browser localStorage (`arcturos-sticky:<id>`) and
  restores it as the default on the next page load. Sticky values SHADOW
  the form's shipped defaults; emptying a field un-sticks it; the
  config-file defaults remain the server-side fallback for fields left
  empty. Password fields are included by design (operator convenience) —
  localStorage is plain text, keys never leave the browser. The script
  (`static/sticky-inputs.js`) is a global contract: every page loads it
  (enforced by test).
- **Judgment-details prompt expander (added 2026-10-15):** on the SxS
  pair page (/judgments/pair) each item name is a tappable toggle that
  expands the case's PROMPT in a collapsible panel underneath — sourced
  from the suite's latest definition snapshot (GET
  /api/eval-suites/{id}/definition; degrades to '(not in the current
  definition snapshot)' when absent). Single-turn shows system + user
  prompt; multi-turn shows each user turn labeled Turn N with assistant
  placeholders marked ('model reply — stored below'). TEXT AND IMAGES:
  items carrying an ``image`` ref render the actual image in the panel —
  http(s) refs load directly in the browser, suite-relative refs resolve
  through GET /api/eval-suites/{id}/image?item=<item_id> (NEW endpoint:
  ref must appear in the snapshot, so no arbitrary filesystem reads;
  local files inline as base64 data URLs with the same resolution rules
  as replay; a missing/broken image degrades to an honest ⚠ note). Both
  models' answers stay in the SxS cards directly below. The toggle is a
  real button (keyboard + aria-expanded accessible), full-width tap
  target on mobile; images cap at 22rem tall and shrink to screen width.
- **Constant sub-nav (added 2026-10-15):** the top bar's second row shows
  ALL page pills on EVERY page — Start · Compare │ Judgments · Create ·
  Run · Results · Reports │ Models · Baselines — with thin separators at
  the section boundaries. Previously only the active section's pills were
  revealed, so the menu visibly changed when moving between pages (the
  owner flagged it as inconsistent). The active section's row-1 tab and
  the current page's pill still fill; narrow screens scroll the strip.
- **Nav order (added 2026-10-15):** Judgments is the FIRST sub-element
  under the Evals section in the top nav, and the EVALS section tab
  itself links to /judgments (operator's standing preference: the
  judging kick-off leads the menu and is the section's default landing).
- **Per-model reasoning controls (added 2026-10-15):** every surface that
  starts an eval suite run offers optional reasoning_effort
  ('minimal'|'low'|'medium'|'high') and reasoning_budget (max thinking
  tokens, 1–32768) PER MODEL: the Run page (single model) and both Model
  A/B cards on Judgements (combined + mixed kick-off). Blank fields mean
  the server default — the fields are omitted from the wire body, so the
  chat request is byte-identical when unused. Values validate honestly
  (422 at every API boundary) and each side's settings apply only to
  that side's replay (B's are inert in mixed mode — B is not replayed).
- **Append-only preserved:** dispatch only ever INSERTs. Re-runs supersede.
  There is no edit/delete of stored data anywhere in the UI or API.
- **Power provenance (ADR 002):** `benchmarks.power_host` /
  `power_gpu_index` stored alongside every watts value; a watts number
  without host is rejected at the API boundary (422). Power is meaningful
  only on dedicated inference hosts — proxy-fronted models store `null`.

**Acceptance criteria**
- [x] All entities (runs, benchmarks, eval suites/results, judgments) creatable via the dashboard.
- [x] Bench and eval runs kick off from the UI against live servers; results land in the store and appear in existing views.
- [x] Dispatch validation: bad target/context/suite shapes rejected with actionable 422 detail.
- [x] Suite existence checked before replay; eval batches commit per item — a mid-replay failure keeps completed items and stamps the run `partial` (v0.2 partial preservation; pre-v0.2 it was all-or-nothing).
- [x] Power numbers always carry host + GPU index provenance.
- [x] Health preflight before dispatch (J6).
- [x] First real benchmark captured via /create dispatch (openai transport, run 9 vs demo host B tabbyAPI GLM-5.3-Flash).
- [x] Async bench jobs with live progress (bar + status + ETA + zoom) on /create.
- [x] Runs can be soft-hidden from list/compare views and unhidden, without deleting data.
- [x] Judgements kick-off: model B can be a stored baseline run (mixed `target_a` + `eval_run_b`), with the suite's recent runs listed newest first; recent judgment runs surface as one-line cards linking to the SxS pair view.
- [x] Sticky inputs: every text box restores the operator's last value on page load (localStorage, shadowing shipped defaults; config remains the server fallback).
- [x] Judgment details: item names expand the case's prompt inline (definition-snapshot sourced, multi-turn aware, images rendered via the snapshot-guarded suite-image endpoint), with both models' answers in the SxS cards below.
- [x] Per-model reasoning: optional reasoning_effort + reasoning_budget on the Run page and on both Judgements model cards, validated at the API boundary, passed through to each side's replay only, and omitted from the wire body when unset.

## 4. Non-goals (v1)
- Distributed multi-node orchestration; multi-tenant UI; managed cloud.
- Training/fine-tuning tooling.

## 5. Constraints & decisions
- Local-first: SQLite storage; no cloud dependency except external judge API.
- Python 3.10+ (the deploy host has 3.12.3). UI = local web dashboard bound
  `0.0.0.0:24816`; server-side Python (stdlib or FastAPI+uvicorn), JSON API,
  client-side charts (Chart.js).
- Benchmarking targets: OpenAI-compatible endpoints on the LAN (llama.cpp,
  vLLM). Engine-specific metrics extracted where the engine exposes them
  (llama.cpp inline `timings`, vLLM `/metrics` spec-decode counters).
- Cold-cache methodology: fresh UUID prompt prefix per run; server restart
  between context-size tests where prefix caching cannot otherwise be
  disabled. See `docs/decisions/`.

## 6. Milestones
1. **M1 Foundation** — repo, PRD, decisions. *(done)*
2. **M2 UX design** — journey walkthroughs, CLI surfaces, view specs. *(done)*
3. **M3 QA plan** — test strategy, fixture suites, golden datasets. *(done)*
4. **M4 Bench core** — J1 implemented and validated against a live server. *(done)*
5. **M5 Comparison** — J2 views on real stored data. *(done)*
6. **M6 Evals** — J3 runner + storage; v0.2: dimensions, references, vision, async jobs.
7. **M7 Judge** — J4 blind A/B; v0.2: position swap, rubrics, in-process jobs.
8. **M8 Reports** — J5 aggregation + exports; v0.2: dimension lens, Wilson CIs, generations.
9. **M9 Hardening** — supporting features, docs, release tag. *(bench side done; eval side tracks M6-M8)*
