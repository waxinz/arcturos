# Arcturos — PRD v0.1

Local-first LLM benchmarking & evaluation suite.
Owner: Alexei. Status: draft for review.

---

## 1. Vision

A repeatable, local-first tool that measures **how fast** an LLM serves tokens
and **how well** it answers prompts, stores every measurement, and makes
comparisons across models and runs effortless.

## 2. Personas

- **P1 — Benchmark operator (Alexei):** technical, wants cold-cache numbers,
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
outputs + performance metrics, single-turn and multi-turn.

- Eval suites are versioned collections of prompts (JSON/YAML), each item:
  prompt, optional system prompt, expected behavior/category tags, and for
  multi-turn: ordered message list with assistant turn placeholders.
- The runner replays suites against a target model (any OpenAI-compatible
  endpoint), records every output, token counts, latency, and engine timings
  alongside, tagged by suite version and model snapshot.
- Concurrency and retry policy configurable; failures recorded not swallowed.

**Acceptance criteria**
- [ ] Suite definition format supports single-turn and multi-turn items.
- [ ] One command runs a suite against a model; all outputs + timings stored.
- [ ] Re-run produces a new result set; results traceable to suite version.

### J4 — Blind A/B preference scoring with an external judge
**Actor:** P2 · **Goal:** have an external LLM rater blindly prefer between
two models' answers for the same eval prompt.

- For each eval item where models A and B both produced outputs, the tool
  submits a **blind** pairwise judgment to a judge model (external LLM via
  API): order randomized, model identities anonymized (Response 1 / Response
  2), judge prompt template versioned and stored with results.
- Judge output (preference + optional confidence/rationale) is parsed,
  validated, and stored; ties supported.
- Operator can re-judge with a different template/judge model; all judgments
  retained (no in-place edits).

**Acceptance criteria**
- [ ] Blind randomized pairwise judging; identity leakage impossible from the stored judgment payload.
- [ ] Judgment records link eval item + both model result IDs + judge template version + judge model.
- [ ] Invalid judge outputs are detected, logged, and retried per policy.

### J5 — Eval performance reports
**Actor:** P2 · **Goal:** report how eval suites performed on models, e.g.
"two models answered 100 prompts in suite X; judge preferred A 62% / B 34% /
tie 4%".

- Aggregate views: win-rate matrix per suite × model pair, per-category
  breakdown, item-level drill-down to the raw outputs and judgments.
- Reports exportable (JSON/CSV/HTML).

**Acceptance criteria**
- [ ] Suite-level A/B win-rate summary with counts and percentages.
- [ ] Category-level breakdown; drill-down to item level.
- [ ] Export to at least one portable format (CSV or JSON).

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

- **Create & Run view (`/create`)** with six cards:
  1. **Bench dispatch** — pick server URL, context targets, n_predict,
     optional API key; the API replays a real cold-cache sweep (tokenize →
     UUID-prefixed prompts → native `/completion`), stores run + benchmark
     points, returns metrics.
  2. **Eval suite create** — name + version; the API echoes the new suite id
     straight into the replay form.
  3. **Suite item editor** — JSON textarea, client-side validated
     (non-empty items, ids present, prompts present; multi-turn shape checked).
  4. **Eval replay dispatch** — suite id + OpenAI-compatible target + model
     fingerprint + optional API key (password field, never stored);
     single- and multi-turn items replay and
     store eval_results.
  5. **Manual benchmark point** — add externally-measured rows to an existing
     run; power provenance (host + GPU index) required with any watts value.
  6. **Judgment record** — blind A/B verdict between two stored results.
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
  exports, and the live job detail table.
- **Run visibility — soft hide (added 2026-09-21):** `PUT
  /api/runs/{id}/visibility` `{hidden: true|false, reason?}` appends a flag
  row (append-only, newest wins). Hidden runs drop out of `/api/runs` and
  are rejected by compare/diff with an actionable 404, but stay directly
  addressable and can be unhidden at any time. Nothing is ever deleted.
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
- [x] Suite existence checked before replay; eval batch is transactional (failure → nothing stored).
- [x] Power numbers always carry host + GPU index provenance.
- [x] Health preflight before dispatch (J6).
- [x] First real benchmark captured via /create dispatch (openai transport, run 9 vs pakuranga tabbyAPI GLM-5.3-Flash).
- [x] Async bench jobs with live progress (bar + status + ETA + zoom) on /create.
- [x] Runs can be soft-hidden from list/compare views and unhidden, without deleting data.

## 4. Non-goals (v1)
- Distributed multi-node orchestration; multi-tenant UI; managed cloud.
- Training/fine-tuning tooling.

## 5. Constraints & decisions
- Local-first: SQLite storage; no cloud dependency except external judge API.
- Python 3.10+ (taupo has 3.12.3). UI = local web dashboard bound
  `0.0.0.0:24816`; server-side Python (stdlib or FastAPI+uvicorn), JSON API,
  client-side charts (Chart.js).
- Benchmarking targets: OpenAI-compatible endpoints on the LAN (llama.cpp,
  vLLM). Engine-specific metrics extracted where the engine exposes them
  (llama.cpp inline `timings`, vLLM `/metrics` spec-decode counters).
- Cold-cache methodology: fresh UUID prompt prefix per run; server restart
  between context-size tests where prefix caching cannot otherwise be
  disabled. See `docs/decisions/`.

## 6. Milestones
1. **M1 Foundation** — repo, PRD, decisions. *(this phase)*
2. **M2 UX design** — journey walkthroughs, CLI surfaces, view specs.
3. **M3 QA plan** — test strategy, fixture suites, golden datasets.
4. **M4 Bench core** — J1 implemented and validated against a live server.
5. **M5 Comparison** — J2 views on real stored data.
6. **M6 Evals** — J3 runner + storage.
7. **M7 Judge** — J4 blind A/B.
8. **M8 Reports** — J5 aggregation + exports.
9. **M9 Hardening** — supporting features, docs, release tag.
