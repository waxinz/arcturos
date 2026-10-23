# STATUS.md — Arcturos

**Progress tracking lives here now** (owner call 2026-10-11: AGENTS.md
stays the conventions/ground-truth file; this file is the living status).
Newest first. Read AGENTS.md first for conventions; read this file to
resume work.

## 2026-10-23 — Judge presentation trims serving-stack wrappers (evals-101 #8v#9 audit)

- **The judge now sees answers, not serving-stack wrappers.** Auditing
  the run #8 (qwen3.8-27b) vs run #9 (Qwen3.8-Flash-Next:IQ2_XS)
  judgments (236 rows, 100 items) found the qwen leading blank-line
  preamble stored verbatim on 222/236 run-8 rows and 0 run-9 rows — and
  the judge citing "stray blank lines" as the deciding reason:
  **27 judgments chose a winner on pairs whose two answers are
  byte-identical after trimming** (21 items: extraction ×8,
  instruction-following ×5, conciseness ×4, accuracy ×3, reasoning ×3,
  vision ×2, coding ×1, tool-calling ×1), every one for the clean side.
  Owner's example: e101-vision-003 judgment #480 — both answers '10:10';
  the preamble side lost on "adds stray blank lines".
- **Fix:** `judging.select_pairs` normalizes judge-visible text at the
  worklist boundary (`_judge_view`: strip leading/trailing whitespace on
  out_a/out_b and multiturn history); the store stays verbatim
  (append-only untouched) — same doctrine as the reference scorer
  (normalize where the value is consumed). No template change →
  `JUDGE_TEMPLATE_VERSION` stays `blind-pair-v3`. Existing #8v#9
  judgments are historical rows and keep their verdicts (append-only;
  re-judging the pair with the fix in place is the operator's call —
  it costs judge tokens).
- Tests: `test_judging.py::test_select_pairs_trims_serving_preamble`
  (worklist trimmed, store raw) + qa/test-plan J4 row.

## 2026-10-15 — Pair page leads with a per-dimension category summary

- **Column order follows the kick-off's A/B run naming:** the pair page
  resolves each URL fingerprint to its newest `A:`/`B:`-prefixed run on
  the suite and displays A on the LEFT — even when the URL's
  model_a/model_b order says otherwise (URL order stays the fallback
  for unnamed runs). Applied to the summary table columns, the meta
  win counts, and the per-item SxS cards.

- **Judge template v3 — verdicts always carry a reason:** the v2 prompt
  said "EXACTLY one line … nothing else", so the first real run (kimi-k3,
  236 judgments) stored 91% bare verdicts. v3 demands `VERDICT: X` +
  `REASON: <one short sentence>`; verdict-only replies get exactly one
  `_reason_followup` nudge (failure → rationale NULL, judgment stays
  valid); `extract_rationale` no longer counts a bare verdict as a
  rationale. `JUDGE_TEMPLATE_VERSION` → `blind-pair-v3` (generation
  scoping distinguishes old rows). Tests: 6 added
  (`test_eval_jobs.py` ×5, `test_judging.py` extensions ×2 updated).
- **Truncation guard on verdict parsing (first-run audit finding):**
  kimi's reasoning chatter overran the judge `max_tokens=1024` on ~9% of
  the first real run's judgments — those replies were cut mid-
  deliberation, never reached a VERDICT line, and the bare-token
  fallback stored a verdict the judge never concluded (safety-101#t3:
  both stored winners contradict the judges' own visible reasoning).
  Fix: `judge_once` default max_tokens 1024 → 2048, and
  `extract_verdict` now accepts the final-mention bare token ONLY when
  it ends the reply — a truncated reply returns None and routes to the
  retry instead of storing a stray-token winner.
- **Deterministic scorer strips markdown emphasis (first-run audit
  finding):** reasoning-005 — GLM answered '**Monopoly**' (correct) but
  the one_of check accepted only bare 'monopoly'; `_normalize` now
  strips `**`/`__`/backticks before exact/one_of comparison (report-time
  rescoring applies retroactively; the explained-answer fail itself is
  honest — the item demands one or two words).
- **First real judged run audited (2026-10-15, kimi-k3, 236 judgments):**
  coverage exact (118 pairs × 2 orders, no duplicates/missing); judge
  verdicts substantively correct on ~97.5% of items (all CoT-leak,
  think-tag, and safety cave-in failures punished; 0 reference-decidable
  errors across 48 ref-carrying items). Substantive misses (6/236
  judgments): reasoning-005 (judge twice preferred GLM's explained
  answer over qwen's exactly-compliant bare answer against the item's
  'one or two words' constraint), reasoning-002 + coding-002 (one order
  each picked qwen's truncated no-answer CoT), safety-101#t3 (both
  orders truncated mid-deliberation → stray-token verdicts; fixed in
  f20dfc9). Order disagreement ~27-31% is pick-vs-tie noise on equal
  outputs; zero reference-decidable flips. Known data caveats:
  presentation order not persisted (auditability gap), confidence null
  (kimi omits it), 400-char rationale tails head-truncated.
- **Pair SxS gets an Arena-style score card:** new
  `GET /api/judgments/pair/summary` (`reports.pair_summary`) aggregates
  one suite × model pair per dimension — score = wins + ½ ties 0–100,
  win/tie %, n, Wilson 95% CI on A's win share, rows sorted n desc —
  with winner orientation normalized to the caller's model_a/model_b
  order and honest empties for unknown pairs; judgment-detail.html
  renders it above the meta strip (table ≥721px, tile grid ≤720px,
  zero page overflow at 390px, graceful hide when the summary is
  empty/404) while the per-item SxS list stays unchanged below. Tests:
  9 added (`test_reports.py` ×6, `test_evalviews.py` ×3) + pair-page
  contract markers.

## 2026-10-15 — Combined kick-off: Judgements page runs evals + judging

- **One kick-off surface:** `POST /api/ops/judge/jobs` accepts
  `target_a`/`target_b` ({url, model, api_key?}) plus optional
  `eval_max_tokens/eval_temperature/eval_concurrency/eval_retries`. With
  both targets the job becomes staged: replay the suite's LATEST
  definition snapshot on model A, then model B, then blind-judge the two
  fresh runs run-scoped (same pairing semantics as the Results-page
  flow). Job record gains `combined` + `stage` (replay-a → replay-b →
  judging); one AbortFlag flows through all three stages (Stop keeps
  completed work). API keys never land on the job record (`targets`
  echoes url+model only). A replay that stores zero results hard-stops
  the job as an error before judging — honest nulls, no fabricated
  pairs. dispatch_eval quirk: its `stored` field is a row LIST, not an
  int — zero-checks must handle both shapes.
- **UI (judgments.html):** Model A / Model B fieldsets (endpoint URL,
  optional API key, model name) in a two-column `ab-grid` that collapses
  to one column ≤640px; the kick button relabels to "▶ Run evals +
  judging" the moment an A/B URL is filled; stage chip (1/3 replay A ·
  2/3 replay B · 3/3 judging) on the progress card; done message links
  to Results with the two run ids. Stored-results judging (names only)
  unchanged; guard-warning + override unchanged.
- **Docs/tests:** PRD J4 v0.3 amendment + acceptance criterion;
  qa/test-plan.md J4 combined-kick-off section; `tests/test_eval_jobs.py`
  +3 (end-to-end with fakes, validation 422 matrix, honest replay
  failure); `test_evalviews` judgments contract pins the A/B cards +
  stage chip. Tests 317 → 320 green; static gates clean (13 pages).
- Sanitize: AGENTS.md operator name → "the operator" (owner-approved).

## 2026-10-15 (late) — Evals 101 suite + dimension enum v1.1

- **New committed suite `qa/suites/evals-101-v1.json` (100 items):** the
  community-grounded broad set — 12 dimensions (accuracy 10, factuality
  8, reasoning 10, instruction-following 9, writing 7, coding 10,
  tool-calling 8, agentic 8, conciseness 7, extraction 8, vision 7,
  safety 8), 13 multi-turn items, 7 image items (6 committed assets:
  4 synthetic reference-backed — chart March=42/Oct=78 max, clock 10:10,
  7-red-square grid, MIND THE GAP sign — plus Eiffel Tower and Great
  Wave public-domain photos), 44 deterministic-reference items / 56
  judge-scored. Every numeric reference value is computed and asserted
  in the builder script; all regex references compile; images verified
  visually before commit. Dimension mix grounded in the eval-landscape
  survey (MT-Bench categories, BFCL-style tool calling, Arena-style
  conciseness, chart-QA synthetic vision).
- **Dimension enum v1.1 (7 → 13):** added `accuracy`, `factuality`,
  `tool-calling`, `agentic`, `conciseness`, `extraction` to
  `evals.DIMENSIONS`, each with a judge rubric in `judging.RUBRICS`;
  `knowledge` stays for arcturos-core-v1 back-compat. ADR 007 amended;
  `tests/test_suites.py` (new) validates every committed suite with the
  real validator and pins the Evals 101 contract (also asserts committed
  suites use repo assets, not remote URLs — no asset rot, no remote
  fetch flakiness).
- **Sanitize sweep (public mirror):** operator name removed from
  PRD.md, ux-design.md, ADR 003/007 headers, and the smoke-reasoning
  item content (already-published copies remain in older commits —
  history left untouched per doctrine). AGENTS.md still carries the
  name; it is approval-gated and flagged to the owner.
- 317 tests green.

## 2026-10-15 (evening) — Results page flow, nav as a module, judge abort fix

- **Results page flow (J5):** new `/evals` Results page — mobile-friendly
  runs table (same pattern as benchmark runs): run ID → run detail
  (`/evals/runs/{id}`), multi-select → Judgements button → run-scoped
  judge job (`POST /api/ops/judge/jobs` now accepts `eval_run_ids`;
  selection + guard run synchronously, scoping filters results to the
  selected replay runs). When judgments exist for a pair, the page links
  to `/judgments/pair` — the SxS diff view (blind rows in canonical a→b
  orientation; the a/b winner flip across the two stored orders is the
  position-bias signal). Run detail and pair detail carry the shared
  shell (data-page `evalrundetail` / `jpair`).
- **Nav is ONE module now (drift-class kill):** `static/partials/nav.html`
  is the single menu source, written by `tools/build_nav.py` from
  NAV_MODEL; pages carry only `<!--NAV-->`; `pagelib.render_page()`
  injects it server-side at request time and every page route serves
  through `page_response()`. A generator regen can no longer silently
  delete a pill (that shipped earlier today). Direct `/static/*.html`
  fetches bypass injection — routes are the canonical entry.
- **Judge-job Stop fixed (500):** `abort_judge_job` called
  `flag.set()` — no such method (AbortFlag exposes `request()`) — so
  POST `/api/ops/judge/jobs/{id}/abort` 500'd since the v0.2 refactor.
  Also: a Stop landing mid-item now breaks between the two order calls
  (bench precedent: check inside the loop) — the paid-for order-0
  judgment is preserved and the job still reports `aborted`.
- Create page is suite-create + items only (replay moved to `/evals/run`
  earlier today); nav row-2 gains the Run pill.
- **Judge-not-contestant guard is a WARNING with an override, not a
  dead-end error** (operator call after hitting it in dev testing): the
  422's `allow_contestant_judge` detail renders as an amber warn card on
  BOTH judge kick-off surfaces (Results `/evals` + `/judgments`), with a
  "Judge anyway (override guard)" button that re-dispatches with
  `allow_contestant_judge=true`. API contract unchanged (guard still
  422s synchronously; the override is the documented operator path).
- Tests: 314 green (new `tests/test_eval_runs_api.py`: runs list/detail,
  run-scoped judging, SxS order-independence, abort regression,
  drill-down shell identity); static gates clean; nav generator
  idempotent (placeholder contract); override flow verified with
  headless Chromium route mocks (guard → warn card → override POST).

## 2026-10-15 — Eval system v0.2 (dimensions, position swap, reports)

The eval side now answers the driving question — **"which model is
better A or B, on which axes"** — end to end. ADR 007 records the
methodology; PRD is v0.2.

- **J3 replay v0.2:** items carry `dimension` (7-value enum: reasoning,
  knowledge, instruction-following, writing, coding, vision, safety),
  freeform `category`, optional `system`, optional `image` (http(s) URL
  or suite-assets-relative path), optional deterministic `reference`.
  Replay defaults max_tokens 1024 (512 shipped truncated answers on
  reasoning models) / temperature 0 / retries 0 / concurrency ≤8;
  per-item commits keep completed items on failure (partial
  preservation — `eval_runs.status='partial'`, trigger whitelists name +
  status). `evals.py` (new) owns validation + single-turn replay +
  reference checkers (exact/numeric/contains/one_of/regex).
- **J4 judging v0.2:** in-process judge runner (`judging.py`, new) with
  `blind-pair-v2` template: blind anonymized responses, dimension
  rubrics, **position-swap double judging** — every item judged in BOTH
  presentation orders (the swap changes what the judge sees, not just
  bookkeeping), both raw rows stored append-only; an order-biased judge
  nets to a tie in aggregate. Judge-not-contestant enforced
  synchronously at the dispatch boundary (normalized fingerprints;
  substring match trips). Retry policy for invalid verdicts; per-item
  commit; cooperative stop. `POST /api/ops/judge/jobs` + status +
  abort. Multi-turn judged per assistant turn with prior-turn history
  serialized (MT-Bench lesson).
- **J5 reports v0.2:** per-dimension breakdown is the headline lens;
  Wilson 95% score intervals + n on every rate (n<20 tinted in the UI);
  deterministic reference pass-rates scored at REPORT time from the
  newest definition snapshot (retroactive, stored rows untouched);
  judge-generation scoping (`judge_model` + `judge_template` params,
  UI selects). `by_dimension`/`reference_scores`/`generations` in the
  report payload; CSV gains a dimension section.
- **UI:** replay moved off /create onto **/evals/run** (async job page:
  live per-item progress, ETA, operator stop, partial-run links); the
  Judgments page hosts the judge kick-off card with the same job
  panel; Reports gains the dimension lens, reference card, and
  generation filters. Nav: Evals section = Create · **Run** · Results ·
  Judgments · Reports.
- **First suite:** `qa/suites/arcturos-core-v1.json` — 49 items across
  all 7 dimensions (44 single-turn, 5 multi-turn; 33 with deterministic
  references; 5 vision items with generated images under
  `qa/suites/images/`). Research notes in `docs/research/eval-landscape.md`.
- Tests: **307 passed** (250 pre-v0.2 baseline + 57 new: evals,
  judging, reports v0.2, async job endpoints). Two real bugs the new
  tests caught and fixed: pair filtering died when 3+ models had
  results (two-model gate ran before the operator's pair filter), and
  the position-swap randomization shuffled bookkeeping instead of the
  actual presentation order.
- **Live smoke (real models, 2026-10-15):** 12-item subset of
  arcturos-core-v1 (incl. one multi-turn + two vision items) replayed
  against qwen3.8-27b and GLM-5.3-Flash-uncensored via the litellm
  proxy — 13 result rows per model, ZERO item errors (vision content
  parts passed through end to end); judged by GLM-5.3-Flash with the
  `allow_contestant_judge` override (family-name substring collision)
  — 26/26 judgments stored, 0 invalid verdicts, position swap and
  retry policy exercised for real. Result: qwen took
  instruction-following 75% and writing 100%; 16/26 ties; reference
  pass-rates (judge-free): qwen 11/11, uncensored GLM 10/11. The
  smoke caught a third real bug: multi-turn `#t2` rows fell into the
  `uncategorized` dimension bucket (metadata map lacked suffix keys) —
  fixed with inheritance + regression test.
- **Live smoke #2 — same weights, temp 0 vs temp 1 (operator-requested):**
  GLM-5.3-Flash alias @ temp 0 vs GLM-5.3-Flash-uncensored alias
  @ temp 1 (identical weights, sampling is the only variable), judged
  by family-neutral qwen3.8-27b — 26/26 replays, 26/26 judgments,
  0 invalid. Result: 22/26 ties (expected for same weights);
  reasoning leaned temp0 2-0, writing leaned temp1 2-0 (n too small to
  matter). The judge-free deterministic checks show the real signal:
  temp0 passed 10/11 references (90.9%) vs temp1 9/11 (81.8%) —
  temp 1 dropped one exact-match check. Both smokes ran through the
  sync dispatch path (dispatch_eval → dispatch_judge → suite_report);
  the async job pages (/evals/run, /judgments) are API-tested but not
  yet exercised against live models.
- **Mobile round (same day):** the top bar was broken at phone widths —
  theme.css's legacy `nav { flex-wrap: wrap }` stacked the section tabs
  into a column and page-level overrides (`.topnav-tabs { order: 3;
  width: 100% }` on two pages) re-stacked it per active section, so the
  theme toggle jumped ~200px when switching sections. Fix: nav.css now
  owns narrow-screen behavior (`flex-wrap: nowrap` inside the bar, the
  tabs strip scrolls instead of stacking, `.sidebar-foot` pins the
  toggle right); page-level overrides removed. The /judgments main
  table also overflowed 165px at 390px — wrapped in the standard
  `.table-scroll`. Verified with headless Chromium at 390×844:
  toggle at identical x on every section, zero horizontal overflow on
  /evals/run, /judgments, /reports, /evals, /compare. Suite pickers on
  /evals/run and /judgments list suites most-recent-first.

## 2026-10-11 (evening) — Nav renames, Run Diff integrated, rail polish

Owner round: "Kick Off → Start, Runs → Compare, Run Diff inside
Compare, nicer menu."

- **Renames:** Kick off → **Start** (rail + page h2 + titles on
  /kickoff and /create), Runs → **Compare** (rail + h2 + title),
  Suite create → **Create**. Page ids, routes, and file names are
  unchanged — only labels moved.
- **Run Diff retired as a page:** with exactly 2 runs selected, the
  Compare page renders "A/B — X vs Y" under the charts (same
  run-diff payload; deltas B−A, % vs A, ±2% gray band, power never
  verdict-coloured). **Skip-blank rule (owner):** a metric missing on
  either side shows NO row — no `—` filler (power_watts case).
  `/diff` 308-redirects to `/compare`; run detail's "Diff against…"
  deep-links `/compare?runs={id}`, which pre-selects the run
  (`?runs=` support added to the page).
- **Rail polish:** brand block + hairline divider; per-row bullet
  dots (accent-filled on the current page); section heads aligned
  with links; tighter vertical rhythm; 236px rail. Desktop rail +
  mobile fold both preserved.
- Tests: 247 green (pins moved to the integrated diff + renames);
  static gates clean.

### Same-round follow-ups (owner feedback while using it)

- **Compare button into the runs toolbar** (ex-sticky bottom bar, felt
  like a random popup): the toolbar row is now Compare · ⤓ CSV ·
  Hide selected · selected-count hint. Sticky bar markup/CSS/JS
  removed; charts still auto-scroll into view after a compare.
- **A/B table fires without a baseline** (call-site bug: it sat inside
  `if (data.baseline)`).
- **A/B table units + neutral columns:** ttft_ms rows display in
  seconds (matching the chart and the "TTFT (s)" label; delta
  recomputed post-scale); `streams` joins `power_watts` as
  never-verdict-coloured (a config fact, not a score).

## 2026-10-11 (late) — Page-size dropdown, sidebar redesign → REPLACED by top nav

- **Paginator dropdown:** 30 / 50 / 100 / All, persisted in
  localStorage (`arcturos-run-page-size`); pager stays visible at one
  page so "All" is reversible; re-page keeps the same first row in
  view (guarded against the 0×Infinity=NaN path when leaving All —
  that bug shipped and was fixed within the round).
- **Navigation: left rail → two-level TOP BAR (final structure).**
  Three rounds of restyling the rail failed ("the structure is
  wrong"); the fourth round changed the container:
  row 1 = brand tile + section tabs (each a link to its section's
  first page) + theme toggle; row 2 = the active section's pages as
  pills (nav.js tags `.in-section` on the active section's pills and
  `.active` on the current page; CSS hides the rest). Content
  reclaims the rail's 260px. Page titles standardized to one size
  (1.6rem) — previously Suite create was a bare h1 (2em),
  Results/Judgments/Reports default h2 (1.5em), Start a local 1.45rem
  (owner flagged the mismatch); "Suite create" heading/title unified
  to "Create" to match the nav. Theme toggle: theme.js now BINDS the
  shell's static [data-theme-toggle] (the interim skip-injection guard
  left the visible moon unwired/dead); legacy injection only for pages
  without a static toggle. Verified live: single toggle, flips
  light↔dark both ways, persists via localStorage.
  Verified in real browser across Benchmarks/Evals/Settings pages:
  single-line rows, uniform pill heights, correct active states,
  full-width table.

## 2026-10-11 (later) — Theme toggle fix; compare columns; Start page rebuild

- **Theme toggle was dead**: the skip-injection guard stopped creating
the legacy button without binding the shell's static one. theme.js
now prefers the static `[data-theme-toggle]` and binds the handler
(legacy injection only when a page ships none). Verified live:
single toggle, flips both ways, persists.
- **Compare runs table**: Engine and Context size columns removed
(owner: surplus — fingerprint chip + run detail carry them).
- **Start page rebuilt** (owner brief): redundant "Benchmark run —
kick off" card heading deleted (menu Start + page title Start is
enough); form is a 2-column grid on desktop / 1 column ≤640px;
n_predict + streams share a line; Server URL & Model gained
`<datalist>` history from the run store (type-or-pick, freshest
first); Stop moved into the actions row; progress panel auto-opens
with a per-context stage-pill strip (pill fills when its point
lands) above the details table — glanceable on mobile where the
wide table only scrolls (research pattern: W&B sweep progress grid /
MLflow run status pills).
- **/create fork killed**: main.py now serves bench-kickoff.html for
both routes; create.html (stale copy: old headings/form) deleted.
- **build_nav.py fixed to rewrite topnav**: it only matched the legacy
`aside.sidebar`, so any post-topnav re-run would have injected a
second header on all pages; now rewrites `<header class="topnav">`
(verified: full regen produces zero diffs).
- Tests 247 green; static gates clean (10 pages — create.html gone).

## 2026-10-11 (evening) — Engine field, transport autodetect, dropdowns

- **Engine is now an operator field** (owner brief): freeform
  type-or-pick on the Start page (history from runs.engine), stored on
  the run row, shown on the Compare page in a new column before Model
  ("Model fingerprint" relabeled "Model"). Legacy runs keep their
  transport-derived engine (llama.cpp / openai).
- **Transport removed from the UI**: wire protocol autodetected —
  OpenAI chat probe first (usual case), llama.cpp /tokenize fallback
  (preflight.detect_transport; endpoints accept transport='auto';
  Check-target probes kind='auto'). API default stays 'native' for
  compat.
- **Start form**: Run name on top; one field per line (only
  n_predict + Parallel streams share a line, both dropdowns now —
  256..65535 default 256; 1-16 default 1); API key follows Server URL.
  Blank name + engine given -> 'Engine / Model' (short
  model; owner later trimmed the targets part off); no engine -> classic 'host · model · targets' (pinned).
- Gotchas hit and fixed: .actions is also .field (flex-direction:
  column by default — the buttons stacked until flex-direction: row
  was explicit); compare cell renderer was value-equality based
  (engine === host string would have misrendered) — now index-based;
  default-name pin caught the first engine-or-default attempt (the
  'Engine / …' name applies only when the OPERATOR gave an engine —
  bare API calls keep the host chain).

## 2026-10-11 (evening, follow-up) — column swap fix, per-row Hide gone

- Owner caught a shipped defect: the cells array listed engine before
  server while the header said Server, Engine — headers and rows were
  verified separately, never against each other, so every row was
  shifted one column from its header. Fixed by index-aligning cells to
  the header list (every cell carries a kind tag) and adding
  test_compare_table_column_order_matches_cells, which extracts BOTH
  literals from compare.html and asserts they correspond positionally —
  drift now fails CI. Per-row Hide button removed (redundant with the
  toolbar's bulk Hide selected); trailing empty actions header and the
  unused .linklike rule went with it.

## 2026-10-11 (later) — UX round: Runs rename, pagination, sticky bar, quiet stop

Owner asked for three more changes after using the merged page; one
commit each for the UX round and a follow-up fix; deployed and verified
in a live browser.

- **Renamed to "Runs"** (title, h2, rail label on all 11 pages via
  build_nav) — "compare is what you do FROM it, not what it is".
  Route stays /compare.
- **Client-side pagination** (25/page, Prev/Next, page-of-N counter):
  rows stay in the DOM, paging only toggles `hidden` — cross-page
  selection, select-all, hide-selected, Compare payload unaffected.
  Verified live: 87 runs → 4 pages, selection persists across pages.
- **Sticky action bar**: Compare + selected-count pinned to the viewport
  bottom, appears once ≥2 runs are selected (or a chart exists) — no
  scrolling to the page end to press Compare. Charts still auto-scroll
  into view after a compare.
- **Runs table polish**: zebra rows, hover/selected tint, fingerprint as
  a short chip with full path in the tooltip, server as host-only with
  full URL in the tooltip.
- **Stop run button quieted** (owner ask): small gray ghost button, no
  full-width red slab. Root cause of "obtrusive" was worse than styling:
  `.stop-btn { display:inline-block }` OVERRODE the `hidden` attribute
  (author CSS beats the UA `[hidden]` rule), so the button shipped
  permanently visible. Fixed with `.stop-btn[hidden] { display:none }`.
- **Same trap bit again** on the new action bar (`display:flex` +
  `hidden` → visible with "no runs selected"); caught in live-browser
  verification, fixed in a follow-up commit. Lesson recorded in the
  arcturos-dashboard skill: any author `display` rule on an element
  toggled via `hidden` needs an explicit `[hidden] { display:none }`.
- Tests: 247 green (merged-page pins updated: h2 Runs, PAGE_SIZE,
  action-bar); static gates clean.

## 2026-10-11 — Compare Runs merge + stop-run + ADR 005 (run #129 math)

All four owner items landed in one round; 247 tests green; deployed to
the lab working repo and verified live.

- **Merged Compare page (`/compare`, nav "Compare").** The Runs
   list and the Compare picker are one page: charts at the TOP, runs
   picker below (LATEST FIRST — render-time reversal of the ascending
   API), metrics card at the bottom with decode/prefill/TTFT/wall
   PRE-CHECKED. Runs-page capabilities carried over: runs CSV export,
   per-row Hide, bulk Hide-selected. `/` and `/runs` 308-redirect to
   `/compare`; `index.html` retired; run detail stays at `/runs/{id}`.
   Same-day owner renames: the page label is "Compare" (ex-"Runs",
   ex-"Compare Runs") and the dispatch page is "Start" (ex-"Kick off");
   the standalone Run Diff page was retired — its A/B table renders
   under the charts when exactly two runs are selected (rows with a
   metric missing on either side are skipped), and `/diff` 308s to
   `/compare`.
   `tools/build_nav.py` sidebar injection is now replace-based (the
   generator OWNS the rail — a NAV_MODEL change rewrites every page);
   `create.html` added to the generator's PAGES so the legacy alias
   stops drifting.
2. **Kick off page:** "✔ Check target - just one ping, Vasily" (label
   only); red **■ Stop run** button while a job is in flight —
   cooperative abort (ADR 006: `AbortFlag` checked between points and
   inside SSE decode loops; `SweepAborted` flows through the
   partial-run net; completed points stored as a partial run; job
   status `aborted`; new endpoint `POST /api/ops/bench/jobs/{id}/abort`).
3. **Run #129 math fixed (ADR 005).** `decode_tps_combined` is now the
   SUM of per-stream rates. ADR 004's shared-window formula
   (Σ tokens / (max wall − min TTFT)) credited the denominator with
   other streams' serialized prefill time — run #129 stored Σ decode
   4.22 t/s BELOW one stream's own 9.28 t/s (131k), 3.44 vs 8.67
   (196k). Combined prefill keeps the window formula (run #129 confirms
   it: Σ 3800 vs 3845 single-stream). Historical rows keep their stored
   values (append-only): ADR-004-window decode values
   (2026-10-08..2026-10-11) are prefill-contaminated lower bounds;
   pre-2026-10-08 are upper bounds (shared-prompt runs).
4. Docs: PRD J7 amended (merge, stop, ADR 005), ux-design §3 + §4.2 +
   §6 rewritten for the merged page, ADRs 005 + 006 recorded.
   Tests: 242 → 247 (merged-page pins, redirect contract, abort
   contract, partial-run-on-abort, rate-sum aggregation).

## 2026-10-08 — Run detail shell fix + ADR 004

(see AGENTS.md log for the full entries of this date and earlier)
