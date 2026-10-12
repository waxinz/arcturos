# STATUS.md — Arcturos

**Progress tracking lives here now** (owner call 2026-10-11: AGENTS.md
stays the conventions/ground-truth file; this file is the living status).
Newest first. Read AGENTS.md first for conventions; read this file to
resume work.

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
  Blank name + engine given -> 'Engine / Model / targets' (short
  model); no engine -> classic 'host · model · targets' (pinned).
- Gotchas hit and fixed: .actions is also .field (flex-direction:
  column by default — the buttons stacked until flex-direction: row
  was explicit); compare cell renderer was value-equality based
  (engine === host string would have misrendered) — now index-based;
  default-name pin caught the first engine-or-default attempt (the
  'Engine / …' name applies only when the OPERATOR gave an engine —
  bare API calls keep the host chain).

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
