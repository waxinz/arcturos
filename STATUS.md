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

## 2026-10-11 (late) — Page-size dropdown, full sidebar redesign

- **Paginator dropdown:** 30 / 50 / 100 / All, persisted in
  localStorage (`arcturos-run-page-size`); pager stays visible at one
  page so "All" is reversible; re-page keeps the same first row in
  view (guarded against the 0×Infinity=NaN path when leaving All —
  that bug shipped and was fixed within the round).
- **Sidebar full redesign (owner: third strike, "needs a full complete
  rethink"):** rail is now its own 260px surface (`--sidebar-bg`,
  light + dark), 1px separator to content; brand = monogram tile +
  name + tagline row; section labels are plain small caps — no dots,
  no bars; link rows are inline-SVG stroke icons + text, uniform 32px
  boxes; active page = filled pill with 3px accent edge (extra left
  padding so the icon gutter stays clear); footer pinned above a
  hairline. Root cause of the old overlap is now guarded in CSS:
  theme.css's legacy `nav { display:flex; flex-wrap:wrap }` out-cascades
  a plain `display:block`, so the sidebar nav is `display:block
  !important` with a comment forbidding bare-nav rail selectors.
  Markup from tools/build_nav.py (NAV_ICONS table), regenerated all
  11 pages. Verified in real browser, light + dark: no overlap, even
  rhythm, clean active state.

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
