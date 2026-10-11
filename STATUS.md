# STATUS.md — Arcturos

**Progress tracking lives here now** (owner call 2026-10-11: AGENTS.md
stays the conventions/ground-truth file; this file is the living status).
Newest first. Read AGENTS.md first for conventions; read this file to
resume work.

## 2026-10-11 — Compare Runs merge + stop-run + ADR 005 (run #129 math)

All four owner items landed in one round; 247 tests green; deployed to
the lab working repo and verified live.

1. **Merged Compare page (`/compare`, nav "Compare runs").** The Runs
   list and the Compare picker are one page: charts at the TOP, runs
   picker below (LATEST FIRST — render-time reversal of the ascending
   API), metrics card at the bottom with decode/prefill/TTFT/wall
   PRE-CHECKED. Runs-page capabilities carried over: runs CSV export,
   per-row Hide, bulk Hide-selected. `/` and `/runs` 308-redirect to
   `/compare`; `index.html` retired; run detail stays at `/runs/{id}`.
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
