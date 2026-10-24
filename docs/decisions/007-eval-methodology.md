# ADR 007 — Eval system methodology: dimensions, position swap, rubrics, references, in-process jobs

**Date:** 2026-10-15
**Status:** Accepted
**Deciders:** the owner, Mantis (implementation)

## Context

The benchmarking side of Arcturos spent a month hardening a set of patterns:
async jobs with live progress and self-correcting ETAs, cooperative operator
stop (ADR 006), partial-run preservation, preflight before dispatch, honest
nulls, labels-vs-measurements (ADR 003), transport autodetect, and judge
isolation discovered ad hoc during the first smoke evals (judge-must-not-be-
a-contestant, reasoning models returning empty `content`, fingerprint
normalization across server suffixes).

The eval side (J3/J4/J5) pre-dates most of those patterns and cannot answer
the owner's primary question: **"which model is better, A or B, based on
specific dimensions and evals?"** Concretely:

1. Items carry a freeform `category` but no primary **dimension** — there is
   no axis to break "which model is better" down by.
2. Judging exists only as a CLI script (`scripts/ab_judge.py`) that
   hard-codes one suite file, judges each item once in one random order
   (unmitigated position bias), has no retry policy, and no dashboard
   surface.
3. Replay is synchronous only (no progress, no stop, no partial
   preservation), hard-codes `max_tokens=512` (reasoning models truncate),
   and silently drops the `system` field of single-turn items.
4. Reports aggregate ALL judgments ever stored — one re-judge with a
   different judge pollutes the aggregate — and there is no confidence
   measure on small-n win rates.
5. No image content support anywhere in the eval path.

Industry practice (surveyed in `docs/research/eval-landscape.md`: Chatbot
Arena, MT-Bench, AlpacaEval 2.0, Arena-Hard, IFEval, GSM8K) converged on
solutions for each of these; this ADR adopts the subset that fits a
local-first, append-only, multi-judge tool.

## Decision

### 1. Dimensions are the primary eval axis

Every eval item carries `dimension` — one of a fixed allowlist
(`reasoning`, `knowledge`, `instruction-following`, `writing`, `coding`,
`vision`, `safety`; extended 2026-10-15 for the community Evals 101 set
with `accuracy`, `factuality`, `tool-calling`, `agentic`, `conciseness`,
`extraction`) — plus the existing freeform `category` sub-tag.
Reports break win rates and pass rates down BY DIMENSION as the headline
view; categories remain the second-level drill-down. Unknown dimensions are
a validation error at suite-creation time; legacy suites without dimensions
report under `uncategorized` (honest fallback, same doctrine as
categories).

### 2. Objective where possible, judged where not

Items MAY carry a `reference` — a deterministic check spec validated at
suite creation and evaluated at REPORT time against stored outputs:

- `{"check": "exact", "value": str}` — whitespace-trimmed, case-insensitive
  equality (industry exact-match normalization).
- `{"check": "numeric", "value": number, "tolerance": number = 0}` — the
  LAST number found in the output must match within tolerance (GSM8K
  convention: final answers end in the number).
- `{"check": "contains", "value": str, "case_sensitive": bool = false}`.
- `{"check": "one_of", "values": [str, ...]}` — normalized membership.
- `{"check": "regex", "pattern": str, "flags": str = ""}` — `re.search`.

Reference items need no judge: reports show per-model pass rates per
dimension alongside judge win rates. Scoring happens at report time from
the newest definition snapshot — improving a reference spec applies
retroactively and the append-only store is never touched. Executing
model-generated code in a sandbox (Aider-style) is explicitly OUT of scope
for v1: arbitrary code execution from the dispatch path is a security
surface v1 does not accept.

### 3. Blind pairwise judging: position swap, dimension rubrics, retry

- The judge runner is IN-PROCESS (`src/arcturos/judging.py`, invoked by
  `POST /api/ops/judge/jobs`) — the same async-job pattern as bench jobs:
  synchronous validation, background thread, live per-item progress,
  self-correcting ETA, cooperative stop (ADR 006's `AbortFlag` checked
  between items), partial preservation (judgments stored as they land).
- **Every item is judged TWICE — once per presentation order — and both
  raw judgments are stored.** No derived "combined" row: reports aggregate
  rows, so an order-flipped item contributes one win to each side (a tie in
  aggregate) instead of trusting a coin flip. Per-item disagreement is
  visible as the flip count.
- The judge prompt template is versioned (`blind-pair-v2`, then
  `blind-pair-v3`) and **dimension-aware**: the item's dimension selects a
  rubric paragraph (correctness-first for reasoning/knowledge/coding;
  instruction-fidelity for instruction-following; prose quality for
  writing; visual grounding for vision; calibration for safety). Template
  version is stored on every judgment row, so reports can scope to a
  generation.
- **v3 (first real run, kimi-k3, 2026-10-15): every verdict carries a
  one-sentence reason.** v2 asked for "EXACTLY one line … nothing else",
  and the judge obeyed — 91% of the 236 real judgments stored no
  rationale at all. v3 requires `VERDICT: X` + `REASON: <one short
  sentence>`; a verdict-only reply triggers exactly one cheap follow-up
  call asking for the sentence (failure leaves rationale NULL — honest,
  never fatal), and a bare verdict line no longer counts as a rationale.
- **Truncation guard (first-run audit finding):** kimi's reasoning
  chatter overran the 1024-token judge budget on ~9% of calls; those
  replies were cut mid-deliberation and the bare-token fallback stored a
  verdict the judge never concluded (safety-101#t3: both stored winners
  contradicted the judge's own visible reasoning). Two fixes: judge
  `max_tokens` default 1024 → 2048, and the final-mention bare token is
  trusted ONLY when it ends the reply — a truncated reply returns None
  and routes to the retry instead of storing a stray-token verdict.
- Verdict parsing tolerates reasoning-model chatter: prefer a
  `VERDICT: <1|2|tie>` line, fall back to the LAST bare 1/2/tie token
  (end-anchored per the truncation guard above). Invalid verdicts retry
  per policy (default 1); after exhaustion they are COUNTED in the job
  summary, never silently dropped, and never stored as fabricated
  verdicts.
- Judge model + judge URL come from the dispatch body with config fallback
  (`[eval] judge_url`, then legacy `[demo] judge_url`). The judge must not
  be a contestant: normalized fingerprints (server suffixes stripped, the
  `ab_judge` lesson) are compared at dispatch; violation is a 422 unless
  explicitly overridden.

### 4. Multi-turn and vision

- Multi-turn items are judged **per assistant turn** (matching the
  store's `#tN` turn records) with the conversation serialized up to and
  including that turn — context-preserving (the MT-Bench lesson: judge the
  later turn WITH the earlier turns, not in isolation), per-turn
  granularity.
- Vision items carry an `image` reference (http(s) URL, or a path relative
  to the suite assets directory `qa/suites/`). Replay sends OpenAI
  `image_url` content parts (local files inlined as base64 data URIs); the
  judge receives the same image. A text-only judge failing on images is an
  honest per-item error, never a fabricated verdict.

### 5. Replay inherits the bench job pattern

`dispatch_eval` gains: `max_tokens` (default 1024), `temperature` (default
0.0), `retries` (default 0), `concurrency` (default 1, 1-8), per-item
progress callbacks, cooperative abort, and per-item commits — a mid-replay
failure keeps completed items (partial preservation, `eval_runs.status =
'partial'`; the eval_runs UPDATE trigger already permits status writes
while blocking measurement columns). Single-turn items honor `system` (the
pre-v0.2 runner dropped it — recorded as a shipped defect, fixed here).
Async variant `POST /api/ops/eval/jobs` mirrors bench jobs; the sync
endpoint stays for scripted use.

### 6. Reports: dimension lens, Wilson intervals, generations

- `suite_report` gains `by_dimension`, `reference_scores` (per model per
  dimension pass/fail), `generations` (distinct judge model + template
  versions with counts), and a `generation` filter (judge model +
  template version; default = newest by first judgment id).
- Every win-rate surface (overall, per pair, per dimension) carries its n
  and a Wilson 95% score interval — closed-form, no bootstrap needed at
  this scale; small-n cells are already UX-tinted below n=20.
- **Pair-scoped aggregation (v0.3):** `pair_summary` recomputes the same
  per-dimension math for ONE suite × model pair (Arena-style score =
  wins + ½ ties, 0–100, orientation normalized to the caller's
  model_a/model_b), powering the category summary table that leads the
  judgment-detail page; the per-item SxS rows remain below it as the
  detail view. Column order on that page follows the kick-off's
  `A:`/`B:` run naming (URL order as fallback).

## Rejected alternatives

- *Single judging pass with order randomized once (v0.1 behaviour).*
  Rejected: position bias is a known, measured judge defect (MT-Bench
  reports significant position inconsistency across strong judges); the
  swap is one extra call per item and buys net-out-in-aggregate honesty.
- *Store a derived combined verdict per item (both orders merged).*
  Rejected: append-only doctrine — derived rows are recomputable at report
  time; storing only raw judgments keeps the store auditable.
- *JSON-structured judge verdicts.* Rejected for v1: local small judges
  (the realistic judge pool on the LAN) break JSON formatting under
  reasoning chatter far more often than a one-token verdict; the strict
  `VERDICT:` line plus last-token fallback is robust to both.
- *Rubric 1-10 single-answer scoring (MT-Bench style) as the primary
  mode.* Deferred: pairwise answers the owner's A-vs-B question directly;
  absolute scores can be added later as a third judgment mode without
  schema changes (judgments.confidence already exists as a 0-1 column).
- *Sandboxed code execution for coding items.* Rejected for v1 (security
  surface); coding items are judge-scored, with `contains`-style reference
  checks available for crude signal.

## Consequences

- Suite JSON gains `dimension` (required for NEW items at creation) and
  optional `reference`/`image` keys; the validator rejects unknown
  dimensions and malformed check specs with actionable messages.
- Two judge calls per item doubles judge traffic; acceptable — judge calls
  are cheap relative to contestant replays and the job pattern surfaces
  progress either way.
- Reports aggregation logic is UNCHANGED (it already counts judgment rows);
  the dimension lens, reference scores, generations, and Wilson intervals
  are additive joins over existing tables + the definition snapshot.
- `scripts/ab_judge.py` keeps working (single-order mode, API-based) for
  scripted/CI use; the dashboard surface is the primary path for operators.
- Historical suites (smoke-reasoning v1) report under `uncategorized`
  dimension until re-judged or re-seeded; nothing is backfilled.

## Amendment (2026-10-23) — judgment forensics + flip resolution

A live audit (suite judgments #521/#522) exposed two gaps in the position-
swap design: presentation order was ephemeral (a rationale's "Response 1"
could not be mapped to a stored side after the fact — one row's rationale
provably praised the opposite side of its winner), and a decided split
(one order prefers A, the other B) left a coin flip as the per-item
verdict.

Decision, fix-forward (append-only; legacy rows unchanged, NULL fields):

1. **Persist forensics on every judgment row:** `presentation`
   (`a_first`/`b_first`), `raw_reply` (the judge's verbatim reply), and
   `role` (`order` for the two swap rows, `tiebreak` for resolution rows).
   Verdict↔rationale coherence becomes mechanically checkable.
2. **Resolve decided splits with one tiebreak call:** when the two orders
   disagree ('a' + 'b'), judge_item fires ONE extra judge call in a fresh
   random order (same retry budget) and stores it as `role='tiebreak'`.
   The existing row-level score math then resolves the pair (2–1 for the
   tiebreak side; a tie tiebreak degrades the split to a draw). Ties,
   both-bad draws, and units with an errored order never tiebreak.
   Default on; `tiebreak_flips=false` disables per job. Cost: ~1 extra
   call per flipped unit (~6% of units in the audited batch).
3. **No report-semantics change:** aggregation still counts judgment
   rows; the tiebreak row participates like any other row, so the
   scoreboard shift is confined to newly judged runs.

