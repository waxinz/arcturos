# ADR 005 — Combined decode is the per-stream rate sum; window math kept for prefill only

Date: 2026-10-11
Status: accepted (supersedes ADR 004 decision 2 for decode only)

## Context

Run #129 (Maya v1.0.34, GLM-5.3-Flash, transport=openai, streams=4, 32k-256k)
stored combined-decode values BELOW the individual streams they sum:

| ctx    | mean per-stream decode | Σ decode (ADR 004 window) |
|--------|------------------------|---------------------------|
| 32768  | 9.83 t/s               | 11.76 t/s                 |
| 65536  | 10.92 t/s              | 10.61 t/s                 |
| 131072 | 9.28 t/s               | 4.22 t/s                  |
| 196608 | 8.67 t/s               | 3.44 t/s                  |
| 256256 | 0.51 t/s               | 1.00 t/s                  |

A "combined capacity" smaller than one stream's own rate is impossible.

Root cause: ADR 004 defined combined decode as
`Σ output_tokens / (max(wall) − min(ttft))`. The denominator is a shared
wall window, and under serialized prefill (256k-token prompts through a
batched server) the later streams' PREFILL time falls inside that window
while no decode is running. Each stream's own decode rate is measured over
its generation span `[ttft_i, wall_i]`, so the window number mixes two
different clocks and craters whenever prefill dominates wall time —
exactly the regime multi-stream large-context benchmarks live in.

Combined PREFILL stays on ADR 004's window formula
(`Σ prompt_tokens / max(ttft)`): prefill genuinely serializes through the
same engine, max TTFT is the moment every stream's prefill has completed,
and run #129 confirms it against single-stream reality (Σ 3800 t/s vs
3845 t/s single-stream at 256k).

## Decision

`decode_tps_combined` = **sum of the per-stream decode rates** — the same
math pre-ADR-004 used, now with per-stream cold prompts (ADR 004
decision 1) so the rates being summed are honest. Each stream's rate is
already measured over its own generation span; the sum is the server's
decode capacity under concurrency. For truly concurrent streams this is
algebraically identical to the window formula; for serialized prefill it
is the only number that cannot come out below an individual stream.

When a stream reports no decode rate (None), it is skipped by the existing
null-skipping sum — a stream that measured nothing contributes nothing.

## Rejected alternatives

- *Keep the window formula and explain it in the UI.* Rejected: run #129
  shows the output is not a defensible "different metric" — it is a wrong
  number at rest, consumed by charts, exports, and diffs.
- *Union of per-stream windows: Σ tokens / |⋃ [ttft_i, wall_i]|.* Rejected
  as over-engineered: with equal spans it reduces to the rate sum, with
  unequal spans the rate sum is already the per-stream-honest capacity,
  and the union variant still divides one number by a clock none of the
  rates were measured on.
- *Drop the combined metric entirely.* Rejected: under genuinely
  concurrent serving the sum IS the meaningful capacity number and the
  queueing gap vs the mean stays informative.

## Consequences

- New multi-stream rows carry rate-sum combined decode; the impossible
  "combined < individual" rows cannot be produced any more.
- Historical rows (runs #103, #129 and every other ADR-004 openai
  multi-stream row) keep their stored values — the store is append-only
  and nothing is rewritten. Combined-decode values from the ADR 004 window
  (2026-10-08..2026-10-11) are LOWER BOUNDS contaminated by serialized
  prefill; treat them as meaningless for capacity comparison. Pre-ADR-004
  rows (rate sum over SHARED prompts) keep their documented upper-bound
  caveat.
- The queueing gap now shows where ADR 004 said it would: in the gap
  between per-stream mean rates and the combined sum, not in a deflated
  combined number.
