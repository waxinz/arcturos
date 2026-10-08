# ADR 004 — Multi-stream correctness: per-stream cold prompts + wall-window combined throughput

Date: 2026-10-08
Status: accepted

## Context

Run #103 (tabbyAPI/GLM-5.3-Flash on ruapehu, transport=openai, streams=4)
stored Σ prefill 1222 T/s against a single-stream baseline of ~614 T/s
(run #102) — a 2x "capacity" the server never had. The server journal for
that run shows what actually happened:

1. The openai transport built ONE cold prompt per point and shared it
   across all streams. The native path was safe (it calls
   `build_cold_prompt` inside `run_benchmark_point`, per invocation; and
   native sends `cache_prompt: false`), but the OpenAI API exposes no
   cache-off switch, so the shared prompt made streams 3-4 100%
   prefix-cache hits: 32k tokens "prefilled" in ~0.2 s, TTFT ≈ queue
   time (~209 s behind the server's `max_batch_size: 2` limit).
2. `_aggregate_stream_points` computed combined throughput as the SUM of
   per-stream rates. Each stream's rate is measured over its own clock,
   so queued streams' tokens land in overlapping windows — summing rates
   double-counts wall time. Even without caching, 4 queued streams would
   have summed to ~1240 T/s of "capacity" while the box computed ~640.

Both defects inflate exactly the metric the PRD defines as "true server
capacity under concurrency".

## Decision

1. **Per-stream cold prompts (ops.py openai path).** `build_cold_prompt`
   moved inside `_one_stream`: every stream gets a fresh UUID prefix over
   the same deterministic body. This preserves ADR 001's two invariants
   (cold-cache measurement; deterministic body for MTP acceptance
   comparability) and matches what the native path already did.
2. **Combined throughput = tokens / shared wall window (bench.py).**
   When every stream reports TTFT (the openai path always does):
   - `prefill_tps_combined` = Σ prompt_tokens / max(ttft_ms) — by max
     TTFT every stream's prefill has completed;
   - `decode_tps_combined` = Σ output_tokens / (max(wall) − min(ttft)) —
     the window in which generation actually overlapped.
   When TTFT is absent (native non-stream points) the old rate-sum
   behaviour is kept unchanged. For truly concurrent streams (equal
   TTFT/wall) the window math reduces algebraically to the rate sum, so
   well-behaved servers see no change.
3. **Per-stream mean rates stay as they were.** They answer "what did a
   stream experience"; combined answers "what can the server do". Both
   are stored; the queueing gap between them is itself informative.

## Rejected alternatives

- *Sum of rates with a queueing caveat in the UI.* Rejected: the number
  is wrong at rest, not just presentationally; dashboards/exports/
  compare reports all consume it.
- *Divide Σ prompt_tokens by max(wall).* Rejected: wall includes
  generation, so it would understate prefill badly at large n_predict.
- *Cap streams at the server's max_batch_size.* Rejected: Arcturos
  cannot portably discover that limit (tabbyAPI exposes no such field),
  and queueing is a legitimate thing to measure — it just must not be
  miscounted as throughput.

## Consequences

- Historical openai multi-stream rows (e.g. run #103) keep their stored
  values — the store is append-only and nothing is rewritten. Anyone
  comparing across the boundary should treat pre-ADR-004 combined values
  as upper bounds.
- Multi-stream openai benchmarks now do real cold prefill per stream, so
  wall time for a streams=N sweep grows (no free cache hits); that is
  the honest cost of measuring capacity.
- ADR 002's quiet-target rule matters more: with per-stream cold
  prompts, background traffic inflates every stream, not just one.
