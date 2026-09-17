# 001 — J1 Bench Methodology

- **Title:** J1 bench methodology — tokenize-sized cold prompts, native endpoint,
  optional power sampling
- **Date:** 2026-09-17
- **Status:** Accepted

## Context

J1 (PRD §J1) requires repeatable performance numbers — prefill tok/s, decode
tok/s, TTFT, wall time, output tokens, MTP acceptance, power draw — at several
context lengths against LAN inference servers (llama.cpp, vLLM). Two problems
dominate the design:

1. **Character estimation is wrong.** A 64k-character prompt tokenized to
   ~85k tokens in practice on a real model. Prompt sizing must come from the
   server's own tokenizer, never from a character-count heuristic.
2. **Prefix/KV caching contaminates measurements.** A prompt that reuses a
   cached prefix runs prefill at near-zero cost and fakes near-zero TTFT. Every
   measured point must be cold-cache.

## Decision

### 1. Prompt sizing via `/tokenize`, not character counts

`plan_prompt_sizes` POSTs 50 copies of a deterministic `unit_text` to the
server's `POST /tokenize` endpoint (`{"content": unit_text * 50}`) and derives
`tokens_per_section = len(tokens) / 50`. Each target then gets
`n_sections = int((target - margin) / tokens_per_section)` with a default
`margin = 16` tokens that absorbs the cold-prefix overhead (UUID prefix ~2-3
tokens + newline). One sizing call serves all targets — the input is identical
for every target, so per-target tokenization would be pure redundancy.

### 2. Cold-cache measurement via fresh UUID prefix; no warmup runs

`build_cold_prompt` prepends a fresh `uuid.uuid4()` string + newline to the
deterministic body. The unique prefix guarantees no run is served from a
prefix/KV cache. **No warmup runs are performed**: a warmup run would populate
the prefix cache and the measured point would be served warm, faking near-zero
TTFT. The deterministic body is what keeps MTP draft acceptance meaningful —
drafts are generated against the same predictable text on every point.

### 3. Native `/completion` endpoint, not `/v1/chat/completions`

Points are POSTed to `{base_url}/completion` with
`{"prompt": ..., "n_predict": ..., "stream": False, "cache_prompt": False}`.
The native llama.cpp endpoint is the only route that exposes
`draft_n` / `draft_n_accepted` inline in `timings`, which is what J1 needs for
MTP acceptance. The OpenAI-compatible `/v1/chat/completions` route does not.

### 4. Non-stream first; TTFT deferred to a streaming variant

The v1 point runner is non-streaming: `ttft_ms` is `None` because the native
non-stream response does not expose a time-to-first-token. TTFT requires a
streaming variant that times the first chunk — tracked as a follow-up.

### 5. `nvidia-smi` power sampling is optional

`power_draw_avg` samples
`nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounlets -i <gpu>`
every 1s for the requested duration from a daemon thread and returns the mean
W. If `nvidia-smi` is missing, the GPU index is invalid, or any sample fails,
it returns `None` — the bench must not require a local GPU.

### 6. Storage reuses the existing append-only schema

`store_benchmark_run` inserts a `runs` row (model fingerprint = `/props` model
path or `"unknown"`, engine `"llama.cpp"`, context size = `/props` n_ctx) plus
one `benchmarks` row per point. Because `runs.context_size` is NOT NULL in the
existing schema, an unknown n_ctx is stored as `0` — a documented sentinel for
"unknown" rather than a fabricated value. No DDL is duplicated; the db.py
helpers (`connect`, `init_db`) are reused.

## Validation (2026-09-17, live smoke on ruapehu 10.10.10.122:8000)

Ran one real point (target 512, n_predict 32) against the live llama.cpp
server. Endpoint shapes matched the assumptions exactly — no code adaptation
was required: `/tokenize` returned `{"tokens": [...]}` and `/completion`
exposed `timings.prompt_per_second` / `predicted_per_second` /
`draft_n` / `draft_n_accepted` inline. Measured: prefill 368.6 tok/s, decode
40.8 tok/s, MTP 20/20 drafted/accepted, wall 3.19s, `tokens_prompt` 479.

Two real-world observations recorded:

1. The uuid4 prefix tokenizes to ~24 tokens on this tokenizer (479 total =
   455 body + 24 prefix), above the 16-token default margin. The margin
   absorbs most of it; the residual shortfall is documented sizing slack, not
   a correctness error — and the exact token count is always re-derived from
   the server's own response (`tokens_prompt`) at run time.
2. This server's `/props` returns `n_ctx: null`, so stored runs for it carry
   `context_size = 0` (the documented "unknown" sentinel) — the storage layer
   degrades gracefully instead of failing.

## Consequences

- Benchmarks are honest cold-cache numbers: prefill is always a full prefill
  from cold state, so numbers are comparable across runs and machines.
- Sizing accuracy is bounded by tokenizer granularity: at small contexts the
  `int()` rounding of sections can miss the target by more than the margin
  (e.g. a 512-token target with ~85 tokens/section lands ~440 tokens). The
  margin absorbs the UUID overhead, not the rounding error; operators should
  pick targets comfortably above the per-section granularity.
- `/completion` ties the tool to llama.cpp's native shape. vLLM targets need a
  separate adapter (spec-decode counters via `/metrics` diff per PRD §J1) —
  out of scope for this decision.
- `ttft_ms` is absent from v1 points; the streaming variant is a tracked
  follow-up.
- Runs remain append-only: re-running appends new rows via the API/storage
  layer; nothing overwrites prior measurements.
- Engine metadata is best-effort: a missing `/props` endpoint yields `None`
  fields and a stored run still works (fingerprint `"unknown"`, context size
  `0`).
