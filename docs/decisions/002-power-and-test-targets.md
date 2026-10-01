# ADR 002 — Power draw scope and test targets

Date: 2026-09-17
Status: accepted

## Context
Power sampling (nvidia-smi over SSH) is meaningful ONLY on dedicated
inference hosts where the GPU serves the benchmarked model exclusively:
- demo host A (DeepSeek-V4-Flash, 6x3090)
- demo host B (GLM-5.3-Flash, 6x3090)
The qwen rotation lives behind a litellm PROXY on taupo; the actual serving
hosts are elsewhere and taupo's own RTX 3080 is unrelated to that workload.
Sampling taupo's GPU during a qwen rotation bench measures the wrong box.

Also: never benchmark demo host A/demo host B while other work may be queued —
with --parallel 1 on llama.cpp servers, probe traffic behind unrelated jobs
produces wildly inflated TTFT (observed 32s-254s vs true 1.5-9s).

## Decision
1. power.py records (host, gpu_index) per run; dashboard stores power_watts
   only when the target is a dedicated inference host. For proxy-fronted
   models (qwen rotation), power_watts stays None with reason 'proxy'.
2. Bench runs against shared/production servers must set an explicit
   'quiet_target' acknowledgment; validation runs use idle hosts only.
3. TTFT for reasoning models = first chunk of ANY payload (reasoning_content
   counts); decode rate counts content chunks only (client-side), so
   reasoning-token overhead is visible as the TTFT->first-content gap.

## Consequences
- Honest metrics per host class; no fake power numbers.
- The observed TTFT instability on demo host A was self-inflicted queueing;
  methodology doc updated to require idle-target checks before benching.
