"""OpenAI-compatible bench path (litellm / vLLM / any /v1/chat/completions).

Unlike llama.cpp native /completion, these servers emit no inline timings —
all metrics are computed CLIENT-SIDE from SSE chunk arrival times, with
`stream_options: {"include_usage": true}` for authoritative token counts.

TTFT  = wall time to the first chunk of any stream payload (reasoning
models emit reasoning first — delta key `reasoning` on vLLM,
`reasoning_content` on tabbyAPI; that latency is real user-facing
latency).

Decode tps resolution — most-authoritative source wins:
  1. usage.completion_tokens_per_sec  (server-accounted generation rate)
  2. usage.completion_tokens / usage.completion_time
  3. client-side content-chunk rate, only when the content span is at
     least _MIN_CLIENT_SPAN_S. Below that the server burst-flushed the
     stream (SSE coalescing): chunk arrival measures socket drain, not
     decode (this produced a 15054 t/s artifact on run 17).

Prefill tps ≈ prompt_tokens / ttft (an estimate; flagged as such).
"""

from __future__ import annotations

import json
import time
import urllib.request as urlreq
import urllib.error as uerr

from dataclasses import dataclass

# Client-side chunk timing below this span means the server burst-flushed
# the stream (SSE coalescing): chunk arrival measures socket drain, not
# decode. Run 17 recorded 15054 t/s from a 17 ms span this way.
_MIN_CLIENT_SPAN_S = 0.25


@dataclass
class OpenAIBenchPoint:
    target_tokens: int
    model: str
    ttft_ms: float | None
    decode_tps: float | None  # resolved: server-authoritative > client
    decode_tps_client: float | None  # raw client chunk rate (content chunks / span)
    decode_tps_server: float | None  # usage.completion_tokens_per_sec
    completion_time_s: float | None  # usage.completion_time
    prefill_tps_estimate: float | None  # derived, not server-authoritative
    wall_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    stop_reason: str | None
    error: str | None = None


def _normalize_base(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def run_openai_stream_point(base_url: str, model: str, messages: list,
                            api_key: str | None = None,
                            max_tokens: int = 128,
                            timeout_s: float = 300.0) -> OpenAIBenchPoint:
    body = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlreq.Request(
        _normalize_base(base_url) + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers=headers)
    t_start = time.monotonic()
    ttft_ms = None
    first_payload_t = None
    last_payload_t = None
    decode_tps_server = None
    completion_time_s = None
    completion_tokens = None
    prompt_tokens = None
    stop_reason = None
    err = None
    reasoning_seen = False
    payload_chunk_count = 0
    try:
        with urlreq.urlopen(req, timeout=timeout_s) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    now = time.monotonic()
                    # TTFT = first chunk of ANY stream payload — reasoning
                    # models emit reasoning before content, and that latency
                    # is real user-facing latency. The reasoning delta key
                    # varies by server: `reasoning` (vLLM) vs
                    # `reasoning_content` (tabbyAPI).
                    reasoning = (delta.get("reasoning")
                                 or delta.get("reasoning_content"))
                    if ttft_ms is None and (reasoning or delta.get("content")):
                        ttft_ms = (now - t_start) * 1000.0
                    if reasoning:
                        reasoning_seen = True
                    if reasoning or delta.get("content"):
                        payload_chunk_count += 1
                        if first_payload_t is None:
                            first_payload_t = now
                        last_payload_t = now
                    fr = choices[0].get("finish_reason")
                    if fr:
                        stop_reason = fr
                usage = chunk.get("usage") or {}
                if usage:
                    completion_tokens = usage.get("completion_tokens",
                                                  completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    cps = usage.get("completion_tokens_per_sec")
                    if cps:
                        decode_tps_server = round(float(cps), 2)
                    ctime = usage.get("completion_time")
                    if ctime is not None:
                        completion_time_s = float(ctime)
    except (uerr.URLError, OSError) as e:
        return OpenAIBenchPoint(
            target_tokens=0, model=model, ttft_ms=None,
            decode_tps=None, decode_tps_client=None,
            decode_tps_server=None, completion_time_s=None,
            prefill_tps_estimate=None,
            wall_s=time.monotonic() - t_start,
            prompt_tokens=None, completion_tokens=None,
            stop_reason=None, error=str(e))
    wall_s = time.monotonic() - t_start
    decode_tps_client = None
    if (first_payload_t is not None and last_payload_t is not None
            and last_payload_t > first_payload_t):
        gen_span = last_payload_t - first_payload_t
        # The client-side payload-chunk rate is a token-rate PROXY only
        # while the server flushes roughly one chunk per token. Reject
        # degenerate spans (burst flush) — see _MIN_CLIENT_SPAN_S.
        if gen_span >= _MIN_CLIENT_SPAN_S:
            decode_tps_client = round(payload_chunk_count / gen_span, 2)
    # Resolution order: server per-sec > server tokens/time > client rate.
    decode_tps = decode_tps_server
    if (decode_tps is None and completion_time_s is not None
            and completion_time_s > 0 and completion_tokens):
        decode_tps = round(completion_tokens / completion_time_s, 2)
    if decode_tps is None:
        decode_tps = decode_tps_client
    prefill_est = None
    if ttft_ms is not None and prompt_tokens is not None:
        prefill_est = round(prompt_tokens / (ttft_ms / 1000.0), 1)
    return OpenAIBenchPoint(
        target_tokens=0, model=model,
        ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
        decode_tps=decode_tps,
        decode_tps_client=decode_tps_client,
        decode_tps_server=decode_tps_server,
        completion_time_s=completion_time_s,
        prefill_tps_estimate=prefill_est,
        wall_s=round(wall_s, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stop_reason=stop_reason, error=err)
