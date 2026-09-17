"""OpenAI-compatible bench path (litellm / vLLM / any /v1/chat/completions).

Unlike llama.cpp native /completion, these servers emit no inline timings —
all metrics are computed CLIENT-SIDE from SSE chunk arrival times, with
`stream_options: {"include_usage": true}` for authoritative token counts.

TTFT  = wall time to the first chunk with content.
Decode tps = (completion_tokens - 1) / (t_last - t_first).
Prefill tps ≈ prompt_tokens / ttft (an estimate; flagged as such).
"""

from __future__ import annotations

import json
import time
import urllib.request as urlreq
import urllib.error as uerr

from dataclasses import dataclass


@dataclass
class OpenAIBenchPoint:
    target_tokens: int
    model: str
    ttft_ms: float | None
    decode_tps_client: float | None
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
    first_content_t = None
    last_content_t = None
    completion_tokens = None
    prompt_tokens = None
    stop_reason = None
    err = None
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
                    if delta.get("content"):
                        now = time.monotonic()
                        if first_content_t is None:
                            first_content_t = now
                            ttft_ms = (now - t_start) * 1000.0
                        last_content_t = now
                    fr = choices[0].get("finish_reason")
                    if fr:
                        stop_reason = fr
                usage = chunk.get("usage") or {}
                if usage:
                    completion_tokens = usage.get("completion_tokens",
                                                  completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except (uerr.URLError, OSError) as e:
        return OpenAIBenchPoint(
            target_tokens=0, model=model, ttft_ms=None,
            decode_tps_client=None, prefill_tps_estimate=None,
            wall_s=time.monotonic() - t_start,
            prompt_tokens=None, completion_tokens=None,
            stop_reason=None, error=str(e))
    wall_s = time.monotonic() - t_start
    decode_tps_client = None
    if first_content_t and last_content_t and last_content_t > first_content_t:
        gen_span = last_content_t - first_content_t
        n = (completion_tokens or 1) - 1
        if gen_span > 0 and n > 0:
            decode_tps_client = round(n / gen_span, 2)
    prefill_est = None
    if ttft_ms and prompt_tokens:
        prefill_est = round(prompt_tokens / (ttft_ms / 1000.0), 1)
    return OpenAIBenchPoint(
        target_tokens=0, model=model,
        ttft_ms=round(ttft_ms, 1) if ttft_ms else None,
        decode_tps_client=decode_tps_client,
        prefill_tps_estimate=prefill_est,
        wall_s=round(wall_s, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stop_reason=stop_reason, error=err)
