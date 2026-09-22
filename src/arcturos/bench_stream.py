"""Streaming bench variant — TTFT via llama.cpp native /completion SSE stream.

TTFT = client wall time to the FIRST chunk (tokens_predicted >= 1).
Decode tok/s = client-side over the streamed token span; the final chunk
carries server-side timings too, which we keep alongside for cross-checking.
"""

from __future__ import annotations

import json
import time
import uuid

import urllib.request as urlreq
import urllib.error as uerr

from dataclasses import dataclass, field


@dataclass
class StreamBenchPoint:
    target_tokens: int
    prompt_tokens: int | None
    ttft_ms: float | None
    decode_tps_client: float | None
    prefill_tps_server: float | None
    decode_tps_server: float | None
    mtp_draft_n: int | None
    mtp_accepted: int | None
    wall_s: float
    output_tokens: int | None
    stop_reason: str | None
    raw_timings: dict = field(default_factory=dict)


def run_stream_point(base_url: str, prompt: str, n_predict: int,
                     timeout_s: float = 600.0,
                     api_key: str | None = None) -> StreamBenchPoint:
    """One cold streaming measurement; raises on connection failure."""
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "stream": True, "cache_prompt": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlreq.Request(base_url.rstrip("/") + "/completion", data=body,
                         headers=headers)
    t_start = time.monotonic()
    ttft_ms: float | None = None
    pred_first: int | None = None
    pred_last: int | None = None
    tokens_evaluated: int | None = None
    final_timings: dict = {}
    stop_reason: str | None = None
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
            if ttft_ms is None and chunk.get("tokens_predicted", 0) >= 1:
                ttft_ms = (time.monotonic() - t_start) * 1000.0
            pred_last = chunk.get("tokens_predicted", pred_last)
            if pred_first is None:
                pred_first = chunk.get("tokens_predicted")
            tokens_evaluated = chunk.get("tokens_evaluated")
            if chunk.get("stop"):
                stop_reason = chunk.get("stop_reason")
                final_timings = chunk.get("timings", {}) or {}
            if chunk.get("stop") and chunk.get("stop_reason") is None:
                pass  # some builds emit stop_reason separately
    wall_s = time.monotonic() - t_start
    decode_tps_client = None
    if (ttft_ms is not None and pred_last is not None and pred_first is not None
            and pred_last > pred_first):
        gen_span_s = wall_s - (ttft_ms / 1000.0)
        n_streamed = pred_last - pred_first
        if gen_span_s > 0:
            decode_tps_client = n_streamed / gen_span_s
    return StreamBenchPoint(
        target_tokens=len(prompt) // 4,  # refined by caller with real token count
        prompt_tokens=tokens_evaluated,
        ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
        decode_tps_client=round(decode_tps_client, 2) if decode_tps_client is not None else None,
        prefill_tps_server=final_timings.get("prompt_per_second"),
        decode_tps_server=final_timings.get("predicted_per_second"),
        mtp_draft_n=final_timings.get("draft_n"),
        mtp_accepted=final_timings.get("draft_n_accepted"),
        wall_s=round(wall_s, 3),
        output_tokens=pred_last,
        stop_reason=stop_reason,
        raw_timings=final_timings,
    )
