"""OpenAI-compatible bench path (litellm / vLLM / any /v1/chat/completions).

Unlike llama.cpp native /completion, these servers emit no inline timings —
all metrics are computed CLIENT-SIDE from SSE chunk arrival times, with
`stream_options: {"include_usage": true}` for authoritative token counts.

TTFT  = wall time to the first chunk of any stream payload (reasoning
models emit reasoning first — delta key `reasoning` on vLLM,
`reasoning_content` on tabbyAPI; that latency is real user-facing
latency).

Decode tps resolution — most-authoritative source wins:
  1. usage.completion_tokens_per_sec  (server-accounted generation rate;
     single-segment points only — a per-request rate does not sum)
  2. summed usage.completion_time across segments (server-accounted
     generation time) -> completion_tokens / completion_time
  3. client-side content-chunk rate, only when the accumulated content
     span is at least _MIN_CLIENT_SPAN_S. Below that the server
     burst-flushed the stream (SSE coalescing): chunk arrival measures
     socket drain, not decode (this produced a 15054 t/s artifact on
     run 17).

n_predict enforcement (2026-10-26): ``max_tokens`` is the EXACT number of
tokens the point must decode. Servers that honor the cap end the single
request with finish_reason "length". Servers that stop at EOS early
(Strata/Maya builds ignore both ignore_eos and min_tokens, and current
engine builds reject logit_bias — measured on maya-ds4 v1.1.28: a
48-token cap produced 5) get CONTINUATION requests — the partial reply
re-sent as an assistant message followed by a user "continue" turn —
until the accumulated usage.completion_tokens reaches the cap. Each
segment asks only for the remaining tokens; token counts are summed from
server usage, so the stored point decodes exactly n_predict tokens no
matter how many segments it took. A server that returns 0 tokens twice in
a row, or that is still short after _MAX_ENFORCE_SEGMENTS segments,
records the shortfall in ``error`` instead of looping forever.

Prefill tps ≈ prompt_tokens / ttft (an estimate; flagged as such). The
estimate reflects the FIRST segment only — continuation segments ride the
server's prompt cache or re-prefill, neither of which changes the decode
window the point measures.
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

# Safety cap on enforcement segments (early-EOS servers top up ~10-100
# tokens per segment, so even a 4096-token cap lands far below this).
_MAX_ENFORCE_SEGMENTS = 256

# The user turn that drives a continuation segment. Content is irrelevant
# (bench prompts are synthetic filler); what matters is that the rendered
# prompt keeps the previous segments' tokens as a cacheable prefix.
_CONTINUE_TURN = "Continue."


@dataclass
class OpenAIBenchPoint:
    target_tokens: int
    model: str
    ttft_ms: float | None
    decode_tps: float | None  # resolved: server-authoritative > client
    decode_tps_client: float | None  # raw client chunk rate (content chunks / span)
    decode_tps_server: float | None  # server-accounted rate (see module doc)
    completion_time_s: float | None  # usage.completion_time, summed over segments
    prefill_tps_estimate: float | None  # derived, not server-authoritative
    wall_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    stop_reason: str | None
    mtp_draft_n: int | None = None
    mtp_accepted: int | None = None
    error: str | None = None


def _normalize_base(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


class _Segment:
    """One request/response cycle's client-side measurements."""
    __slots__ = ("ttft_ms", "completion_tokens", "prompt_tokens",
                 "tokens_per_sec", "completion_time_s", "stop_reason",
                 "mtp_draft_n", "mtp_accepted", "gen_span_s",
                 "payload_chunks", "content_pieces", "error")

    ttft_ms: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    tokens_per_sec: float | None
    completion_time_s: float | None
    stop_reason: str | None
    mtp_draft_n: int | None
    mtp_accepted: int | None
    gen_span_s: float
    payload_chunks: int
    content_pieces: list
    error: str | None

    def __init__(self):
        self.ttft_ms = None
        self.completion_tokens = None
        self.prompt_tokens = None
        self.tokens_per_sec = None
        self.completion_time_s = None
        self.stop_reason = None
        self.mtp_draft_n = None
        self.mtp_accepted = None
        self.gen_span_s = 0.0
        self.payload_chunks = 0
        self.content_pieces = []
        self.error = None


def _read_chat_stream(base_url: str, body: dict, api_key: str | None,
                      timeout_s: float, abort, t_start: float) -> _Segment:
    """POST one streaming chat completion and read its SSE stream."""
    seg = _Segment()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlreq.Request(
        _normalize_base(base_url) + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers=headers)
    first_payload_t = None
    last_payload_t = None
    try:
        with urlreq.urlopen(req, timeout=timeout_s) as resp:
            for line in resp:
                # Mid-stream abort (ADR 006): a stop requested while the
                # stream is decoding raises here — the point is abandoned
                # (its partial counts are NOT stored as a completed point;
                # only whole points land in the store).
                if abort is not None:
                    abort.check()
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
                    if seg.ttft_ms is None and (reasoning or delta.get("content")):
                        seg.ttft_ms = (now - t_start) * 1000.0
                    if reasoning or delta.get("content"):
                        seg.payload_chunks += 1
                        if first_payload_t is None:
                            first_payload_t = now
                        last_payload_t = now
                    if delta.get("content"):
                        seg.content_pieces.append(delta["content"])
                    fr = choices[0].get("finish_reason")
                    if fr:
                        seg.stop_reason = fr
                usage = chunk.get("usage") or {}
                if usage:
                    seg.completion_tokens = usage.get(
                        "completion_tokens", seg.completion_tokens)
                    seg.prompt_tokens = usage.get("prompt_tokens",
                                                  seg.prompt_tokens)
                    cps = usage.get("completion_tokens_per_sec")
                    if cps:
                        seg.tokens_per_sec = round(float(cps), 2)
                    ctime = usage.get("completion_time")
                    if ctime is not None:
                        seg.completion_time_s = float(ctime)
                    # MTP / speculative-decoding acceptance (tabbyAPI):
                    # accepted_prediction_tokens / rejected_prediction_tokens
                    # in the final usage chunk. draft_n = accepted + rejected;
                    # absent fields stay None (honest nulls).
                    acc = usage.get("completion_tokens_details", {}) or {}
                    accepted = acc.get("accepted_prediction_tokens")
                    rejected = acc.get("rejected_prediction_tokens")
                    if accepted is not None and rejected is not None:
                        seg.mtp_accepted = int(accepted)
                        seg.mtp_draft_n = int(accepted) + int(rejected)
    except (uerr.URLError, OSError) as e:
        seg.error = str(e)
    if first_payload_t is not None and last_payload_t is not None:
        seg.gen_span_s = max(0.0, last_payload_t - first_payload_t)
    return seg


def run_openai_stream_point(base_url: str, model: str, messages: list,
                            api_key: str | None = None,
                            max_tokens: int = 128,
                            timeout_s: float = 300.0,
                            abort=None) -> OpenAIBenchPoint:
    """Decode exactly ``max_tokens`` tokens; measure the decode window.

    Early-EOS servers are topped up with continuation segments until the
    server-accounted completion token count reaches the cap (module
    docstring). TTFT/prefill come from the first segment; decode rates
    aggregate every segment (single-segment per-request rate first, then
    summed server-accounted time, then the client chunk span).
    """
    t_start = time.monotonic()
    total_tokens = 0
    prompt_tokens = None
    ttft_ms = None
    single_cps = None
    completion_time_sum = 0.0
    have_completion_time = False
    gen_span_total = 0.0
    payload_chunk_total = 0
    stop_reason = None
    mtp_draft_n = None
    mtp_accepted = None
    acc_text = []
    base_messages = list(messages)
    chat_messages = base_messages
    stall = 0
    segments = 0
    err = None

    while True:
        remaining = max_tokens - total_tokens
        body = {
            "model": model, "messages": chat_messages,
            "max_tokens": remaining, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
        }
        seg = _read_chat_stream(base_url, body, api_key, timeout_s, abort,
                                t_start)
        segments += 1
        if seg.error:
            return OpenAIBenchPoint(
                target_tokens=max_tokens, model=model,
                ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
                decode_tps=None, decode_tps_client=None,
                decode_tps_server=None,
                completion_time_s=(completion_time_sum
                                   if have_completion_time else None),
                prefill_tps_estimate=None,
                wall_s=round(time.monotonic() - t_start, 3),
                prompt_tokens=prompt_tokens,
                completion_tokens=total_tokens if total_tokens else None,
                mtp_draft_n=mtp_draft_n, mtp_accepted=mtp_accepted,
                stop_reason=stop_reason, error=seg.error)
        seg_tokens = seg.completion_tokens or 0
        total_tokens += seg_tokens
        if prompt_tokens is None:
            prompt_tokens = seg.prompt_tokens
        if ttft_ms is None:
            ttft_ms = seg.ttft_ms
        gen_span_total += seg.gen_span_s
        payload_chunk_total += seg.payload_chunks
        if seg.tokens_per_sec is not None:
            single_cps = seg.tokens_per_sec
        if seg.completion_time_s is not None:
            completion_time_sum += seg.completion_time_s
            have_completion_time = True
        if seg.stop_reason:
            stop_reason = seg.stop_reason
        if seg.mtp_draft_n is not None:
            mtp_draft_n = seg.mtp_draft_n
        if seg.mtp_accepted is not None:
            mtp_accepted = seg.mtp_accepted
        if seg.content_pieces:
            acc_text.extend(seg.content_pieces)

        if total_tokens >= max_tokens or remaining <= 0:
            break  # cap reached: exactly max_tokens decoded
        # Enforcement: the server stopped early (EOS / stop token). Top up.
        if seg_tokens == 0:
            stall += 1
            if stall >= 2:
                err = (f"n_predict enforcement stalled: server returned 0 "
                       f"tokens twice ({total_tokens}/{max_tokens})")
                break
        else:
            stall = 0
        if segments >= _MAX_ENFORCE_SEGMENTS:
            err = (f"n_predict {max_tokens} not reached after "
                   f"{segments} segments ({total_tokens} tokens)")
            break
        chat_messages = base_messages + [
            {"role": "assistant", "content": "".join(acc_text)},
            {"role": "user", "content": _CONTINUE_TURN},
        ]

    wall_s = time.monotonic() - t_start
    # Aggregated decode: a single segment's per-request rate is the most
    # authoritative (it is the server's own accounting); multiple segments
    # fall back to summed server-accounted time, then to the client chunk
    # rate — a PROXY only while the server flushes roughly one chunk per
    # token, and only over a span above the burst floor (_MIN_CLIENT_SPAN_S).
    decode_tps_server = None
    if segments == 1 and single_cps is not None:
        decode_tps_server = single_cps
    elif have_completion_time and completion_time_sum > 0 and total_tokens:
        decode_tps_server = round(total_tokens / completion_time_sum, 2)
    decode_tps_client = None
    if gen_span_total >= _MIN_CLIENT_SPAN_S and payload_chunk_total:
        decode_tps_client = round(payload_chunk_total / gen_span_total, 2)
    decode_tps = decode_tps_server
    if decode_tps is None:
        decode_tps = decode_tps_client
    prefill_est = None
    if ttft_ms is not None and prompt_tokens is not None:
        prefill_est = round(prompt_tokens / (ttft_ms / 1000.0), 1)
    return OpenAIBenchPoint(
        target_tokens=max_tokens, model=model,
        ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
        decode_tps=decode_tps,
        decode_tps_client=decode_tps_client,
        decode_tps_server=decode_tps_server,
        completion_time_s=completion_time_sum if have_completion_time else None,
        prefill_tps_estimate=prefill_est,
        wall_s=round(wall_s, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=total_tokens if total_tokens else None,
        mtp_draft_n=mtp_draft_n,
        mtp_accepted=mtp_accepted,
        stop_reason=stop_reason, error=err)
