"""OpenAI-compatible bench path tests — SSE parsing, usage extraction, errors."""

import sys
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos.bench_openai import run_openai_stream_point, _normalize_base  # noqa: E402


class _FakeResp:
    def __init__(self, text):
        self._lines = text.encode().splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse(content_chunks=3, usage={"prompt_tokens": 100, "completion_tokens": 30,
                                  "completion_time": 0.65,
                                  "completion_tokens_per_sec": 46.15}):
    import json as _json
    lines = []
    for i in range(content_chunks):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def test_normalize_base_variants():
    assert _normalize_base("http://h:4000/v1/") == "http://h:4000"
    assert _normalize_base("http://h:4000") == "http://h:4000"
    assert _normalize_base("http://h:4000/v1") == "http://h:4000"


def test_happy_path_extracts_ttft_and_usage():
    fake = _sse()
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "test-model",
                                    [{"role": "user", "content": "hi"}],
                                    max_tokens=30)
    assert p.ttft_ms is not None
    assert p.prompt_tokens == 100
    assert p.completion_tokens == 30
    assert p.stop_reason == "stop"
    assert p.error is None
    assert p.prefill_tps_estimate  # derived
    # server-authoritative decode wins; the instant fake span is below the
    # burst floor so the client chunk rate is rejected
    assert p.decode_tps == 46.15
    assert p.decode_tps_server == 46.15
    assert p.completion_time_s == 0.65
    assert p.decode_tps_client is None


def test_error_path_returns_error_not_raise():
    class _BoomResp:
        def __iter__(self):
            raise OSError("connection reset")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_BoomResp()):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}])
    assert p.error is not None
    assert "reset" in p.error


def test_empty_stream():
    fake = "data: [DONE]\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}])
    assert p.ttft_ms is None
    assert p.completion_tokens is None
    # n_predict enforcement (2026-10-26): a server that returns nothing
    # twice in a row cannot reach the cap — the stall guard records the
    # shortfall instead of looping forever.
    assert p.error is not None
    assert "stalled" in p.error


def test_reasoning_model_ttft_uses_first_reasoning_chunk():
    import json as _json
    lines = []
    for rc in ["We", " need"]:
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"reasoning_content": rc}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    fake = "\n\n".join(lines) + "\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=5)
    assert p.ttft_ms is not None  # first reasoning chunk counts
    assert p.completion_tokens == 5
    # instant fake stream -> content span ~0 -> decode rate honestly None
    assert p.decode_tps_client is None
    assert p.decode_tps is None  # no server timing in this fake either
    # happy path with content chunks still computes decode over real span:
    # (covered by test_happy_path_extracts_ttft_and_usage timing variance)


def test_completion_time_fallback_without_per_sec():
    import json as _json
    lines = []
    for i in range(4):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"t{i}"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 30,
                   "completion_time": 0.75}}))
    fake = "\n\n".join(lines) + "\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=30)
    assert p.decode_tps_server == 40.0  # server-accounted: 30 tokens / 0.75 s
    assert p.completion_time_s == 0.75
    assert p.decode_tps == 40.0  # 30 tokens / 0.75 s


def test_reasoning_only_stream_gets_server_decode():
    # Run-17 point-2 class: a reasoning model burns all max_tokens on
    # reasoning_content -> zero content chunks -> client rate impossible.
    # The server's completion_tokens_per_sec still yields a decode value.
    import json as _json
    lines = []
    for rc in ["thin", "king", "hard"]:
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"reasoning_content": rc},
                          "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                   "completion_time": 0.6,
                   "completion_tokens_per_sec": 5.0}}))
    fake = "\n\n".join(lines) + "\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=3)
    assert p.decode_tps == 5.0
    assert p.decode_tps_client is None


def test_burst_span_rejected_without_server_timing():
    # Run-17 point-1 class: all chunks arrive in one burst (span ~17 ms).
    # Simulated by the instant fake stream; without server timing the
    # degenerate span must yield None, not a 15000 t/s artifact.
    import json as _json
    lines = []
    for i in range(64):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 64}}))
    fake = "\n\n".join(lines) + "\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=64)
    assert p.decode_tps is None
    assert p.decode_tps_client is None


def test_client_rate_used_when_span_real_and_no_server_timing():
    import itertools
    import json as _json
    from unittest.mock import MagicMock
    lines = []
    for i in range(5):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    fake = "\n\n".join(lines) + "\n\n"
    clock = MagicMock()
    clock.monotonic.side_effect = itertools.count(0, 0.1)
    with patch("arcturos.bench_openai.time", clock), \
         patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=5)
    # 5 content chunks over a 0.4 s span (0.1 s per tick) -> 12.5 chunks/s
    assert p.decode_tps_client == 12.5
    assert p.decode_tps == 12.5


def test_vllm_reasoning_key_ttft_and_client_rate():
    # Run-19 class: vLLM emits reasoning under delta key `reasoning`
    # (NOT `reasoning_content`) and its usage carries no timing fields.
    # TTFT must fire on the first reasoning chunk, and the client-rate
    # fallback must count reasoning chunks so decode populates.
    import itertools
    import json as _json
    from unittest.mock import MagicMock
    lines = []
    for rc in ["thin", "king", "hard"]:
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"reasoning": rc}, "finish_reason": None}]}))
    for i in range(2):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"tok{i}"},
                          "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    fake = "\n\n".join(lines) + "\n\n"
    clock = MagicMock()
    clock.monotonic.side_effect = itertools.count(0, 0.1)
    with patch("arcturos.bench_openai.time", clock), \
         patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:8000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=5)
    assert p.ttft_ms is not None  # first reasoning chunk counts
    assert p.completion_tokens == 5
    assert p.prompt_tokens == 10
    assert p.prefill_tps_estimate is not None
    # vLLM usage has no completion_tokens_per_sec / completion_time
    assert p.decode_tps_server is None
    assert p.completion_time_s is None
    # 5 payload chunks (3 reasoning + 2 content) over a 0.4 s span
    assert p.decode_tps_client == 12.5
    assert p.decode_tps == 12.5


def test_vllm_usage_without_timing_fields_stays_null():
    # vLLM usage chunk: bare token counts only. The decode resolution
    # chain must keep honest nulls (no fabricated server rate).
    import json as _json
    lines = []
    for i in range(3):
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": f"t{i}"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 64, "completion_tokens": 128,
                   "total_tokens": 192}}))
    fake = "\n\n".join(lines) + "\n\n"
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:8000/v1", "m",
                                    [{"role": "user", "content": "x"}])
    assert p.decode_tps_server is None
    assert p.completion_time_s is None
    assert p.completion_tokens == 128
    assert p.prompt_tokens == 64
    # instant fake stream -> burst floor rejects the client rate too
    assert p.decode_tps_client is None
    assert p.decode_tps is None


def test_reasoning_content_stream_still_measured():
    # Regression: tabbyAPI-style `reasoning_content` streams keep firing
    # TTFT and counting toward the client-rate fallback.
    import itertools
    import json as _json
    from unittest.mock import MagicMock
    lines = []
    for rc in ["We", " need"]:
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"reasoning_content": rc},
                          "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]}))
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}}))
    fake = "\n\n".join(lines) + "\n\n"
    clock = MagicMock()
    clock.monotonic.side_effect = itertools.count(0, 0.15)
    with patch("arcturos.bench_openai.time", clock), \
         patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:8000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=3)
    assert p.ttft_ms is not None
    assert p.decode_tps_server is None
    assert p.completion_time_s is None
    # 3 payload chunks (2 reasoning_content + 1 content) over a 0.3 s span
    # (0.15 s per tick) -> 10.0 chunks/s
    assert p.decode_tps_client == 10.0
    assert p.decode_tps == 10.0


# ------------------------------------------------------- MTP acceptance ----


def test_mtp_acceptance_extracted_from_usage_details():
    """tabbyAPI-style speculative decoding counters in the final usage
    chunk map onto mtp_accepted / mtp_draft_n (accepted + rejected)."""
    fake = _sse(usage={"prompt_tokens": 100, "completion_tokens": 200,
                       "completion_time": 2.95,
                       "completion_tokens_details": {
                           "accepted_prediction_tokens": 109,
                           "rejected_prediction_tokens": 73}})
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:8000/v1", "m",
                                    [{"role": "user", "content": "x"}])
    assert p.mtp_accepted == 109
    assert p.mtp_draft_n == 182  # 109 accepted + 73 rejected


def test_mtp_fields_null_when_usage_details_absent():
    """No completion_tokens_details -> honest nulls (never zeros)."""
    fake = _sse()
    with patch("arcturos.bench_openai.urlreq.urlopen", return_value=_FakeResp(fake)):
        p = run_openai_stream_point("http://fake:8000/v1", "m",
                                    [{"role": "user", "content": "x"}],
                                    max_tokens=30)
    assert p.mtp_draft_n is None
    assert p.mtp_accepted is None


# ------------------------------------------------- n_predict enforcement ----


def _seg_sse(content, completion, finish, prompt=100,
             completion_time=None):
    """One segment's SSE body: content deltas + a final usage chunk."""
    import json as _json
    lines = []
    if content:
        lines.append("data: " + _json.dumps(
            {"choices": [{"delta": {"content": content},
                          "finish_reason": None}]}))
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if completion_time is not None:
        usage["completion_time"] = completion_time
    lines.append("data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": finish}], "usage": usage}))
    return "\n\n".join(lines) + "\n\n"


def test_enforcement_continuation_reaches_cap():
    """Early-EOS server (finish 'stop' below the cap) is topped up with
    continuation requests until usage.completion_tokens sums to exactly
    max_tokens (2026-10-26: runs #187/#188 decoded 15-116 tokens)."""
    import json as _json
    calls = []

    def fake_urlopen(req, timeout=None):
        body = _json.loads(req.data.decode())
        calls.append(body)
        if len(calls) == 1:
            return _FakeResp(_seg_sse("part one ", 30, "stop",
                                      completion_time=0.6))
        return _FakeResp(_seg_sse("part two", 18, "length",
                                  prompt=140, completion_time=0.4))

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "hi"}],
                                    max_tokens=48)
    assert len(calls) == 2
    # first segment asks for the full cap; second asks only for the rest
    assert calls[0]["max_tokens"] == 48
    assert calls[1]["max_tokens"] == 18
    # continuation shape: original messages + partial reply + continue turn
    assert calls[1]["messages"][0] == {"role": "user", "content": "hi"}
    assert calls[1]["messages"][1] == {"role": "assistant",
                                       "content": "part one "}
    assert calls[1]["messages"][2]["role"] == "user"
    # exactly the cap decoded, summed from server usage
    assert p.completion_tokens == 48
    assert p.target_tokens == 48  # was hardcoded 0 before enforcement
    assert p.error is None
    assert p.stop_reason == "length"  # last segment's finish
    assert p.prompt_tokens == 100  # first segment's prompt
    # aggregated server-accounted decode: 48 tokens over 1.0 s
    assert p.completion_time_s == 1.0
    assert p.decode_tps_server == 48.0
    assert p.decode_tps == 48.0


def test_enforcement_cap_honored_single_segment():
    """A server that honors the cap (finish 'length' at max_tokens) runs
    ONE request; target_tokens records the configured count (the old code
    hardcoded 0, making a shortfall invisible in the store)."""
    import json as _json
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(_json.loads(req.data.decode()))
        return _FakeResp(_seg_sse("x" * 8, 48, "length",
                                  completion_time=2.0))

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "hi"}],
                                    max_tokens=48)
    assert len(calls) == 1
    assert p.completion_tokens == 48
    assert p.target_tokens == 48
    assert p.error is None


def test_enforcement_stall_guard():
    """A server returning 0 tokens twice in a row ends the point with an
    honest error instead of looping forever."""
    import json as _json
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(_json.loads(req.data.decode()))
        if len(calls) == 1:
            return _FakeResp(_seg_sse("bit", 10, "stop"))
        return _FakeResp("data: [DONE]\n\n")

    with patch("arcturos.bench_openai.urlreq.urlopen",
               side_effect=fake_urlopen):
        p = run_openai_stream_point("http://fake:4000/v1", "m",
                                    [{"role": "user", "content": "hi"}],
                                    max_tokens=128)
    assert len(calls) == 3  # 10-token segment + two 0-token segments
    assert p.completion_tokens == 10
    assert p.target_tokens == 128
    assert p.error is not None
    assert "stalled" in p.error


def test_enforcement_native_payload_carries_ignore_eos():
    """The native /completion path enforces the cap server-side with
    llama.cpp's ignore_eos (exact n_predict in one request)."""
    from arcturos import bench
    seen = {}

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {"timings": {}, "tokens_predicted": 16}

    def fake_post(url, json=None, **kw):
        seen.update(json)
        return _Resp()

    class _Client:
        def __init__(self, post):
            self.post = post

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    plan = bench.PromptPlan(target_tokens=256, sections=1, unit_text="u")
    with patch.object(bench, "_client",
                      return_value=_Client(fake_post)), \
         patch.object(bench.time, "perf_counter", side_effect=[0.0, 1.0]):
        point = bench.run_benchmark_point("http://fake:8000", "m", plan,
                                          n_predict=16)
    assert seen["n_predict"] == 16
    assert seen["ignore_eos"] is True
    assert point.n_predict == 16
