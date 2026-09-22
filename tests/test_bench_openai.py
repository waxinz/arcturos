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
                                    max_tokens=32)
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
    assert p.error is None


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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
    assert p.decode_tps_server is None
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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
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
                                    [{"role": "user", "content": "x"}])
    assert p.mtp_draft_n is None
    assert p.mtp_accepted is None
