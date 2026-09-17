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


def _sse(content_chunks=3, usage={"prompt_tokens": 100, "completion_tokens": 30}):
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
    assert p.decode_tps_client  # 29 tokens over measured span


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
