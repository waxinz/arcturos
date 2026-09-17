"""Tests for the streaming bench variant and power sampler."""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos.bench_stream import run_stream_point, StreamBenchPoint  # noqa: E402
from arcturos.power import PowerSampler, measure_power_during  # noqa: E402


def _sse_chunks(n_tokens: int, final_timings: dict) -> str:
    import json as _json
    lines = []
    for i in range(1, n_tokens + 1):
        chunk = {"index": 0, "content": f" t{i}", "tokens_predicted": i,
                 "tokens_evaluated": 27, "stop": False}
        if i == n_tokens:
            chunk["stop"] = True
            chunk["stop_reason"] = "eos"
            chunk["timings"] = final_timings
        lines.append("data: " + _json.dumps(chunk))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


class _FakeResp:
    def __init__(self, text):
        # real urlopen yields byte lines; mirror that
        self._lines = text.encode().splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_stream_point_parses_ttft_and_client_decode():
    fake = _sse_chunks(10, {"prompt_per_second": 400.0,
                            "predicted_per_second": 50.0,
                            "draft_n": 30, "draft_n_accepted": 25})
    with patch("arcturos.bench_stream.urlreq.urlopen", return_value=_FakeResp(fake)):
        point = run_stream_point("http://fake:8000", "some prompt", 10)
    assert point.ttft_ms is not None and point.ttft_ms >= 0
    assert point.output_tokens == 10
    assert point.stop_reason == "eos"
    assert point.mtp_draft_n == 30 and point.mtp_accepted == 25
    assert point.decode_tps_server == 50.0
    assert point.prefill_tps_server == 400.0
    assert point.prompt_tokens == 27


def test_stream_point_no_tokens_ttft_none():
    fake = "data: [DONE]\n\n"
    with patch("arcturos.bench_stream.urlreq.urlopen", return_value=_FakeResp(fake)):
        point = run_stream_point("http://fake:8000", "p", 1)
    assert point.ttft_ms is None
    assert point.output_tokens is None


def test_power_sampler_mean_or_none():
    with patch("arcturos.power._query_power", return_value=250.0):
        s = PowerSampler("fakehost", interval_s=0.05)
        s.start()
        time.sleep(0.3)
        watts = s.stop()
    assert watts is not None and 248 <= watts <= 252

    with patch("arcturos.power._query_power", return_value=None):
        s2 = PowerSampler("fakehost", interval_s=0.05)
        s2.start()
        time.sleep(0.15)
        assert s2.stop() is None


def test_measure_power_during_returns_both():
    def work(x):
        time.sleep(0.15)
        return x * 2

    with patch("arcturos.power._query_power", return_value=249.0):
        (res, watts) = measure_power_during("fakehost", work, 21)
    assert res == 42
    assert watts is not None
