"""Unit tests for the SSH power sampler (power.py).

The sampler shells out to `ssh host nvidia-smi ...`; these tests stub the
subprocess layer so no real host is contacted.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos.power import PowerSampler, _query_power  # noqa: E402


def _run(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_query_power_parses_watts():
    with patch("subprocess.run", return_value=_run("238.41 W\n")):
        assert _query_power("host", 0) == 238.41


def test_query_power_nonzero_returncode_is_none():
    with patch("subprocess.run", return_value=_run("", returncode=255)):
        assert _query_power("host", 0) is None


def test_query_power_garbage_output_is_none():
    with patch("subprocess.run", return_value=_run("not-a-number\n")):
        assert _query_power("host", 0) is None


def test_query_power_empty_output_is_none():
    with patch("subprocess.run", return_value=_run("")):
        assert _query_power("host", 0) is None


def test_query_power_timeout_is_none():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=15)):
        assert _query_power("host", 0) is None


def test_query_power_gpu_index_reaches_command():
    captured = {}

    def fake_run(cmd, **_kw):
        captured["cmd"] = cmd
        return _run("100 W\n")

    with patch("subprocess.run", side_effect=fake_run):
        _query_power("gpu-host", 3)
    assert "-i 3" in captured["cmd"][-1]
    assert captured["cmd"][1] == "gpu-host"


def test_power_sampler_stop_returns_rounded_mean():
    """start()/stop(): the sampler records samples in its background loop
    and stop() returns the rounded mean; None when nothing was sampled."""
    outputs = [_run("100 W\n"), _run("200 W\n"), _run("300.4 W\n")]
    with patch("subprocess.run", side_effect=outputs):
        sampler = PowerSampler("gpu-host", 0, interval_s=0.001)
        sampler.start()
        import time
        deadline = time.monotonic() + 2.0
        while len(sampler.samples) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        mean = sampler.stop()
    assert mean == round((100.0 + 200.0 + 300.4) / 3, 1)


def test_power_sampler_stop_without_samples_is_none():
    with patch("subprocess.run", return_value=_run("", returncode=255)):
        sampler = PowerSampler("gpu-host", 0, interval_s=0.001)
        sampler.start()
        import time
        time.sleep(0.05)
        assert sampler.stop() is None
