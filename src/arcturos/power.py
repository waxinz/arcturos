"""Power draw sampling — nvidia-smi on a remote GPU host over SSH.

Average power (W) during a benchmark point. Optional: if SSH or nvidia-smi
fails, callers get None and the bench proceeds without power metrics.
"""

from __future__ import annotations

import statistics
import threading
import time


def _query_power(host: str, gpu_index: int = 0) -> float | None:
    import subprocess

    try:
        out = subprocess.run(
            ["ssh", host, "nvidia-smi --query-gpu=power.draw "
             "--format=csv,noheader,nounlets -i %d" % gpu_index],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0].replace(" W", "").strip())
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


class PowerSampler:
    """Background sampler; call start() before the bench point and stop()
    after. Returns mean W over the window, or None on failure."""

    def __init__(self, host: str, gpu_index: int = 0, interval_s: float = 1.0):
        self.host = host
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = _query_power(self.host, self.gpu_index)
            if w is not None:
                self.samples.append(w)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=20)
        if not self.samples:
            return None
        return round(statistics.mean(self.samples), 1)


def measure_power_during(host: str, fn, *args, gpu_index: int = 0,
                         **kwargs) -> tuple:
    """Run fn(*args, **kwargs) while sampling power on host; returns
    (fn_result, mean_watts_or_None)."""
    sampler = PowerSampler(host, gpu_index=gpu_index)
    sampler.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        watts = sampler.stop()
    return result, watts
