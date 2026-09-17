"""J1 bench core — cold-cache performance measurement against llama.cpp servers.

Pure logic module: no I/O dependencies beyond ``httpx`` (already declared in
requirements.txt). The HTTP layer is injectable via a ``transport`` kwarg so
tests can mock the server with ``httpx.MockTransport``; storage reuses the
``arcturos.db`` connection helpers and schema (no duplicated DDL).

Methodology (recorded in docs/decisions/001-bench-methodology.md):

* Prompt sizes are derived from the server's own ``/tokenize`` endpoint, never
  from character counts — a 64k-char prompt tokenizes to ~85k tokens in
  practice.
* Every prompt gets a fresh ``uuid4`` string + newline so no run can be served
  from a prefix/KV cache: a cold-cache measurement. No warmup runs.
* Points hit the native ``{base_url}/completion`` endpoint (not
  ``/v1/chat/completions``) because it is the only route that exposes
  ``draft_n`` / ``draft_n_accepted`` inline in ``timings``.
* Non-stream first: TTFT is deferred to a streaming variant (tracked
  follow-up); ``ttft_ms`` is ``None`` here because the native non-stream
  response does not expose it.
* Power sampling via ``nvidia-smi`` is optional — the bench must not require a
  local GPU.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx

from arcturos import db

# Number of unit_text copies tokenized in one sizing call. One call derives a
# per-section token count shared by every target (the input is identical for
# all targets, so re-tokenizing per target would be pure redundancy).
_TOKENIZE_COPIES = 50

# Sizing margin (tokens): absorbs the uuid4 prefix (~2-3 tokens) plus the
# separating newline, so the full cold prompt lands near the target.
DEFAULT_MARGIN = 16

# Deterministic prose repeated verbatim to build sized prompt bodies. Chosen so
# the tokenizer emits a stable token count per section across runs and engines.
DEFAULT_UNIT_TEXT = (
    "Arcturos measures cold-cache inference performance at controlled context "
    "lengths. This section is deterministic prose: repeated verbatim it yields "
    "a stable token count per section, so prompt sizing is reproducible across "
    "runs and machines. The fresh UUID prefix defeats prefix and KV caching, "
    "and the predictable body keeps multi-token prediction draft acceptance "
    "meaningful.\n"
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class PromptPlan:
    """A sized cold prompt: ``unit_text`` repeated ``sections`` times.

    ``sections`` is derived from the server's own tokenizer so the resulting
    prompt lands near ``target_tokens`` (within the sizing margin's slack).
    """

    target_tokens: int
    sections: int
    unit_text: str


@dataclass
class BenchPoint:
    """One measured point (one context length) from a benchmark run."""

    target_tokens: int
    prefill_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    ttft_ms: Optional[float] = None
    wall_s: Optional[float] = None
    output_tokens: Optional[int] = None
    mtp_draft_n: Optional[int] = None
    mtp_accepted: Optional[int] = None
    stop_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    power_watts: Optional[float] = None
    power_host: Optional[str] = None
    power_gpu_index: Optional[int] = None


class RunNotFoundError(ValueError):
    """Raised by export when a run_id does not exist (404-style)."""

    status_code = 404


# ------------------------------------------------------------ helpers --------


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(d: dict, *keys):
    """First non-None value among *keys, or None. Tolerates shape drift."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _client(timeout: float, transport, api_key: str | None = None) -> httpx.Client:
    kwargs: dict = {"timeout": timeout}
    if transport is not None:
        kwargs["transport"] = transport
    if api_key:
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return httpx.Client(**kwargs)


def _check_response(resp: httpx.Response, context: str) -> None:
    if resp.status_code != 200:
        raise RuntimeError(f"{context}: HTTP {resp.status_code}: {resp.text[:500]!r}")


# ------------------------------------------------------------- sizing --------


def plan_prompt_sizes(
    target_tokens: list[int],
    tokenize_url: str,
    unit_text: str,
    margin: int = DEFAULT_MARGIN,
    transport=None,
    api_key: str | None = None,
) -> list[PromptPlan]:
    """Size prompts for each target using the server's real tokenizer.

    Tokenizes ``unit_text * 50`` once via ``POST {tokenize_url}`` with body
    ``{"content": unit_text * 50}`` (llama.cpp shape ``{"tokens": [...]}``),
    derives ``tokens_per_section = len(tokens) / 50``, then computes
    ``n_sections = int((target - margin) / tokens_per_section)`` for each
    target. The margin leaves room for the cold-cache UUID prefix + newline.

    Returns one :class:`PromptPlan` per target, in input order.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin!r}")
    body = unit_text * _TOKENIZE_COPIES
    with _client(60.0, transport, api_key=api_key) as client:
        resp = client.post(tokenize_url, json={"content": body})
    _check_response(resp, f"tokenize sizing failed ({tokenize_url})")
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            f"tokenize sizing: non-JSON response {resp.text[:500]!r}"
        ) from None
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"unexpected /tokenize response shape: {data!r}")
    tokens_per_section = len(tokens) / _TOKENIZE_COPIES
    if tokens_per_section <= 0:
        raise ValueError("tokenizer returned zero tokens per section")

    plans: list[PromptPlan] = []
    for target in target_tokens:
        sections = max(1, int((target - margin) / tokens_per_section))
        plans.append(
            PromptPlan(target_tokens=target, sections=sections, unit_text=unit_text)
        )
    return plans


def build_cold_prompt(plan: PromptPlan) -> str:
    """Fresh UUID prefix + newline over the deterministic unit_text body.

    The unique prefix defeats prefix/KV caching (cold-cache measurement); the
    deterministic body keeps MTP draft acceptance meaningful (the drafts are
    always generated against the same predictable text).
    """
    return f"{uuid.uuid4()}\n{plan.unit_text * plan.sections}"


# ----------------------------------------------------------- measurement ----


def run_benchmark_point(
    base_url: str,
    model: str,
    plan: PromptPlan,
    n_predict: int,
    timeout: float = 3600,
    transport=None,
    api_key: str | None = None,
) -> BenchPoint:
    """Run one cold prompt against the native llama.cpp ``/completion`` endpoint.

    POSTs ``{"prompt": cold_prompt, "n_predict": n_predict, "stream": False,
    "cache_prompt": False}`` to ``{base_url}/completion`` and parses the
    inline ``timings`` object:

    * ``timings.prompt_per_second`` -> ``prefill_tps``
    * ``timings.predicted_per_second`` -> ``decode_tps``
    * ``timings.draft_n`` / ``timings.draft_n_accepted`` -> MTP fields
      (``None`` when the engine does not report them)
    * ``tokens_predicted`` -> ``output_tokens``; ``stop_reason`` -> ``stop_reason``
    * ``tokens_prompt`` / ``tokens_evaluated`` -> ``prompt_tokens`` if present

    ``ttft_ms`` is always ``None`` here (native non-stream exposes no TTFT;
    deferred to a streaming variant). ``wall_s`` is the client-side round trip.
    ``model`` is accepted for interface symmetry with multi-model servers; the
    native endpoint is single-model and takes no model field.
    """
    prompt = build_cold_prompt(plan)
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "cache_prompt": False,
    }
    url = f"{base_url.rstrip('/')}/completion"
    with _client(timeout, transport, api_key=api_key) as client:
        start = time.perf_counter()
        resp = client.post(url, json=payload)
        wall_s = time.perf_counter() - start
    _check_response(resp, f"benchmark point failed ({url}, model={model!r})")
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            f"benchmark point: non-JSON response {resp.text[:500]!r}"
        ) from None

    timings = data.get("timings") or {}
    return BenchPoint(
        target_tokens=plan.target_tokens,
        prefill_tps=_as_float(timings.get("prompt_per_second")),
        decode_tps=_as_float(timings.get("predicted_per_second")),
        ttft_ms=None,
        wall_s=wall_s,
        output_tokens=_as_int(data.get("tokens_predicted")),
        mtp_draft_n=_as_int(timings.get("draft_n")),
        mtp_accepted=_as_int(timings.get("draft_n_accepted")),
        stop_reason=data.get("stop_reason"),
        prompt_tokens=_as_int(
            _first_present(data, "tokens_prompt", "tokens_evaluated", "prompt_tokens")
        ),
    )


def run_benchmark(
    base_url: str,
    model: str,
    targets: list[int],
    n_predict: int,
    timeout: float,
    on_point: Optional[Callable[[BenchPoint], None]] = None,
    unit_text: str = DEFAULT_UNIT_TEXT,
    transport=None,
    api_key: str | None = None,
) -> list[BenchPoint]:
    """Plan sizes for every target, then run each point sequentially.

    Each point is planned via ``{base_url}/tokenize`` and executed via the
    native ``{base_url}/completion`` endpoint. ``on_point`` is invoked with
    each :class:`BenchPoint` immediately after it completes, so a caller can
    stream results into the DB (or a UI) as they arrive.
    """
    tokenize_url = f"{base_url.rstrip('/')}/tokenize"
    plans = plan_prompt_sizes(targets, tokenize_url, unit_text, transport=transport,
                              api_key=api_key)
    points: list[BenchPoint] = []
    for plan in plans:
        point = run_benchmark_point(
            base_url, model, plan, n_predict, timeout=timeout, transport=transport,
            api_key=api_key
        )
        points.append(point)
        if on_point is not None:
            on_point(point)
    return points


# ------------------------------------------------------------- power --------


def _read_power_draw(cmd: list[str]) -> Optional[float]:
    """One nvidia-smi power.draw sample, or None on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def power_draw_avg(gpu_index: int, duration_s: int) -> Optional[float]:
    """Mean GPU power draw (W) over ``duration_s``, sampling every 1s.

    Samples ``nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounlets
    -i <gpu_index>`` from a daemon thread and returns the mean of the samples.
    Returns ``None`` gracefully if ``nvidia-smi`` is missing or fails (a bench
    must not require a local GPU).
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    samples: list[float] = []

    def _sample_loop() -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            value = _read_power_draw(cmd)
            if value is None:
                return
            samples.append(value)
            time.sleep(1)

    thread = threading.Thread(target=_sample_loop, daemon=True)
    thread.start()
    thread.join()
    if not samples:
        return None
    return sum(samples) / len(samples)


# --------------------------------------------------------- engine meta ------


_PROPS_KEYS = ("n_ctx", "model_path", "build", "engine", "system_info")


def capture_engine_metadata(base_url: str, timeout: float = 15.0, transport=None,
                            api_key: str | None = None) -> dict:
    """Capture engine metadata from the llama.cpp ``/props`` endpoint.

    Returns a dict with ``n_ctx``, ``model_path``, ``build``, ``engine`` and
    ``system_info`` keys. If the endpoint is missing, unreachable, or a field
    is absent, the corresponding value is ``None`` (never raises).
    """
    empty = {key: None for key in _PROPS_KEYS}
    try:
        with _client(timeout, transport, api_key=api_key) as client:
            resp = client.get(f"{base_url.rstrip('/')}/props")
    except httpx.HTTPError:
        return empty
    if resp.status_code != 200:
        return empty
    try:
        props = resp.json()
    except ValueError:
        return empty
    return {key: props.get(key) for key in _PROPS_KEYS}


# ------------------------------------------------------------ storage --------


def store_benchmark_run(
    db_path, base_url: str, engine_metadata: dict, points: list[BenchPoint]
) -> int:
    """Store a benchmark run + its points into the Arcturos SQLite schema.

    Inserts one ``runs`` row (server_url, model_fingerprint = engine model path
    or ``"unknown"``, engine = ``"llama.cpp"``, context_size = metadata n_ctx,
    created_at = now ISO) and one ``benchmarks`` row per point. The schema's
    ``runs.context_size`` column is NOT NULL, so an unknown n_ctx is stored as
    ``0`` (documented sentinel for "unknown"). Returns the new run_id.
    """
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        fingerprint = engine_metadata.get("model_path") or "unknown"
        n_ctx = engine_metadata.get("n_ctx")
        cur = conn.execute(
            "INSERT INTO runs (server_url, model_fingerprint, engine, context_size, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                base_url,
                fingerprint,
                "llama.cpp",
                n_ctx if n_ctx is not None else 0,
                _utcnow(),
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("insert into runs did not return a rowid")
        run_id = int(cur.lastrowid)
        now = _utcnow()
        for point in points:
            conn.execute(
                "INSERT INTO benchmarks (run_id, context_tokens, prefill_tps, decode_tps,"
                " ttft_ms, wall_s, output_tokens, mtp_draft_n, mtp_accepted, power_watts,"
                " power_host, power_gpu_index, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    point.target_tokens,
                    point.prefill_tps,
                    point.decode_tps,
                    point.ttft_ms,
                    point.wall_s,
                    point.output_tokens,
                    point.mtp_draft_n,
                    point.mtp_accepted,
                    point.power_watts,
                    point.power_host,
                    point.power_gpu_index,
                    now,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return run_id


def export_run_json(db_path, run_id: int) -> dict:
    """Export a stored run + its benchmark points as a dict.

    Raises :class:`RunNotFoundError` (a ValueError subclass) if the run_id is
    missing.
    """
    conn = db.connect(db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise RunNotFoundError(f"run {run_id} not found")
        # benchmarks has no id column (keyed by run_id); rowid = insert order.
        points = conn.execute(
            "SELECT * FROM benchmarks WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()
    finally:
        conn.close()
    return {"run": dict(run), "benchmarks": [dict(p) for p in points]}
