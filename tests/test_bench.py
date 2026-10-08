"""Tests for the J1 bench core (src/arcturos/bench.py).

The HTTP layer is mocked with ``httpx.MockTransport`` so no real server is
needed; storage round trips run against a temp SQLite database initialized via
the db.py helpers (``connect`` + ``init_db``).
"""

import json
import subprocess
import uuid

import httpx
import pytest

from arcturos import bench
from arcturos import db as arcturos_db

# Synthetic tokenizer: unit_text * 50 tokenizes to tokens_per_copy * 50 tokens,
# i.e. a fixed tokens-per-section of tokens_per_copy. The uuid4 prefix +
# newline are assumed to tokenize to 3 tokens total (2 + 1) — the sizing
# margin (16) is what absorbs that overhead in production.
UUID_OVERHEAD_TOKENS = 3


def _tokenize_handler(unit_text: str, tokens_per_copy: int):
    """MockTransport handler whose /tokenize returns a fixed token count.

    Asserts the sizing body is exactly ``unit_text * 50`` (the cold-sizing
    contract) and rejects anything that is not a /tokenize call.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tokenize"):
            body = json.loads(request.content)
            assert body["content"] == unit_text * 50, (
                f"tokenize must receive unit_text*50, got {len(body['content'])} chars"
            )
            return httpx.Response(
                200, json={"tokens": list(range(1, tokens_per_copy * 50 + 1))}
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return handler


def _completion_response(
    *, prompt_tokens=512, predicted=32, prefill_tps=170.0, decode_tps=20.0,
    stop_reason="limit", draft_n=None, draft_n_accepted=None,
) -> dict:
    timings = {
        "prompt_n": prompt_tokens,
        "prompt_ms": 3000.0,
        "prompt_per_second": prefill_tps,
        "predicted_n": predicted,
        "predicted_ms": 1600.0,
        "predicted_per_second": decode_tps,
    }
    if draft_n is not None:
        timings["draft_n"] = draft_n
        timings["draft_n_accepted"] = draft_n_accepted
        timings["draft_n_drafted"] = draft_n
        timings["draft_n_rejected"] = draft_n - draft_n_accepted
    return {
        "content": "x" * predicted,
        "stop_reason": stop_reason,
        "tokens_prompt": prompt_tokens,
        "tokens_predicted": predicted,
        "timings": timings,
    }


def _init_db(tmp_path, name="bench.db"):
    db_path = tmp_path / name
    conn = arcturos_db.connect(db_path)
    arcturos_db.init_db(conn)
    conn.close()
    return db_path


# ------------------------------------------------------------ sizing --------


def test_plan_prompt_sizes_lands_within_one_percent_of_target():
    unit = "x"
    tokens_per_copy = 10  # 50 copies -> 500 tokens -> 10 tokens/section
    targets = [4096, 16384, 65536]
    transport = httpx.MockTransport(_tokenize_handler(unit, tokens_per_copy))

    plans = bench.plan_prompt_sizes(
        targets, "http://server:8000/tokenize", unit, margin=16, transport=transport
    )

    assert [p.target_tokens for p in plans] == targets
    for plan, target in zip(plans, targets):
        # Exact plan math: sections from (target - margin) / tokens_per_section.
        assert plan.sections == int((target - 16) / tokens_per_copy)
        assert plan.sections * tokens_per_copy <= target - 16  # never overshoots
        assert plan.unit_text == unit
        # Full cold prompt (uuid + newline + body) lands within 1% of target.
        prompt = bench.build_cold_prompt(plan)
        total_tokens = plan.sections * tokens_per_copy + UUID_OVERHEAD_TOKENS
        assert abs(total_tokens - target) / target < 0.01


def test_build_cold_prompt_unique_prefix_shared_body():
    plan = bench.PromptPlan(target_tokens=512, sections=3, unit_text="abc ")
    a = bench.build_cold_prompt(plan)
    b = bench.build_cold_prompt(plan)

    # Fresh UUID prefix on every call, valid uuid4, newline-separated.
    assert uuid.UUID(a[:36])
    assert uuid.UUID(b[:36])
    assert a[36] == "\n"
    assert a[:36] != b[:36]

    # Deterministic body after the prefix: identical across calls.
    body = plan.unit_text * plan.sections
    assert a[37:] == b[37:] == body
    assert a != b  # whole prompts differ only by the prefix


def test_plan_prompt_sizes_rejects_bad_tokenize_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 50})  # no "tokens" key

    transport = httpx.MockTransport(handler)
    with pytest.raises(ValueError, match="tokenize response shape"):
        bench.plan_prompt_sizes(
            [512], "http://server:8000/tokenize", "x", transport=transport
        )


# ------------------------------------------------------- measurement --------


def test_run_benchmark_point_parses_timings():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completion")
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["cache_prompt"] is False
        assert payload["n_predict"] == 32
        # Cold prompt structure: uuid4 prefix + newline + 5 copies of the body.
        assert uuid.UUID(payload["prompt"][:36])
        assert payload["prompt"][36] == "\n"
        assert payload["prompt"].count("x") == 5
        return httpx.Response(200, json=_completion_response(draft_n=3, draft_n_accepted=2))

    transport = httpx.MockTransport(handler)
    plan = bench.PromptPlan(target_tokens=512, sections=5, unit_text="x")
    point = bench.run_benchmark_point(
        "http://server:8000", "ds4-flash", plan, n_predict=32, timeout=10,
        transport=transport,
    )

    assert point.target_tokens == 512
    assert point.prefill_tps == pytest.approx(170.0)
    assert point.decode_tps == pytest.approx(20.0)
    assert point.mtp_draft_n == 3
    assert point.mtp_accepted == 2
    assert point.output_tokens == 32
    assert point.stop_reason == "limit"
    assert point.prompt_tokens == 512
    assert point.ttft_ms is None  # non-stream native response exposes no TTFT
    assert point.wall_s is not None and point.wall_s > 0
    assert point.power_watts is None


def test_run_benchmark_point_missing_draft_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "eos",
                "tokens_predicted": 7,
                "timings": {"prompt_per_second": 100.0, "predicted_per_second": 25.0},
            },
        )

    transport = httpx.MockTransport(handler)
    plan = bench.PromptPlan(target_tokens=256, sections=3, unit_text="x")
    point = bench.run_benchmark_point(
        "http://server:8000", "ds4-flash", plan, n_predict=7, transport=transport
    )

    assert point.mtp_draft_n is None
    assert point.mtp_accepted is None
    assert point.prompt_tokens is None  # response had no prompt token count
    assert point.decode_tps == pytest.approx(25.0)
    assert point.prefill_tps == pytest.approx(100.0)
    assert point.stop_reason == "eos"


def test_run_benchmark_point_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    transport = httpx.MockTransport(handler)
    plan = bench.PromptPlan(target_tokens=512, sections=5, unit_text="x")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        bench.run_benchmark_point(
            "http://server:8000", "ds4-flash", plan, 16, transport=transport
        )


def test_run_benchmark_sequential_order_and_on_point():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tokenize"):
            events.append("tokenize")
            return httpx.Response(200, json={"tokens": [1] * 50})  # 1 token/section
        if request.url.path.endswith("/completion"):
            payload = json.loads(request.content)
            body_copies = payload["prompt"].count("x")
            events.append(f"completion:{body_copies}")
            return httpx.Response(
                200,
                json=_completion_response(
                    prompt_tokens=body_copies, predicted=16, prefill_tps=200.0,
                    decode_tps=30.0,
                ),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    seen: list[bench.BenchPoint] = []
    points = bench.run_benchmark(
        "http://server:8000", "ds4", targets=[256, 512], n_predict=16,
        timeout=10, on_point=seen.append, unit_text="x", transport=transport,
    )

    # Sizing first (one tokenize call), then completions in target order:
    # 256 -> sections = (256-16)/1 = 240 copies; 512 -> 496 copies.
    assert events == ["tokenize", "completion:240", "completion:496"]
    assert [p.target_tokens for p in points] == [256, 512]
    assert seen == points  # on_point fired once per point, in order
    assert all(p.decode_tps == pytest.approx(30.0) for p in points)


# ------------------------------------------------------- engine meta --------


def test_capture_engine_metadata_happy():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/props")
        return httpx.Response(
            200,
            json={
                "n_ctx": 262144,
                "model_path": "/m/models/DeepSeek-V4-Flash.gguf",
                "build": "b4321 (deadbeef)",
                "engine": "llama.cpp",
                "system_info": "AVX2",
            },
        )

    transport = httpx.MockTransport(handler)
    meta = bench.capture_engine_metadata("http://server:8000", transport=transport)
    assert meta["n_ctx"] == 262144
    assert meta["model_path"].endswith(".gguf")
    assert meta["build"] == "b4321 (deadbeef)"
    assert meta["engine"] == "llama.cpp"


def test_capture_engine_metadata_missing_endpoint_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    meta = bench.capture_engine_metadata("http://server:8000", transport=transport)
    assert meta == {
        "n_ctx": None, "model_path": None, "build": None,
        "engine": None, "system_info": None,
        "model_fingerprint": None,
    }


def test_capture_engine_metadata_unreachable_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    meta = bench.capture_engine_metadata("http://server:8000", transport=transport)
    assert all(value is None for value in meta.values())


# ------------------------------------------------------------- power --------


def test_power_draw_avg_graceful_none_when_nvidia_smi_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    assert bench.power_draw_avg(0, 1) is None


def test_power_draw_avg_returns_mean(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, _seconds):
            self.now += 1.0

    fake = FakeClock()
    monkeypatch.setattr(bench.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(bench.time, "sleep", fake.sleep)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="250.00\n", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    assert bench.power_draw_avg(0, 3) == pytest.approx(250.0)


# ---------------------------------------------------------- storage --------


def test_store_benchmark_run_roundtrip(tmp_path):
    db_path = _init_db(tmp_path)
    points = [
        bench.BenchPoint(
            target_tokens=4096, prefill_tps=170.0, decode_tps=20.0, wall_s=12.4,
            output_tokens=250, mtp_draft_n=3, mtp_accepted=2,
            stop_reason="limit", prompt_tokens=4096, power_watts=250.0,
        ),
        bench.BenchPoint(
            target_tokens=8192, prefill_tps=160.0, decode_tps=19.5, wall_s=25.0,
            output_tokens=200, stop_reason="eos",
        ),
    ]
    meta = {
        "n_ctx": 262144,
        "model_path": "/models/DeepSeek-V4-Flash-Q4_K_M.gguf",
        "build": "b4321",
    }

    run_id = bench.store_benchmark_run(
        str(db_path), "http://fixt-host-a:8000", meta, points
    )
    assert isinstance(run_id, int) and run_id >= 1

    out = bench.export_run_json(db_path, run_id)
    assert out["run"]["server_url"] == "http://fixt-host-a:8000"
    assert out["run"]["model_fingerprint"] == meta["model_path"]
    assert out["run"]["engine"] == "llama.cpp"
    assert out["run"]["context_size"] == 262144
    assert len(out["benchmarks"]) == 2

    first, second = out["benchmarks"]
    assert first["context_tokens"] == 4096
    assert first["prefill_tps"] == pytest.approx(170.0)
    assert first["mtp_draft_n"] == 3
    assert first["mtp_accepted"] == 2
    assert first["power_watts"] == pytest.approx(250.0)
    assert first["ttft_ms"] is None
    assert first["created_at"]  # timestamp present
    assert second["context_tokens"] == 8192
    assert second["mtp_accepted"] is None  # absent metrics stored as NULL
    assert second["power_watts"] is None


def test_store_benchmark_run_unknown_metadata_fallback(tmp_path):
    db_path = _init_db(tmp_path)
    run_id = bench.store_benchmark_run(
        str(db_path), "http://server:8000", {}, [bench.BenchPoint(target_tokens=1024)]
    )
    out = bench.export_run_json(db_path, run_id)
    assert out["run"]["model_fingerprint"] == "unknown"
    # runs.context_size is NOT NULL in the schema: 0 is the "unknown" sentinel.
    assert out["run"]["context_size"] == 0
    assert out["benchmarks"][0]["context_tokens"] == 1024


def test_export_run_json_missing_run(tmp_path):
    db_path = _init_db(tmp_path)
    with pytest.raises(ValueError, match="run 999 not found"):
        bench.export_run_json(db_path, 999)


def test_export_run_json_404_style_error(tmp_path):
    db_path = _init_db(tmp_path)
    try:
        bench.export_run_json(db_path, 999)
    except bench.RunNotFoundError as exc:
        assert exc.status_code == 404
    else:
        pytest.fail("expected RunNotFoundError")


# ------------------------------------------------- combined throughput ----

def _mk_point(target=128, prefill=400.0, decode=50.0, wall=1.0, out=8):
    from arcturos.bench import BenchPoint
    return BenchPoint(target_tokens=target, prefill_tps=prefill,
                      decode_tps=decode, wall_s=wall, output_tokens=out)


def test_aggregate_combined_is_sum_of_stream_rates():
    """Combined = SUM across streams (true server capacity), not mean."""
    from arcturos.bench import _aggregate_stream_points
    pts = [_mk_point(prefill=400.0, decode=50.0),
           _mk_point(prefill=380.0, decode=48.0),
           _mk_point(prefill=420.0, decode=52.0)]
    agg = _aggregate_stream_points(pts, 3)
    assert agg.decode_tps_combined == 150.0    # 50 + 48 + 52
    assert agg.prefill_tps_combined == 1200.0  # 400 + 380 + 420
    import pytest as _pytest
    assert agg.decode_tps == _pytest.approx(50.0)   # mean still tracked
    assert agg.prefill_tps == _pytest.approx(400.0)
    assert agg.streams == 3


def test_aggregate_combined_skips_null_streams():
    """A stream reporting None for a rate does not zero the combined sum."""
    from arcturos.bench import _aggregate_stream_points
    pts = [_mk_point(decode=50.0), _mk_point(decode=None),
           _mk_point(decode=46.0)]
    agg = _aggregate_stream_points(pts, 3)
    assert agg.decode_tps_combined == 96.0
    assert agg.decode_tps == 48.0


def test_aggregate_single_stream_combined_equals_rate():
    """streams=1 aggregation: combined == the single stream's rate."""
    from arcturos.bench import _aggregate_stream_points
    agg = _aggregate_stream_points([_mk_point(decode=50.0, prefill=400.0)], 1)
    assert agg.decode_tps_combined == 50.0
    assert agg.prefill_tps_combined == 400.0


# ------------------- combined throughput under queueing (ADR 004) ----------

def _mk_ttft_point(target=32768, prefill=300.0, decode=40.0, wall=110.0,
                   out=8, ttft_ms=105000.0, prompt=32231):
    from arcturos.bench import BenchPoint
    return BenchPoint(target_tokens=target, prefill_tps=prefill,
                      decode_tps=decode, wall_s=wall, output_tokens=out,
                      ttft_ms=ttft_ms, prompt_tokens=prompt)


def test_combined_prefill_uses_wall_window_not_rate_sum_when_queued():
    """Regression (run #103): streams queued behind a server batch limit
    must NOT have their per-stream rates summed. Two streams prefilled
    cold at ~300 T/s (TTFT ~105 s) and two hit the prefix cache with
    TTFT ~= queue time; summing rates claimed 1222 T/s of capacity the
    server never had. Correct answer: total prompt tokens / max TTFT."""
    from arcturos.bench import _aggregate_stream_points
    pts = [
        _mk_ttft_point(ttft_ms=105000.0),   # cold stream 1
        _mk_ttft_point(ttft_ms=105000.0),   # cold stream 2
        _mk_ttft_point(ttft_ms=209000.0, prefill=160000.0),  # cached, queued
        _mk_ttft_point(ttft_ms=209000.0, prefill=160000.0),  # cached, queued
    ]
    agg = _aggregate_stream_points(pts, 4)
    # 4 * 32231 prompt tokens over the max TTFT (209 s) — the window in
    # which every stream's prefill completed. NOT 300+300+160000+160000.
    assert agg.prefill_tps_combined == round(4 * 32231 / 209.0, 1)
    assert (agg.prefill_tps_combined or 0) < 1300  # old math: ~320k here


def test_combined_decode_uses_generation_window():
    """Decode combined = total generated tokens / (max wall - min TTFT),
    the window in which generation actually overlapped."""
    from arcturos.bench import _aggregate_stream_points
    pts = [
        _mk_ttft_point(wall=110.0, out=8, ttft_ms=105000.0),
        _mk_ttft_point(wall=110.0, out=8, ttft_ms=105000.0),
        _mk_ttft_point(wall=211.0, out=8, ttft_ms=209000.0),
        _mk_ttft_point(wall=211.0, out=8, ttft_ms=209000.0),
    ]
    agg = _aggregate_stream_points(pts, 4)
    # 32 tokens over (211 - 105) s
    assert agg.decode_tps_combined == round(32 / (211.0 - 105.0), 2)


def test_combined_falls_back_to_rate_sum_without_ttft():
    """Native non-stream points have no TTFT: keep the old rate-sum
    behaviour (byte-compatible with pre-ADR-004 native rows)."""
    from arcturos.bench import _aggregate_stream_points
    pts = [_mk_point(prefill=400.0, decode=50.0),
           _mk_point(prefill=380.0, decode=48.0)]
    agg = _aggregate_stream_points(pts, 2)
    assert agg.prefill_tps_combined == 780.0
    assert agg.decode_tps_combined == 98.0


def test_combined_concurrent_streams_match_rate_sum():
    """When all streams run truly concurrently (equal TTFT/wall), the
    wall-window math reduces to the rate sum — no regression for
    well-behaved servers."""
    from arcturos.bench import _aggregate_stream_points
    pts = [
        _mk_ttft_point(prefill=400.0, decode=50.0, wall=90.0, out=8,
                       ttft_ms=80000.0, prompt=32000),
        _mk_ttft_point(prefill=400.0, decode=50.0, wall=90.0, out=8,
                       ttft_ms=80000.0, prompt=32000),
    ]
    agg = _aggregate_stream_points(pts, 2)
    assert agg.prefill_tps_combined == round(64000 / 80.0, 1)  # = 400+400
    assert agg.decode_tps_combined == round(16 / (90.0 - 80.0), 2)  # = 50+50
