"""API tests for the CRUD/dispatch endpoints (/api/ops/*) + create view."""

import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@pytest.fixture()
def client(tmp_path):
    from arcturos.main import create_app
    app = create_app(tmp_path / "ops_api.db")
    return __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)


def test_create_view_served(client):
    r = client.get("/create")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_create_view_live_table_single_stream_stays_compact(client):
    """Default (single-stream) live table keeps the compact 7-column head —
    Σ columns only appear when a job actually runs parallel streams."""
    html = client.get("/create").text
    assert "<th>ttft (s)</th><th>wall</th><th>streams</th>" in html
    # the static default head (server-rendered) has no Σ columns
    assert "Σ prefill" not in html.split('id="bp-head"')[1].split("</tr>")[0]


def test_create_view_live_table_adapts_for_multi_stream_jobs(client):
    """The adaptive-column logic exists: kick-off reads the streams field,
    bpColumns rewrites the head with Σ prefill/decode for streams > 1, and
    bpRow renders the combined cells for multi-stream jobs."""
    html = client.get("/create").text
    assert "bpColumns" in html
    assert "Σ prefill</th><th>Σ decode</th>" in html
    assert "bpPoll.streams = parseInt($('bench-streams').value, 10) || 1" in html
    assert "p.prefill_tps_combined" in html
    assert "p.decode_tps_combined" in html


def test_run_detail_view_served(client):
    r = client.get("/runs/1")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_ops_bench_streams_out_of_bounds_is_422(client):
    r = client.post("/api/ops/bench", json={
        "server_url": "http://fake:8000", "targets": [32],
        "n_predict": 4, "streams": 17})
    assert r.status_code == 422
    assert "streams" in r.json()["detail"]
    r0 = client.post("/api/ops/bench", json={
        "server_url": "http://fake:8000", "targets": [32],
        "n_predict": 4, "streams": 0})
    assert r0.status_code == 422


def test_ops_bench_validation(client):
    assert client.post("/api/ops/bench", json={"server_url": "ftp://x", "targets": [1]}).status_code == 422
    assert client.post("/api/ops/bench", json={"server_url": "http://x:8000", "targets": ["a"]}).status_code == 422
    assert client.post("/api/ops/bench", json={"server_url": "http://x:8000", "targets": [1], "n_predict": 0}).status_code == 422


def test_ops_bench_roundtrip(client, monkeypatch):
    """Full dispatch through the API: bench runs, stores run + points."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(100))})
        if request.url.path == "/completion":
            return httpx.Response(200, json={
                "tokens_predicted": 4, "tokens_evaluated": 2,
                "stop_reason": "eos",
                "timings": {"prompt_per_second": 500.0,
                            "predicted_per_second": 30.0}})
        if request.url.path == "/props":
            return httpx.Response(200, json={"n_ctx": 2048,
                                             "model_path": "/m/t.gguf"})
        return httpx.Response(404)

    # main.py calls ops.dispatch_bench without transport; give the real
    # function a MockTransport via its keyword, without replacing the
    # function itself (that's what caused infinite recursion before).
    real_dispatch = __import__("arcturos.ops", fromlist=["dispatch_bench"]).dispatch_bench

    def fake_dispatch(**kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_dispatch(**kwargs)

    import arcturos.main as main_mod
    monkeypatch.setattr(main_mod.ops, "dispatch_bench", fake_dispatch)
    r = client.post("/api/ops/bench", json={
        "server_url": "http://fake:8000", "targets": [32], "n_predict": 4})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["run_id"] == 1
    assert data["points"][0]["decode_tps"] == 30.0
    # visible through the normal runs API
    assert client.get("/api/runs").json()[0]["id"] == 1
    assert len(client.get("/api/runs/1/benchmarks").json()) == 1


def test_ops_eval_validation(client):
    assert client.post("/api/ops/eval", json={"suite_id": 0}).status_code == 422
    assert client.post("/api/ops/eval", json={
        "suite_id": 1, "target": "http://x/v1", "model_fingerprint": "m"}).status_code == 422
    assert client.post("/api/ops/eval", json={
        "suite_id": 1, "target": "http://x/v1", "model_fingerprint": "m",
        "suite": {}}).status_code == 422


def test_ops_eval_requires_existing_suite(client):
    r = client.post("/api/ops/eval", json={
        "suite_id": 999, "target": "http://x/v1", "model_fingerprint": "m",
        "suite": {"items": [{"id": "q1", "prompt": "hi"}]}})
    # dispatch validates suite existence BEFORE calling the target -> 422
    assert r.status_code == 422
    assert "not found" in r.json()["detail"]


def test_ops_eval_roundtrip(client, monkeypatch):
    suite = client.post("/api/eval-suites", json={"name": "s", "version": "1"}).json()
    # preflight would hit real DNS for the fake target — stub it passing
    from arcturos.ops import preflight as pf_mod
    monkeypatch.setattr(pf_mod, "preflight", lambda *a, **kw: pf_mod.PreflightResult(
        target=a[0], kind="eval",
        checks=[{"name": "reachable", "ok": True, "detail": "stub"},
                {"name": "chat", "ok": True, "detail": "stub"}]))
    with patch("arcturos.multiturn.post_openai_chat") as fake_post:
        fake_post.return_value = {
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        r = client.post("/api/ops/eval", json={
            "suite_id": suite["id"], "target": "http://fake:4000/v1",
            "model_fingerprint": "model-x",
            "suite": {"items": [{"id": "q1", "prompt": "2+2?"}]}})
    assert r.status_code == 200, r.text
    assert len(r.json()["stored"]) == 1
    stored = r.json()["stored"][0]
    assert stored["item_id"] == "q1"
    # visible through the evals API
    res = client.get(f"/api/eval-suites/{suite['id']}/results").json()
    assert len(res) == 1


def test_power_provenance_validation(client):
    """ADR 002: power_watts without power_host is rejected (422)."""
    run = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000/v1",
        "model_fingerprint": "m", "engine": "llama.cpp",
        "context_size": 262144}).json()
    r = client.post(f"/api/runs/{run['id']}/benchmarks", json={
        "context_tokens": 4096, "power_watts": 134.0})
    assert r.status_code == 422
    ok = client.post(f"/api/runs/{run['id']}/benchmarks", json={
        "context_tokens": 4096, "power_watts": 134.0,
        "power_host": "alexei@10.10.10.122", "power_gpu_index": 0})
    assert ok.status_code == 201
    assert ok.json()["power_host"] == "alexei@10.10.10.122"



# ------------------------------------------ REVIEW-FIX-ROUND regressions ----


def test_hidden_run_rejected_by_baseline_deltas(client):
    """Soft-hide gates every compare surface: a hidden run is 404 as the
    delta target (parity with run-diff and the series endpoint)."""
    run = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": "/m/h.gguf", "engine": "llama.cpp",
        "context_size": 4096}).json()
    client.post(f"/api/runs/{run['id']}/benchmarks",
                json={"context_tokens": 4096, "decode_tps": 30.0})
    client.put(f"/api/runs/{run['id']}/visibility", json={"hidden": True})
    r = client.get(
        f"/api/compare/baseline-deltas?runs={run['id']}&run={run['id']}"
        f"&metrics=decode_tps&baseline={run['id']}")
    assert r.status_code == 404
    assert "hidden" in r.json()["detail"]


def test_hidden_run_rejected_as_baseline_reference(client):
    """A hidden run cannot serve as the baseline= reference either."""
    a = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": "/m/a.gguf", "engine": "llama.cpp",
        "context_size": 4096}).json()
    b = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": "/m/b.gguf", "engine": "llama.cpp",
        "context_size": 4096}).json()
    client.post(f"/api/runs/{a['id']}/benchmarks",
                json={"context_tokens": 4096, "decode_tps": 30.0})
    client.put(f"/api/runs/{b['id']}/visibility", json={"hidden": True})
    r = client.get(
        f"/api/compare/benchmarks?runs={a['id']}&metrics=decode_tps"
        f"&baseline={b['id']}")
    assert r.status_code == 404
    assert "hidden" in r.json()["detail"]


def test_create_view_ttft_column_shows_seconds(client):
    """TTFT displays in seconds (2026-09-22 UX rule): live-table headers
    say 'ttft (s)' and the row renderer divides stored ms by 1000."""
    html = client.get("/create").text
    assert "<th>ttft (s)</th>" in html
    assert "secs(p.ttft_ms)" in html          # ms -> s at render time
    assert "cell(p.ttft_ms, ' ms')" not in html


def test_create_view_detail_table_scroll_container(client):
    """The live-job detail table lives in a horizontal-scroll wrapper so
    wide tables grow the panel instead of clipping under the next card."""
    html = client.get("/create").text
    assert 'class="table-scroll"' in html
    assert ".bp-detail .table-scroll { overflow-x: auto; }" in html
    assert "min-width: 480px" in html


def test_runs_view_name_column(client):
    """/runs page renders the Name column header (second column) and the
    row template reads r.name with an honest '—' fallback."""
    html = client.get("/runs").text
    assert "<th>Name</th>" in html
    assert "r.name || '—'" in html


def test_compare_view_legend_prefers_run_name(client):
    """compare.html legendName() prefers r.name over the model label."""
    html = client.get("/compare").text
    assert "function legendName" in html
    assert "r.name" in html.split("function legendName")[1].split("}")[0]


def test_diff_view_headers_use_run_name(client):
    """diff.html 'Metrics — A vs B' headers prefer run names."""
    html = client.get("/diff").text
    assert "data.run_a.name || data.run_a.model_fingerprint_short" in html


def test_bench_job_endpoint_validation_errors(client):
    """POST /api/ops/bench/jobs validates synchronously: bad shapes are
    422 with actionable detail, before any thread starts."""
    bad_bodies = [
        ({}, "server_url"),
        ({"server_url": "ftp://x"}, "server_url"),
        ({"server_url": "http://x", "targets": "nope"}, "targets"),
        ({"server_url": "http://x", "targets": [0]}, "targets"),
        ({"server_url": "http://x", "targets": [512], "n_predict": 0}, "n_predict"),
        ({"server_url": "http://x", "targets": [512], "transport": "grpc"}, "transport"),
        ({"server_url": "http://x", "targets": [512], "streams": 0}, "streams"),
        ({"server_url": "http://x", "targets": [512], "streams": True}, "streams"),
        ({"server_url": "http://x", "targets": [512], "power_host": ""}, "power_host"),
        ({"server_url": "http://x", "targets": [512], "power_gpu_index": -1}, "power_gpu_index"),
        ({"server_url": "http://x", "targets": [512], "name": ""}, "name"),
    ]
    for body, needle in bad_bodies:
        r = client.post("/api/ops/bench/jobs", json=body)
        assert r.status_code == 422, (body, r.status_code)
        assert needle in r.json()["detail"]


def test_bench_job_status_unknown_id_404(client):
    r = client.get("/api/ops/bench/jobs/does-not-exist")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_eval_dispatch_validation_errors(client):
    """POST /api/ops/eval rejects bad shapes with 422 before touching
    the target: suite_id, target URL, fingerprint, suite payload, name."""
    base = {"suite_id": 1, "target": "http://10.10.10.14:8080",
            "model_fingerprint": "m", "suite": {"items": []}}
    r = client.post("/api/ops/eval", json={**base, "suite_id": "one"})
    assert r.status_code == 422 and "suite_id" in r.json()["detail"]
    r = client.post("/api/ops/eval", json={**base, "target": "ftp://x"})
    assert r.status_code == 422 and "target" in r.json()["detail"]
    r = client.post("/api/ops/eval", json={**base, "model_fingerprint": ""})
    assert r.status_code == 422 and "model_fingerprint" in r.json()["detail"]
    r = client.post("/api/ops/eval", json={**base, "suite": "nope"})
    assert r.status_code == 422 and "suite" in r.json()["detail"]
    r = client.post("/api/ops/eval", json={**base, "name": ""})
    assert r.status_code == 422 and "name" in r.json()["detail"]


def test_preflight_endpoints_validation_errors(client):
    """Both preflight surfaces reject bad target/kind with 422."""
    r = client.post("/api/ops/preflight", json={"target": "ftp://x"})
    assert r.status_code == 422 and "target" in r.json()["detail"]
    r = client.post("/api/ops/preflight",
                    json={"target": "http://x", "kind": "nope"})
    assert r.status_code == 422 and "kind" in r.json()["detail"]
    r = client.get("/api/ops/preflight", params={"target": "no-scheme"})
    assert r.status_code == 422 and "target" in r.json()["detail"]
    r = client.get("/api/ops/preflight",
                   params={"target": "http://x", "kind": "nope"})
    assert r.status_code == 422 and "kind" in r.json()["detail"]
