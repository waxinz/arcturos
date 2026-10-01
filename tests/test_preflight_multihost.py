"""J6 health preflight + multi-host compare metadata tests."""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from arcturos.preflight import preflight  # noqa: E402


def _transport(health=200, tokenize=200, tokenize_body=None, chat=200):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(health, json={"status": "ok"})
        if path == "/tokenize":
            return httpx.Response(tokenize, json=tokenize_body or {"tokens": [1, 2, 3]})
        if path == "/v1/chat/completions":
            return httpx.Response(chat, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_preflight_bench_happy():
    pf = preflight("http://fake:8000", kind="bench", transport=_transport())
    assert pf.ok()
    assert [c["name"] for c in pf["checks"]] == ["reachable", "tokenize"]


def test_preflight_eval_happy_with_model():
    pf = preflight("http://fake:4000/v1", kind="eval",
                   model="qwen", transport=_transport())
    assert pf.ok()
    assert pf["checks"][1]["name"] == "chat"


def test_preflight_unreachable():
    # /health 404 -> warning (server answered), NOT a gate failure
    pf = preflight("http://fake:8000", kind="bench",
                   transport=_transport(health=404))
    assert pf.ok()
    assert "404" in pf["checks"][0]["detail"]
    assert "non-blocking" in pf["checks"][0]["detail"]


def test_preflight_health_503_tabbyapi_style_is_nonblocking():
    """tabbyAPI /health aggregates unrelated internal issues into its
    status (e.g. a stale 'IndexError' entry) and returns 503 while
    inference works fine — dispatch must not be blocked by it."""
    pf = preflight("http://fake:8000", kind="eval",
                   transport=_transport(health=503))
    assert pf.ok()
    assert "503" in pf["checks"][0]["detail"]
    assert "non-blocking" in pf["checks"][0]["detail"]


def test_preflight_tokenize_missing_on_eval_target():
    # eval targets don't serve native /tokenize -> bench preflight fails
    pf = preflight("http://fake:4000/v1", kind="bench",
                   transport=_transport(tokenize=404))
    assert not pf.ok()
    assert "native llama.cpp endpoint missing" in pf["checks"][1]["detail"]


def test_preflight_chat_model_hint():
    pf = preflight("http://fake:4000/v1", kind="eval", model="wrong-name",
                   transport=_transport(chat=400))
    assert not pf.ok()
    assert "model name may be wrong" in pf["checks"][1]["detail"]


def test_preflight_bad_kind():
    with pytest.raises(ValueError, match="unknown preflight kind"):
        preflight("http://fake:8000", kind="surgery")


@pytest.fixture()
def client(tmp_path):
    from arcturos.main import create_app
    app = create_app(tmp_path / "pf.db")
    return TestClient(app)


def test_preflight_endpoint_validation(client):
    # POST form: key travels in the JSON body, never a URL (UX review).
    assert client.post("/api/ops/preflight", json={"target": "ftp://x"}).status_code == 422
    assert client.post("/api/ops/preflight",
                       json={"target": "http://x", "kind": "nope"}).status_code == 422


def test_preflight_endpoint_live(client, monkeypatch):
    monkeypatch.setattr("arcturos.ops.preflight.preflight",
                        lambda *a, **kw: {"target": kw.get("target", a[0] if a else ""),
                                          "kind": kw.get("kind", "bench"),
                                          "checks": [{"name": "reachable", "ok": True,
                                                      "detail": "GET /health -> 200"}]})
    r = client.post("/api/ops/preflight",
                    json={"target": "http://fake:8000", "kind": "bench"})
    assert r.status_code == 200
    assert r.json()["checks"][0]["ok"] is True


def test_preflight_endpoint_key_never_in_url(client, monkeypatch):
    """api_key goes via POST body; GET probes are keyless by contract."""
    seen = {}

    def fake_pf(target, kind="bench", model=None, transport=None,
                timeout=None, api_key=None):
        seen["api_key"] = api_key
        return {"target": target, "kind": kind, "checks": [
            {"name": "reachable", "ok": True, "detail": "stub"}]}

    monkeypatch.setattr("arcturos.ops.preflight.preflight", fake_pf)
    # GET with a key param must NOT receive it (keyless fallback endpoint)
    client.get("/api/ops/preflight",
               params={"target": "http://fake:8000", "api_key": "sk-leak"})
    assert seen.get("api_key") is None
    r = client.post("/api/ops/preflight",
                    json={"target": "http://fake:8000", "api_key": "sk-ok"})
    assert r.status_code == 200
    assert seen["api_key"] == "sk-ok"


def test_preflight_endpoint_dead_target(client):
    """Unroutable target -> 200 with honest failed checks (not an error
    status): the UI shows WHY the target is down. 502 is reserved for
    endpoint-level failures (e.g. a bad transport config)."""
    r = client.post("/api/ops/preflight",
                    json={"target": "http://127.0.0.1:1", "kind": "bench"})
    assert r.status_code == 200
    checks = r.json()["checks"]
    assert all(c["ok"] is False for c in checks)
    assert any("failed" in c["detail"] or "timed out" in c["detail"]
               for c in checks)


def test_compare_payload_carries_host_metadata(client):
    """Multi-host compare: server_url + host_label present per run."""
    run = client.post("/api/runs", json={
        "server_url": "http://fixt-host-a:8000/v1",
        "model_fingerprint": "/models/ds4.gguf", "engine": "llama.cpp",
        "context_size": 262144}).json()
    client.post(f"/api/runs/{run['id']}/benchmarks", json={"context_tokens": 4096,
                                                           "decode_tps": 30.0})
    data = client.get("/api/compare/benchmarks",
                      params={"runs": run["id"]}).json()
    r = data["runs"][0]
    assert r["server_url"] == "http://fixt-host-a:8000/v1"
    assert r["host_label"] == "fixt-host-a"
    assert r["model_fingerprint"] == "/models/ds4.gguf"


def test_run_diff_metadata_includes_host(client):
    """Two runs on different hosts -> host diff shows both."""
    ra = client.post("/api/runs", json={
        "server_url": "http://fixt-host-a:8000", "model_fingerprint": "m",
        "engine": "llama.cpp", "context_size": 4096}).json()
    rb = client.post("/api/runs", json={
        "server_url": "http://fixt-host-b:8000", "model_fingerprint": "m",
        "engine": "llama.cpp", "context_size": 4096}).json()
    for rid in (ra["id"], rb["id"]):
        client.post(f"/api/runs/{rid}/benchmarks", json={"context_tokens": 4096,
                                                         "decode_tps": 30.0})
    data = client.get(f"/api/compare/run-diff/{ra['id']}/{rb['id']}").json()
    host_diff = data["engine_metadata"]["host"]
    assert host_diff["a"] == "fixt-host-a"
    assert host_diff["b"] == "fixt-host-b"
    assert host_diff["same"] is False
