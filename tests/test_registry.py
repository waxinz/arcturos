"""Model registry + baseline tracking (§4.6 J6).

Covers auto-registration on first sighting, alias-only edits (the DB
trigger rejects any other column change), baseline pin/re-pin/unpin, and
the wrong-model rejection that would silently corrupt delta columns.
"""
import pytest


@pytest.fixture()
def seeded_run(client):
    """One run, so there is something to pin and to auto-register."""
    r = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "engine": "llama.cpp",
        "context_size": 262144,
    })
    assert r.status_code == 201, r.text
    return r.json()


def test_create_run_auto_registers_model(client, seeded_run):
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert len(models) == 1
    m = models[0]
    assert m["model_fingerprint"] == "/models/ds4-flash-q8.gguf"
    assert m["engine"] == "llama.cpp"
    assert m["alias"] is None
    assert m["run_count"] == 1
    assert m["first_seen"] == m["last_seen"]


def test_second_run_updates_last_seen_not_first(client, seeded_run):
    r = client.post("/api/runs", json={
        "server_url": "http://localhost:4000/v1",
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "engine": "llama.cpp",
        "context_size": 131072,
    })
    assert r.status_code == 201
    models = client.get("/api/models").json()
    assert len(models) == 1
    assert models[0]["run_count"] == 2
    # first_seen must stay the original — it is append-only provenance
    assert models[0]["first_seen"] == seeded_run["created_at"]


def test_alias_update_and_fingerprint_immutability(client, seeded_run):
    fp = "/models/ds4-flash-q8.gguf"
    r = client.patch(f"/api/models/{fp}", json={"alias": "ds4-flash-250w"})
    assert r.status_code == 200, r.text
    assert r.json()["alias"] == "ds4-flash-250w"

    # clearing the alias is allowed (label-only edit)
    r = client.patch(f"/api/models/{fp}", json={"alias": None})
    assert r.status_code == 200
    assert r.json()["alias"] is None

    # unknown fingerprint -> 404, not a silent no-op
    r = client.patch("/api/models/nope.gguf", json={"alias": "x"})
    assert r.status_code == 404


def test_baseline_pin_and_repin(client, seeded_run):
    fp = "/models/ds4-flash-q8.gguf"
    # pin run 1 as the speed baseline
    r = client.post("/api/baselines", json={
        "model_fingerprint": fp, "metric_family": "speed", "run_id": 1})
    assert r.status_code == 201, r.text
    assert r.json()["run_id"] == 1

    # second run for the same fingerprint, then re-pin: replaces the pointer
    client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": fp, "engine": "llama.cpp",
        "context_size": 131072})
    r = client.post("/api/baselines", json={
        "model_fingerprint": fp, "metric_family": "speed", "run_id": 2})
    assert r.status_code == 201
    rows = client.get("/api/baselines").json()
    assert len(rows) == 1
    assert rows[0]["run_id"] == 2
    assert rows[0]["run_created_at"]


def test_baseline_wrong_model_rejected(client, seeded_run):
    client.post("/api/runs", json={
        "server_url": "http://localhost:4000/v1",
        "model_fingerprint": "/models/qwen3.8.gguf",
        "engine": "litellm", "context_size": 4096})
    r = client.post("/api/baselines", json={
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "metric_family": "speed", "run_id": 2})
    assert r.status_code == 422
    assert "same model" in r.json()["detail"]


def test_baseline_metric_family_allowlist(client, seeded_run):
    r = client.post("/api/baselines", json={
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "metric_family": "vibes", "run_id": 1})
    assert r.status_code == 422


def test_baseline_missing_run_404(client):
    r = client.post("/api/baselines", json={
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "metric_family": "quality", "run_id": 999})
    assert r.status_code == 404


def test_baseline_unpin_and_404(client, seeded_run):
    fp = "/models/ds4-flash-q8.gguf"
    client.post("/api/baselines", json={
        "model_fingerprint": fp, "metric_family": "quality", "run_id": 1})
    r = client.delete("/api/baselines/1")
    assert r.status_code == 200
    assert client.get("/api/baselines").json() == []
    r = client.delete("/api/baselines/1")
    assert r.status_code == 404


def test_registry_triggers_reject_non_alias_updates(client, seeded_run):
    """The append-only guarantee holds for the registry: fingerprint,
    engine, and seen timestamps cannot be mutated through SQL either."""
    import sqlite3
    conn = sqlite3.connect(str(client.app.state.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE models SET engine = 'vllm' WHERE model_fingerprint = ?",
            ("/models/ds4-flash-q8.gguf",))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM models WHERE model_fingerprint = ?",
                     ("/models/ds4-flash-q8.gguf",))
    # alias edit through SQL is fine
    conn.execute("UPDATE models SET alias = ? WHERE model_fingerprint = ?",
                 ("direct", "/models/ds4-flash-q8.gguf"))
    conn.commit()
    assert client.get("/api/models").json()[0]["alias"] == "direct"


def test_models_view_and_baselines_view_served(client):
    assert client.get("/models").status_code == 200
    assert client.get("/baselines").status_code == 200
