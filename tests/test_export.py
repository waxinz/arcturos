"""§4.6 per-view CSV export endpoints.

Every CSV row carries schema_version; missing values are empty cells
(honest nulls, never fabricated zeros); 404s are specific.
"""
import csv
import io

import pytest


@pytest.fixture()
def seeded_run(client):
    """One run, so the export endpoints have something to export."""
    r = client.post("/api/runs", json={
        "server_url": "http://10.10.10.122:8000",
        "model_fingerprint": "/models/ds4-flash-q8.gguf",
        "engine": "llama.cpp", "context_size": 262144})
    assert r.status_code == 201, r.text
    return r.json()


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_export_runs_csv(client, seeded_run):
    r = client.get("/api/export/runs")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    rows = _rows(r.text)
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "arcturos-v1"
    assert rows[0]["model_fingerprint"] == "/models/ds4-flash-q8.gguf"
    assert rows[0]["run_id"] == str(seeded_run["id"])


def test_export_run_benchmarks_csv(client, seeded_run):
    client.post("/api/runs/1/benchmarks",
                json={"context_tokens": 4096, "decode_tps": 300.0})
    client.post("/api/runs/1/benchmarks",
                json={"context_tokens": 16384})  # no metrics: honest nulls
    r = client.get("/api/export/runs/1/benchmarks")
    assert r.status_code == 200
    rows = _rows(r.text)
    assert len(rows) == 2
    assert rows[0]["decode_tps"] == "300.0"
    # a point without the metric is an empty cell, never 0
    assert rows[1]["decode_tps"] == ""
    assert rows[0]["model_fingerprint"] == "/models/ds4-flash-q8.gguf"


def test_export_run_benchmarks_404(client):
    r = client.get("/api/export/runs/999/benchmarks")
    assert r.status_code == 404


def test_export_eval_results_csv(client):
    suite = client.post("/api/eval-suites",
                        json={"name": "smoke", "version": "1"}).json()
    client.post(f"/api/eval-suites/{suite['id']}/results", json={
        "model_fingerprint": "/m/ds4.gguf", "item_id": "q1",
        "output": "hello", "prompt_tokens": 10, "completion_tokens": 5,
        "latency_ms": 50.0})
    r = client.get("/api/export/eval-results")
    assert r.status_code == 200
    rows = _rows(r.text)
    assert len(rows) == 1
    assert rows[0]["suite_name"] == "smoke"
    assert rows[0]["suite_version"] == "1"
    assert rows[0]["output"] == "hello"
    # filtered by suite
    r = client.get(f"/api/export/eval-results?suite_id={suite['id']}")
    assert len(_rows(r.text)) == 1
    r = client.get("/api/export/eval-results?suite_id=999")
    assert r.status_code == 404


def test_export_judgments_csv_resolves_fingerprints(client):
    suite = client.post("/api/eval-suites",
                        json={"name": "s", "version": "1"}).json()
    ids = []
    for i in range(2):
        r = client.post(f"/api/eval-suites/{suite['id']}/results", json={
            "model_fingerprint": f"/m/model-{i}.gguf", "item_id": "q1",
            "output": "x", "prompt_tokens": 10, "completion_tokens": 5,
            "latency_ms": 50.0})
        ids.append(r.json()["id"])
    client.post("/api/judgments", json={
        "eval_result_a": ids[0], "eval_result_b": ids[1],
        "judge_model": "judge-1", "judge_template_version": "blind-v1",
        "winner": "a"})
    r = client.get("/api/export/judgments")
    assert r.status_code == 200
    rows = _rows(r.text)
    assert len(rows) == 1
    assert rows[0]["fp_a"] == "/m/model-0.gguf"
    assert rows[0]["fp_b"] == "/m/model-1.gguf"
    assert rows[0]["winner"] == "a"
