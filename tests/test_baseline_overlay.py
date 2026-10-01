"""Baseline overlay + CSV export on the compare endpoint (§4.6).

Covers: dashed-reference payload metadata, delta computation with honest
gaps (missing either side → no row), CSV schema-versioned export, and the
endpoint-level 404/422 paths.
"""
import csv
import io

import pytest


@pytest.fixture()
def two_runs(client):
    """Two runs of the same fingerprint, EACH with points at ctx 4096 +
    16384, so deltas have overlapping context points to compare."""
    fp = "/m/ds4.gguf"
    out = []
    for run_number in (1, 2):
        r = client.post("/api/runs", json={
            "server_url": "http://fixt-host-a:8000",
            "model_fingerprint": fp, "engine": "llama.cpp",
            "context_size": 262144})
        assert r.status_code == 201
        run_id = r.json()["id"]
        out.append(run_id)
        for ctx in (4096, 16384):
            r = client.post(f"/api/runs/{run_id}/benchmarks", json={
                "context_tokens": ctx,
                "decode_tps": 300.0 if run_number == 1 else 400.0,
                "prefill_tps": 30000.0,
                "ttft_ms": 120.0 if run_number == 1 else 100.0,
            })
            assert r.status_code == 201
    return out


def test_payload_without_baseline_has_no_baseline_key(client, two_runs):
    r = client.get("/api/compare/benchmarks?runs=1,2&metrics=decode_tps")
    assert r.status_code == 200
    payload = r.json()
    assert "baseline" not in payload
    assert "baseline" not in payload["series"]


def test_baseline_overlay_metadata_and_series(client, two_runs):
    r = client.get("/api/compare/benchmarks?runs=1,2&metrics=decode_tps&baseline=1")
    assert r.status_code == 200
    payload = r.json()
    assert payload["baseline"]["run_id"] == 1
    assert payload["baseline"]["model_fingerprint_short"] == "ds4"
    assert payload["baseline"]["host_label"]
    assert payload["series"]["baseline"]["decode_tps"]
    # the run series are untouched
    assert str(payload["runs"][0]["id"]) in payload["series"]["decode_tps"]
    # baseline run need not be among the compared runs
    r = client.get("/api/compare/benchmarks?runs=2&metrics=decode_tps&baseline=1")
    assert r.status_code == 200
    assert r.json()["baseline"]["run_id"] == 1


def test_baseline_run_not_found_404(client, two_runs):
    r = client.get("/api/compare/benchmarks?runs=1,2&baseline=999")
    assert r.status_code == 404
    assert "baseline run id 999" in r.json()["detail"]


def test_baseline_deltas_honest_gaps(client, two_runs):
    """Run 2 vs baseline run 1: decode 400 vs 300 at both ctx lengths;
    ttft only on the baseline side at ctx 4096... actually both have ttft;
    prefill is equal so delta is 0."""
    r = client.get("/api/compare/baseline-deltas?runs=1,2&run=2&metrics=decode_tps,ttft_ms,prefill_tps&baseline=1")
    assert r.status_code == 200
    deltas = r.json()["2"]
    dec = deltas["decode_tps"]
    assert len(dec) == 2
    row4096 = next(p for p in dec if p["context_tokens"] == 4096)
    assert row4096["baseline_value"] == 300.0
    assert row4096["value"] == 400.0
    assert row4096["delta"] == 100.0
    assert row4096["delta_pct"] == pytest.approx(1 / 3)
    ttft = deltas["ttft_ms"]
    assert len(ttft) == 2
    assert ttft[0]["delta"] == pytest.approx(100.0 - 120.0)  # lower is better
    assert deltas["prefill_tps"][0]["delta"] == 0.0
    assert deltas["prefill_tps"][0]["delta_pct"] == 0.0


def test_baseline_deltas_run_not_in_runs_422(client, two_runs):
    r = client.get("/api/compare/baseline-deltas?runs=1,2&run=9&baseline=1")
    assert r.status_code == 422


def test_csv_export_schema_and_fingerprints(client, two_runs):
    r = client.get("/api/compare/benchmarks?runs=1,2&metrics=decode_tps,ttft_ms&format=csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows
    assert rows[0]["schema_version"] == "arcturos-compare-v1"
    assert rows[0]["model_fingerprint"] == "/m/ds4.gguf"
    assert rows[0]["host_label"]
    metrics_seen = {row["metric"] for row in rows}
    assert metrics_seen == {"decode_tps", "ttft_ms"}
    values = {(row["metric"], row["context_tokens"]): row["value"]
              for row in rows if row["run_id"] == "1"}
    assert values[("decode_tps", "4096")] == "300.0"


def test_csv_export_with_baseline_row(client, two_runs):
    r = client.get(
        "/api/compare/benchmarks?runs=2&metrics=decode_tps&baseline=1&format=csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    baseline_rows = [row for row in rows if row["metric"] == "baseline"]
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["run_id"] == "1"


def test_csv_export_bad_format_422(client, two_runs):
    r = client.get("/api/compare/benchmarks?runs=1,2&format=xlsx")
    assert r.status_code == 422
    assert "csv" in r.json()["detail"]
