"""Integration tests: fixture suites + seed script work against the real API.

These tests exercise J3/J4 contracts using qa/suites/ fixtures via
TestClient — the same code path scripts/seed_demo.py uses in live mode.
Run from repo root: venv/bin/python -m pytest qa/suites/test_j3j4_contract.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

SUITE_PATH = ROOT / "qa" / "suites" / "smoke-reasoning-v1.json"
SEED_SCRIPT = ROOT / "scripts" / "seed_demo.py"


@pytest.fixture()
def client(tmp_path):
    from arcturos.main import create_app
    app = create_app(tmp_path / "arcturos_test.db")
    return TestClient(app)


def load_suite():
    return json.loads(SUITE_PATH.read_text())


def test_fixture_is_well_formed():
    suite = load_suite()
    assert suite["suite"] == "smoke-reasoning"
    ids = [i["id"] for i in suite["items"]]
    assert len(ids) == len(set(ids)), "item ids must be unique"
    for item in suite["items"]:
        if item["type"] == "multi-turn":
            turns = item["turns"]
            assert turns[-1]["role"] == "user", "multi-turn must end on user turn"
            assert any(t["role"] == "assistant" for t in turns), \
                "multi-turn needs an assistant placeholder turn"
        elif item["type"] == "single-turn":
            assert "prompt" in item
        else:
            pytest.fail(f"unknown item type: {item['type']}")


def test_suite_roundtrip_and_judgments(client):
    suite = load_suite()
    r = client.post("/api/eval-suites",
                    json={"name": suite["suite"], "version": suite["version"]})
    assert r.status_code == 201
    suite_id = r.json()["id"]

    # store both model outputs for each single-turn item
    result_ids = {}
    for model in ("model-alpha", "model-beta"):
        for item in suite["items"]:
            if item["type"] != "single-turn":
                continue
            rr = client.post(f"/api/eval-suites/{suite_id}/results", json={
                "model_fingerprint": model,
                "item_id": item["id"],
                "output": f"{model} answer",
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "latency_ms": 100.0,
            })
            assert rr.status_code == 201
            result_ids.setdefault(item["id"], []).append(rr.json()["id"])

    # blind judgments over each pair
    for winner, item_id in zip(("a", "tie", "b"),
                               list(result_ids)[:3]):
        ra, rb = result_ids[item_id]
        jr = client.post("/api/judgments", json={
            "eval_result_a": ra,
            "eval_result_b": rb,
            "judge_model": "external-judge",
            "judge_template_version": "blind-pair-v1",
            "winner": winner,
            "confidence": 0.8,
        })
        assert jr.status_code == 201

    assert len(client.get("/api/judgments").json()) == 3


def test_seed_script_importable_and_targets_contract(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_demo", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.seed_benchmarks)
    assert callable(mod.seed_synthetic)

    demo = json.loads((ROOT / "qa" / "suites" / "demo-seed.json").read_text())
    # demo payloads must satisfy the API contract exactly (one full run)
    from arcturos.main import create_app
    app = create_app(tmp_path / "arcturos_test.db")
    c = TestClient(app)
    for run in demo["runs"]:
        payload = {k: run[k] for k in
                   ("server_url", "model_fingerprint", "engine", "context_size")}
        r = c.post("/api/runs", json=payload)
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        for bench in run["benchmarks"]:
            rb = c.post(f"/api/runs/{rid}/benchmarks", json=bench)
            assert rb.status_code == 201, rb.text
        break  # one run is enough to prove contract match
