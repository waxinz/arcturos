"""API tests for the suite-item image endpoint (2026-10-15).

Feeds the judgment-details prompt expander: GET
/api/eval-suites/{id}/image?item=<item_id> resolves one item's image —
http(s) refs pass through as ``url``; local refs are inlined as a base64
``data_url`` with the SAME resolution rules as replay. The ref must come
from the suite's definition snapshot (no arbitrary filesystem reads).
"""

import base64
import json

import pytest

from arcturos import db as dbmod
from arcturos import evals as evals_mod

# 1x1 transparent PNG
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd400000000"
    "49454e44ae426082")


def _seed_suite(tmp_path, items, suite_id=1, snapshot=True):
    conn = dbmod.connect(tmp_path / "api.db")
    dbmod.init_db(conn)
    conn.execute("INSERT INTO eval_suites (id, name, version) "
                 "VALUES (?, 's', 'v1')", (suite_id,))
    if snapshot:
        conn.execute(
            "INSERT INTO suite_definitions (suite_id, payload, created_at) "
            "VALUES (?, ?, 'now')",
            (suite_id, json.dumps({"items": items})))
    conn.commit()
    conn.close()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from arcturos.main import create_app
    monkeypatch.setattr(evals_mod, "SUITE_ASSETS_DIR", tmp_path / "assets")
    app = create_app(tmp_path / "api.db")
    return TestClient(app), tmp_path


def test_local_image_inlined_as_data_url(client, tmp_path):
    tc, _ = client
    assets = tmp_path / "assets" / "images"
    assets.mkdir(parents=True)
    (assets / "test.png").write_bytes(PNG_1PX)
    _seed_suite(tmp_path, [{"id": "img1", "dimension": "vision",
                            "prompt": "what is this?",
                            "image": "images/test.png"}])
    res = tc.get("/api/eval-suites/1/image", params={"item": "img1"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["url"] is None
    assert body["data_url"].startswith("data:image/png;base64,")
    b64 = body["data_url"].split(",", 1)[1]
    assert base64.b64decode(b64) == PNG_1PX


def test_http_image_passes_through_untouched(client, tmp_path):
    tc, _ = client
    _seed_suite(tmp_path, [{"id": "img2", "dimension": "vision",
                            "prompt": "what is this?",
                            "image": "https://example.com/cat.png"}])
    res = tc.get("/api/eval-suites/1/image", params={"item": "img2"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["url"] == "https://example.com/cat.png"
    assert body["data_url"] is None


def test_missing_local_image_is_404_not_crash(client, tmp_path):
    tc, _ = client
    _seed_suite(tmp_path, [{"id": "img3", "dimension": "vision",
                            "prompt": "x", "image": "images/gone.png"}])
    res = tc.get("/api/eval-suites/1/image", params={"item": "img3"})
    assert res.status_code == 404
    assert "gone.png" in res.json()["detail"]


def test_unknown_suite_and_no_snapshot_404(client, tmp_path):
    tc, tmp = client
    _seed_suite(tmp, [{"id": "q1", "prompt": "x"}])   # no image, snapshotted
    # unknown suite id entirely
    r = tc.get("/api/eval-suites/99/image", params={"item": "q1"})
    assert r.status_code == 404
    # known suite, but never snapshotted
    conn = dbmod.connect(tmp / "api.db")
    conn.execute("INSERT INTO eval_suites (id, name, version) "
                 "VALUES (2, 's2', 'v1')")
    conn.commit()
    conn.close()
    r = tc.get("/api/eval-suites/2/image", params={"item": "q1"})
    assert r.status_code == 404 and "snapshot" in r.json()["detail"]


def test_item_without_image_404(client, tmp_path):
    tc, tmp = client
    _seed_suite(tmp, [{"id": "q1", "dimension": "reasoning", "prompt": "x"}])
    res = tc.get("/api/eval-suites/1/image", params={"item": "q1"})
    assert res.status_code == 404
    assert "no image" in res.json()["detail"]


def test_no_arbitrary_filesystem_read(client, tmp_path):
    """The item param must name a snapshot item — a raw path (even one
    that exists on disk) is rejected, so the endpoint can't be walked."""
    tc, tmp = client
    secret = tmp / "secret.png"
    secret.write_bytes(PNG_1PX)
    _seed_suite(tmp, [{"id": "q1", "dimension": "vision", "prompt": "x",
                       "image": "images/test.png"}])
    for bad in ("../secret.png", "secret.png", "images/test.png NOT"):
        r = tc.get("/api/eval-suites/1/image", params={"item": bad})
        assert r.status_code == 404, bad
        assert "not in suite" in r.json()["detail"]


def test_item_id_must_match_snapshot_exactly(client, tmp_path):
    tc, tmp = client
    _seed_suite(tmp, [{"id": "img1", "dimension": "vision", "prompt": "x",
                       "image": "images/test.png"}])
    r = tc.get("/api/eval-suites/1/image", params={"item": "img1#t2"})
    assert r.status_code == 404
