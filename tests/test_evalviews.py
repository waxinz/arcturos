"""J3-J5 eval/judgment/report view tests — routes + suite report endpoint.

Covers: static view routes registered (200 text/html), the
/api/reports/suite/{id} happy path against a seeded suite + results +
judgments, 404 on unknown suite, and nav links added to the shared views.
"""

# ---------------------------------------------------------- fixtures --------

def _seed_suite_with_judgments(client) -> tuple[int, list[int]]:
    """One suite, two models × two items, two judgments (a win + b win)."""
    suite = client.post("/api/eval-suites", json={"name": "smoke", "version": "v1"})
    assert suite.status_code == 201, suite.text
    suite_id = suite.json()["id"]

    ids: list[int] = []
    for item_id in ("sr-001", "sr-002"):
        for fp in ("model-alpha", "model-beta"):
            r = client.post(
                f"/api/eval-suites/{suite_id}/results",
                json={
                    "model_fingerprint": fp,
                    "item_id": item_id,
                    "output": f"{fp} answer for {item_id}",
                    "prompt_tokens": 20,
                    "completion_tokens": 30,
                    "latency_ms": 100.0,
                },
            )
            assert r.status_code == 201, r.text
            ids.append(r.json()["id"])
    # ids order: [alpha sr-001, beta sr-001, alpha sr-002, beta sr-002]
    j1 = client.post(
        "/api/judgments",
        json={
            "eval_result_a": ids[0], "eval_result_b": ids[1],
            "judge_model": "judge-x", "judge_template_version": "blind-pair-v1",
            "winner": "a", "confidence": 0.8, "rationale": "shorter",
        },
    )
    assert j1.status_code == 201, j1.text
    j2 = client.post(
        "/api/judgments",
        json={
            "eval_result_a": ids[2], "eval_result_b": ids[3],
            "judge_model": "judge-x", "judge_template_version": "blind-pair-v1",
            "winner": "b", "confidence": None,
        },
    )
    assert j2.status_code == 201, j2.text
    return suite_id, ids


# ------------------------------------------------------------ views ---------


def test_eval_views_registered(client):
    for path in ("/evals", "/judgments", "/reports"):
        res = client.get(path)
        assert res.status_code == 200, f"{path}: {res.status_code}"
        assert "text/html" in res.headers["content-type"], path


def test_evals_view_contract(client):
    text = client.get("/evals").text
    assert "Evals" in text
    assert "/api/eval-suites" in text          # suite list + per-suite results
    assert "/judgments?result=" in text        # result-row drill-down links


def test_judgments_view_contract(client):
    text = client.get("/judgments").text
    assert "/api/judgments" in text
    assert "/api/eval-results" in text         # item_id resolution via results


def test_reports_view_contract(client):
    text = client.get("/reports").text
    assert "/api/reports/suite/" in text       # suite selector -> report fetch


def test_nav_links_in_shared_views(client):
    # 2026-09-30 grouped nav: every page labels the three sections and
    # still links all views. "Reports" (dropped from the nav in the same
    # round) is asserted via its API surface instead.
    for path in ("/", "/compare", "/diff", "/evals", "/judgments", "/reports"):
        text = client.get(path).text
        assert "Evals" in text, path
        assert "Judgments" in text, path
        assert "nav-group-label" in text, path        # grouped nav present
        assert "Benchmarks" in text, path


def test_new_views_have_full_nav(client):
    # Evals group pages cross-link each other; all pages reach Runs and
    # Compare; /create stays reachable (it hosts the eval cards).
    for path in ("/evals", "/judgments", "/reports"):
        text = client.get(path).text
        for link in ("href=\"/evals\"", "href=\"/judgments\"",
                     "href=\"/create#eval-suite-create\"",
                     "href=\"/\"", "href=\"/compare\""):
            assert link in text, f"{path} missing {link}"


# ------------------------------------------------- report endpoint ---------


def test_suite_report_happy_path(client):
    suite_id, _ = _seed_suite_with_judgments(client)
    res = client.get(f"/api/reports/suite/{suite_id}")
    assert res.status_code == 200, res.text
    data = res.json()

    report = data["report"]
    assert report["suite"] == {"id": suite_id, "name": "smoke", "version": "v1"}
    assert report["prompts"] == 2
    assert report["models"] == ["model-alpha", "model-beta"]
    assert report["results_stored"] == 4
    assert report["judgments_stored"] == 2
    assert report["overall"]["counts"] == {"a": 1, "b": 1, "tie": 0}
    assert report["overall"]["total"] == 2
    assert report["overall"]["percent"] == {"a": 50.0, "b": 50.0, "tie": 0.0}
    assert len(report["pairs"]) == 1
    pair = report["pairs"][0]
    assert pair["model_a"] == "model-alpha"
    assert pair["model_b"] == "model-beta"
    assert pair["counts"] == {"a": 1, "b": 1, "tie": 0}
    assert len(report["item_level"]) == 2

    csv = data["csv"]
    assert csv.startswith("suite,version,model_a")
    lines = csv.strip().splitlines()
    assert len(lines) == 2  # header + one pair row
    assert "smoke,v1,model-alpha,model-beta,1,1,0,2" in lines[1]


def test_suite_report_unknown_suite_404(client):
    res = client.get("/api/reports/suite/999")
    assert res.status_code == 404
    assert "999" in res.json()["detail"]


def test_suite_report_empty_suite(client):
    suite = client.post("/api/eval-suites", json={"name": "empty", "version": "v1"})
    suite_id = suite.json()["id"]
    res = client.get(f"/api/reports/suite/{suite_id}")
    assert res.status_code == 200, res.text
    report = res.json()["report"]
    assert report["overall"]["counts"] == {"a": 0, "b": 0, "tie": 0}
    assert report["overall"]["percent"] == {"a": None, "b": None, "tie": None}
    assert report["pairs"] == []
