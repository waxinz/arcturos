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
    for path in ("/evals", "/judgments"):
        res = client.get(path)
        assert res.status_code == 200, f"{path}: {res.status_code}"
        assert "text/html" in res.headers["content-type"], path


def test_evals_view_contract(client):
    text = client.get("/evals").text
    assert "Evals" in text
    assert "/api/eval-suites" in text          # suite list + per-suite results
    # (2026-10-25) the per-result "judgments" link column went with the
    # Judgments page's ledger table — judgments are reached through
    # /judgments/<id> and the pair view now.
    assert "/judgments?result=" not in text


def test_judgments_view_contract(client):
    text = client.get("/judgments").text
    assert "/api/judgments" in text
    assert "/api/eval-results" in text         # item_id resolution via results
    # Run-scoped judging round (2026-10-24): the per-model detail fields
    # are gone — the page only picks previously completed runs; fresh
    # replays and live A/B kick-offs live on the Run page.
    assert "/api/ops/judge/jobs" in text
    assert "/api/eval-runs" in text            # completed-run pickers
    assert "Run A" in text and "Run B" in text
    assert "judge-ma-url" not in text          # endpoint fields removed
    assert "judge-mb-url" not in text
    assert "/evals/run" in text                # pointer to the Run page
    assert "stage-chip" in text                # progress chrome retained
    # Ledger removed (2026-10-25, owner call): the paginated judgments
    # table + its judge-model dropdown are gone — run cards are the only
    # list on the page and /api/judgments feeds just them.
    assert 'id="judg-pager"' not in text
    assert 'id="judge-filter"' not in text
    assert "LEDGER_PAGE_SIZE" not in text
    assert 'id="judgments-body"' not in text


def test_reports_view_redirects(client):
    # Reports page dropped as unused (owner call 2026-10-25); /reports
    # 308s to the Evals landing page so old bookmarks still work. The
    # /api/reports/* endpoints stay (remove-UI-keep-API doctrine).
    # follow_redirects=False: the TestClient follows redirects by default
    # and would surface the destination page's 200 instead of the 308.
    res = client.get("/reports", follow_redirects=False)
    assert res.status_code == 308, res.status_code
    assert res.headers["location"] == "/evals/run"


def test_pair_page_has_category_summary(client):
    # Pair page (data-page jpair) leads with the per-category summary card;
    # the card hides itself when the summary endpoint has nothing to show.
    text = client.get("/judgments/pair").text
    assert 'id="jsum"' in text
    assert "Category summary" in text
    assert "Blind pairwise score" in text


def test_nav_links_in_shared_views(client):
    # 2026-10-11 top-nav round: the rail became a two-level top bar.
    # Every page still carries the three section identities and the
    # data-page/data-section identity for nav.js.
    for path in ("/", "/compare", "/diff", "/kickoff", "/evals",
                 "/evals/suites/new", "/judgments",
                 "/models", "/baselines", "/runs/1"):
        text = client.get(path).text
        assert 'data-nav-section="Benchmarks"' in text, path
        assert 'data-nav-section="Evals"' in text, path
        assert 'data-nav-section="Settings"' in text, path
        assert 'class="topnav"' in text, path
        assert 'data-page=' in text, path


def test_run_detail_uses_shared_shell(client):
    # 2026-10-08: run detail was missed by the round-2 shell migration and
    # still shipped the legacy header nav (wrong menu) + <h1>Arcturos</h1>
    # header. It now carries the top-nav shell like every other page, with
    # drill-down identity: Benchmarks section lit, NO page pill highlighted
    # (it is not a nav destination), and the legacy header is gone.
    text = client.get("/runs/1").text
    assert 'class="topnav"' in text
    assert '<body class="shell" data-page="runsdetail" ' \
           'data-section="Benchmarks">' in text
    assert 'data-nav-section="Benchmarks"' in text
    # legacy scaffolding must stay retired:
    assert '<h1>Arcturos</h1>' not in text
    assert 'nav-group-label' not in text
    assert 'Grouped-nav active highlight' not in text
    # content sits in the shell's .page container:
    assert '<main class="page">' in text


def test_evals_nav_order_and_no_reports(client):
    # Owner call (2026-10-25): Run is the Evals landing page and first
    # pill, Judgments right after it; the Reports pill is gone (page
    # dropped as unused — /reports itself 308s to /evals/run).
    text = client.get("/evals").text
    assert 'data-page="reports"' not in text
    assert '<a href="/reports" ' not in text
    sub = text.split('class="topnav-sub"')[1].split('</nav>')[0]
    hrefs = [seg.split('"', 1)[0]
             for seg in sub.split('href="')[1:]]
    evals = [h for h in hrefs
             if h in ('/evals/run', '/judgments', '/evals',
                      '/evals/suites/new')]
    assert evals == ['/evals/run', '/judgments', '/evals',
                     '/evals/suites/new'], evals
    # The Evals section head links to the landing page (Run): the first
    # occurrence of the attribute is the head div, and its href follows it.
    head = text.split('data-nav-section="Evals"')[1].split('href="', 1)[1]
    assert head.startswith('/evals/run'), head


def test_eval_suite_page_has_two_cards(client):
    # v0.2 (2026-10-15): replay moved to the /evals/run async job page —
    # the Create page keeps ONLY suite create + item definition.
    text = client.get("/evals/suites/new").text
    assert 'id="eval-suite-create"' in text
    assert 'id="eval-suite-items"' in text
    assert '<h3>Eval suite — create</h3>' in text
    assert 'id="eval-replay-kick"' not in text
    assert '<h3>Eval replay — kick off</h3>' not in text
    # the page points at the Run page for replaying
    assert 'href="/evals/run"' in text


def test_kickoff_page_bench_only(client):
    # Start page (owner rename 2026-10-11, ex-"Kick off"): ONLY the
    # benchmark card — no eval cards, no anchors.
    text = client.get("/kickoff").text
    assert 'id="bench-btn"' in text                  # bench card present
    assert 'id="eval-suite-create"' not in text      # eval cards absent
    assert 'id="eval-replay-kick"' not in text
    assert '/create#' not in text                    # no anchor nav links
    # /create (legacy alias) serves the same page
    alias = client.get("/create").text
    assert 'id="bench-btn"' in alias
    assert 'id="eval-suite-create"' not in alias


def test_split_pages_served(client):
    for path, marker in [("/kickoff", "Start"),
                         ("/evals/suites/new", "Create")]:
        res = client.get(path)
        assert res.status_code == 200, path
        assert "text/html" in res.headers["content-type"], path
        assert marker in res.text, path


def test_new_views_have_full_nav(client):
    # Evals group pages cross-link each other + reach Runs/Compare via the rail.
    for path in ("/evals", "/judgments"):
        text = client.get(path).text
        for link in ('href="/evals"', 'href="/judgments"',
                     'href="/evals/suites/new"',
                     'href="/"', 'href="/compare"'):
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
    assert report["overall"]["counts"] == {"a": 1, "b": 1, "tie": 0, "both_bad": 0}
    assert report["overall"]["total"] == 2
    assert report["overall"]["percent"] == {"a": 50.0, "b": 50.0, "tie": 0.0,
                                                "both_bad": 0.0}
    assert len(report["pairs"]) == 1
    pair = report["pairs"][0]
    assert pair["model_a"] == "model-alpha"
    assert pair["model_b"] == "model-beta"
    assert pair["counts"] == {"a": 1, "b": 1, "tie": 0, "both_bad": 0}
    assert len(report["item_level"]) == 2

    csv = data["csv"]
    assert csv.startswith("suite,version,model_a")
    lines = csv.strip().splitlines()
    # v0.2: pair rows + trailing dimension section (5 lines here)
    assert len(lines) == 5
    assert "smoke,v1,model-alpha,model-beta,1,1,0,0,2" in lines[1]
    assert "dimension,wins_a,wins_b,ties,both_bad,total" in lines[3]


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
    assert report["overall"]["counts"] == {"a": 0, "b": 0, "tie": 0, "both_bad": 0}
    assert report["overall"]["percent"] == {"a": None, "b": None, "tie": None,
                                                "both_bad": None}
    assert report["pairs"] == []


# --------------------------------------- pair summary endpoint --------------


def _seed_pair_summary_suite(client) -> int:
    """One suite, two models x two items, BOTH judgments favor model-alpha."""
    suite = client.post("/api/eval-suites", json={"name": "pairsum", "version": "v1"})
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
    for ra, rb in ((ids[0], ids[1]), (ids[2], ids[3])):
        j = client.post(
            "/api/judgments",
            json={
                "eval_result_a": ra, "eval_result_b": rb,
                "judge_model": "judge-x", "judge_template_version": "blind-pair-v1",
                "winner": "a", "confidence": 0.9,
            },
        )
        assert j.status_code == 201, j.text
    return suite_id


def test_pair_summary_endpoint_caller_orientation(client):
    # 'a' in the response ALWAYS means the caller's model_a won, whatever
    # canonical order the judgment rows were stored in.
    suite_id = _seed_pair_summary_suite(client)
    res = client.get("/api/judgments/pair/summary", params={
        "suite_id": suite_id, "model_a": "model-alpha", "model_b": "model-beta"})
    assert res.status_code == 200, res.text
    s = res.json()
    assert s["total"] == 2
    assert s["overall"]["counts"] == {"a": 2, "b": 0, "tie": 0, "both_bad": 0}
    assert s["overall"]["score_a"] == 100.0
    assert s["overall"]["score_b"] == 0.0
    assert s["overall"]["wilson_a"] is not None
    assert s["rows"] and s["rows"][0]["dimension"] == "uncategorized"
    assert s["judges"] == ["judge-x"]
    assert s["templates"] == ["blind-pair-v1"]

    # swapped call mirrors the a/b columns; totals/ties/rows never move
    rev = client.get("/api/judgments/pair/summary", params={
        "suite_id": suite_id, "model_a": "model-beta", "model_b": "model-alpha"})
    assert rev.status_code == 200, rev.text
    r = rev.json()
    assert r["total"] == 2
    assert r["overall"]["counts"] == {"a": 0, "b": 2, "tie": 0, "both_bad": 0}
    assert r["overall"]["score_a"] == 0.0
    assert r["overall"]["score_b"] == 100.0
    assert [row["dimension"] for row in r["rows"]] == \
        [row["dimension"] for row in s["rows"]]


def test_pair_summary_endpoint_404_and_empty_pair(client):
    # unknown suite -> 404 (require_parent, same as the rows endpoint)
    res = client.get("/api/judgments/pair/summary", params={
        "suite_id": 999, "model_a": "a", "model_b": "b"})
    assert res.status_code == 404
    # known suite, unjudged pair -> honest empty summary
    suite = client.post("/api/eval-suites", json={"name": "empty", "version": "v1"})
    empty_id = suite.json()["id"]
    res = client.get("/api/judgments/pair/summary", params={
        "suite_id": empty_id, "model_a": "model-alpha", "model_b": "model-beta"})
    assert res.status_code == 200, res.text
    s = res.json()
    assert s["rows"] == []
    assert s["total"] == 0
    assert s["overall"]["percent"] == {"a": None, "b": None, "tie": None,
                                           "both_bad": None}
