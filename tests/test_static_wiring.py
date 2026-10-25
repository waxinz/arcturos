"""Static-surface wiring tests (2026-10-15):

- sticky-inputs.js: served, and included by EVERY static page (the sticky
  defaults feature is global by contract — a page missing the script tag
  silently loses operator-entered defaults).
- nav order: Judgments is the FIRST sub-element under the Evals section
  (operator's standing preference: the judging kick-off leads the menu).
- Judgments page: the Run A/B completed-run pickers and the recent
  judgment-runs list exist; the old per-item work-detail table is gone.
  (2026-10-24 run-scoped round: the per-model endpoint/model/reasoning
  fields were removed — the page only judges previously completed runs.)
- Reasoning controls (2026-10-15): reasoning_effort + reasoning_budget
  fields ship on the eval-run start surface (Run page) and are wired
  into the kick-off payload. (Judgements left the run-start business
  in the 2026-10-24 run-scoped round.)
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient  # noqa: F401 — client fixture

from arcturos.main import STATIC_DIR

HTML_PAGES = sorted(p.name for p in Path(STATIC_DIR).glob("*.html"))


def test_sticky_inputs_script_served(client):
    res = client.get("/static/sticky-inputs.js")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
    # the feature's core behaviors must be present in the shipped file
    body = res.text
    assert "arcturos-sticky:" in body            # localStorage key prefix
    assert "DOMContentLoaded" in body            # restore on load
    assert "textarea" in body                    # textareas stick too


def test_every_page_includes_sticky_inputs(client):
    assert HTML_PAGES, "static dir must contain HTML pages"
    for name in HTML_PAGES:
        html = (Path(STATIC_DIR) / name).read_text()
        assert "sticky-inputs.js" in html, f"{name} is missing sticky-inputs.js"


def test_nav_run_first_under_evals():
    nav = (Path(STATIC_DIR) / "partials" / "nav.html").read_text()
    sub = re.search(r'<nav class="topnav-sub">(.*?)</nav>', nav, re.S)
    assert sub, "nav partial must keep the sub-level nav"
    evals_links = re.findall(
        r'<a href="([^"]+)"[^>]*data-nav-section="Evals"', sub.group(1))
    assert evals_links, "Evals section must have sub-elements"
    # Owner call 2026-10-25 (supersedes the 2026-10-15 Judgments-first
    # order): Run is the first pill, Judgments right after it; the
    # Reports pill is gone (page dropped as unused).
    assert evals_links == ["/evals/run", "/judgments", "/evals",
                           "/evals/suites/new"], (
        f"Evals pills must be Run, Judgments, Results, Create, got "
        f"{evals_links}")
    # The EVALS section tab itself lands on Run (owner call 2026-10-25:
    # clicking 'Evals' defaults to the eval run kick-off page).
    head = re.search(
        r'<a class="nav-section-head" href="([^"]+)"'
        r'[^>]*>Evals</a>', nav)
    assert head and head.group(1) == "/evals/run", (
        "the Evals section tab must link to /evals/run")


def test_no_page_level_nav_or_header_css():
    """Rail-era page <style> rules must stay dead (2026-10-23 round).

    `nav a { margin-right: 1rem; ... }` / `nav a.active { underline }` /
    `header { border-bottom; padding; margin }` in a page's <style> hit
    the INJECTED top bar (pages carry no <nav>/<header> of their own —
    pagelib injects .topnav at request time). Seven pages shipped with
    this block and spaced/underlined the top menu differently from the
    rest. nav.css owns all chrome styling.
    """
    assert HTML_PAGES, "static dir must contain HTML pages"
    for name in HTML_PAGES:
        html = (Path(STATIC_DIR) / name).read_text()
        assert "nav a {" not in html, f"{name} carries a legacy nav a rule"
        assert "nav a.active" not in html, (
            f"{name} carries a legacy nav a.active rule")
        assert not re.search(r"^\s*header \{", html, re.M), (
            f"{name} carries a legacy header rule")


def test_nav_css_owns_tab_decoration():
    """Section tabs must kill the UA link underline themselves.

    theme.css styles link color only; an unstyled anchor renders with
    Chrome's default underline — which shipped on the pages without the
    legacy nav rules. The kill now lives in nav.css, the single owner.
    """
    css = (Path(STATIC_DIR) / "nav.css").read_text()
    block = re.search(r"\.nav-section-head \{[^}]*\}", css)
    assert block, ".nav-section-head rule missing from nav.css"
    assert "text-decoration: none" in block.group(0)


def test_nav_pills_constant_across_pages(client):
    """2026-10-15: the sub-nav menu must be IDENTICAL on every page — the
    per-section pill swap read as 'the menu changed' when moving between
    pages. All 9 pills ship in the partial and stay visible; section
    boundaries are thin separators, not show/hide."""
    nav = (Path(STATIC_DIR) / "partials" / "nav.html").read_text()
    sub_m = re.search(r'<nav class="topnav-sub">(.*?)</nav>', nav, re.S)
    assert sub_m, "nav partial must keep the sub-level nav"
    sub = sub_m.group(1)
    all_pages = ["kickoff", "compare", "judgments", "evalsuite", "evalrun",
                 "evals", "models", "baselines"]
    for page in all_pages:
        assert f'data-page="{page}"' in sub, f"nav is missing pill {page}"
    assert sub.count('class="sub-sep"') == 2       # one between each pair
    # the hide-non-active-section rule is gone from the stylesheet
    css = (Path(STATIC_DIR) / "nav.css").read_text()
    assert ".topnav-sub a { display: none; }" not in css
    # and the served page really renders every pill in the sub row
    page = client.get("/judgments").text
    sub_dom = re.search(r'<nav class="topnav-sub">(.*?)</nav>', page, re.S)
    assert sub_dom, "judgments page must carry the shared nav"
    for page_id in all_pages:
        assert f'data-page="{page_id}"' in sub_dom.group(1)


def test_judgments_page_run_pickers_and_run_list(client):
    res = client.get("/judgments")
    assert res.status_code == 200
    text = res.text
    # Run A + Run B completed-run pickers (run-scoped judging round,
    # 2026-10-24: the ONLY per-side input on this page — endpoint/model/
    # reasoning fields were removed; mixed/combined kick-offs stay
    # API-only on POST /api/ops/judge/jobs)
    assert 'id="judge-ma-run"' in text
    assert 'id="judge-mb-run"' in text
    assert "/api/eval-runs" in text               # dropdown data source
    assert "eval_run_ids" in text                 # run-scoped re-judge payload
    assert "eval_run_a" not in text               # mixed payload left the UI
    assert "eval_run_b" not in text
    assert "target_a" not in text                 # combined payload too
    assert "target_b" not in text
    assert "reasonProps" not in text              # reasoning fields removed
    # the pickers list finished runs that stored results only
    assert "status === 'complete'" in text
    # recent judgment runs list (replaces the per-item work table)
    assert 'id="judgrun-list"' in text
    assert "/judgments/pair?" in text             # deep link to SxS view
    # the old work-detail table + zoom toggle are fully removed
    assert 'id="jbp-detail"' not in text
    assert 'id="jbp-zoom"' not in text
    assert 'id="jbp-rows"' not in text


def test_judgment_primary_key_round(client):
    """2026-10-25: judgments are addressed by their OWN primary key.
    The recent run cards enter the SxS pair page through ?judgment=<id>
    (one PK in the URL, not a suite/model foreign-key soup); the
    single-judgment drill-down serves at /judgments/<id>; the pair page
    accepts the PK entry and links every verdict card back to
    /judgments/<id>. (Same-day owner call: the paginated ledger table
    under the run cards is gone — cards are the only list.)"""
    led = client.get("/judgments").text
    # Ledger removed (2026-10-25, owner call): the paginated table + its
    # judge-model dropdown are gone; individual judgments are reached
    # through the /judgments/<id> drill-down from the pair page's cards.
    assert "<th>#</th>" not in led
    assert "'/judgments/' + j.id" not in led
    assert "'/judgments/pair?judgment=' + g.latest" in led   # run cards
    assert "'&model_a=' + encodeURIComponent(g.fpLo" not in led  # soup gone

    # single-judgment drill-down: shell identity + presentation-order
    # cards (the rationale's "Response 1" is the first card)
    res = client.get("/judgments/2903")
    assert res.status_code == 200
    jtext = res.text
    assert 'data-page="judgment"' in jtext
    assert 'data-section="Evals"' in jtext
    assert "/api/judgments/" in jtext             # PK data source
    assert "shown to judge as Response 1" in jtext
    assert "all judgments for this pair" in jtext
    assert "'/judgments/pair?judgment=' + row.id" in jtext

    # pair page: PK entry + per-row drill-in links
    pair = client.get("/judgments/pair").text
    assert "q.get('judgment')" in pair            # PK URL shape accepted
    assert "'/api/judgments/' + jref" in pair     # resolution fetch
    assert 'class="jref" href="/judgments/' + "' + row.id" in pair

    # Results page deep-link carries a judgment PK too
    ev = client.get("/evals").text
    assert "'/judgments/pair?judgment=' + rows[0].id" in ev


def test_judgments_ping_button_and_done_link(client):
    """2026-10-25: the judge kick-off gets the same preflight ping the
    Run page has ('One ping only'), and a completed job links its fresh
    results through a judgment primary key instead of /evals."""
    text = client.get("/judgments").text
    assert 'id="judge-pf-btn"' in text
    assert "One ping only" in text                # same label as Run page
    assert "/api/ops/preflight" in text
    assert "eval_target" in text                  # blank-URL default source
    assert "open judgment results" in text        # done-link
    assert "linkFreshResults" in text
    assert "link.href = '/evals'" not in text     # old Results hop retired


def test_judgment_detail_prompt_expander(client):
    """J8 (2026-10-15): tapping an item name on the SxS pair page expands
    the case's prompt underneath (definition-snapshot sourced)."""
    res = client.get("/judgments/pair")
    assert res.status_code == 200
    text = res.text
    assert "makeItemHead" in text                 # button-style item head
    assert "fillPrompt" in text
    assert "definition" in text                   # snapshot data source
    assert "prompt-box" in text                   # collapsible panel
    assert "aria-expanded" in text                # accessible toggle
    assert "Turn " in text                        # multi-turn turn labels
    # image support: http refs load directly; suite-relative refs go
    # through the guarded suite-image endpoint; failures degrade to a note
    assert "p-img" in text
    assert "/api/eval-suites/' + suiteId + '/image?item=" in text
    assert "image failed to load" in text
    assert "loading" in text                      # lazy loading


def test_reasoning_fields_on_every_run_start_surface(client):
    """J9 (2026-10-15): optional per-model reasoning_effort +
    reasoning_budget on the surface that starts an eval suite run —
    the single-model replay page. (2026-10-24 run-scoped round: Judgements
    no longer starts eval runs — its model cards became completed-run
    pickers, so the replay page is the only run-start surface.)"""
    # single-model replay page (/evals/run)
    run = client.get("/evals/run").text
    assert 'id="eval-reason"' in run
    assert 'id="eval-budget"' in run
    assert "body.reasoning_effort" in run          # wired into the POST body
    assert "body.reasoning_budget" in run
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        assert f"<option>{effort}</option>" in run
