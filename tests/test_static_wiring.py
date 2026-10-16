"""Static-surface wiring tests (2026-10-15):

- sticky-inputs.js: served, and included by EVERY static page (the sticky
  defaults feature is global by contract — a page missing the script tag
  silently loses operator-entered defaults).
- nav order: Judgments is the FIRST sub-element under the Evals section
  (operator's standing preference: the judging kick-off leads the menu).
- Judgments page: the Model B baseline-reuse picker and the recent
  judgment-runs list exist; the old per-item work-detail table is gone.
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


def test_nav_judgments_first_under_evals():
    nav = (Path(STATIC_DIR) / "partials" / "nav.html").read_text()
    sub = re.search(r'<nav class="topnav-sub">(.*?)</nav>', nav, re.S)
    assert sub, "nav partial must keep the sub-level nav"
    evals_links = re.findall(
        r'<a href="([^"]+)"[^>]*data-nav-section="Evals"', sub.group(1))
    assert evals_links, "Evals section must have sub-elements"
    assert evals_links[0] == "/judgments", (
        f"Judgments must be the FIRST Evals sub-element, got {evals_links}")


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
                 "evals", "reports", "models", "baselines"]
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


def test_judgments_page_baseline_picker_and_run_list(client):
    res = client.get("/judgments")
    assert res.status_code == 200
    text = res.text
    # Model B "reuse a recent run" picker
    assert 'id="judge-mb-run"' in text
    assert "/api/eval-runs" in text               # dropdown data source
    assert "eval_run_b" in text                   # mixed kick-off payload
    # recent judgment runs list (replaces the per-item work table)
    assert 'id="judgrun-list"' in text
    assert "/judgments/pair?" in text             # deep link to SxS view
    # the old work-detail table + zoom toggle are fully removed
    assert 'id="jbp-detail"' not in text
    assert 'id="jbp-zoom"' not in text
    assert 'id="jbp-rows"' not in text


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
