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
