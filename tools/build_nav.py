#!/usr/bin/env python3
"""Generator for the 2026-09-30 round-2 page shell (sidebar nav).

Page model (owner call): features get their OWN pages, not anchors.
- /kickoff          -> bench-kickoff.html  (bench kick-off card only)
- /evals/suites/new -> eval-suite-new.html (suite create + items + replay)
- /create           -> legacy path: serves bench-kickoff.html

Rather than hand-editing every file (drift-prone), this script owns the
shared shell — head links, sidebar markup, body identity — and rewrites
each page in place. Page-specific <style> bodies and scripts survive;
legacy header/nav/title scaffolding is stripped. Regenerable:

    python3 tools/build_nav.py
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "src" / "arcturos" / "static"

# Existing file -> (page-id, section, <h1> title)
PAGES: list[tuple[str, str, str, str]] = [
    ("bench-kickoff.html",   "kickoff",   "Benchmarks", "Kick off"),
    ("index.html",           "runs",      "Benchmarks", "Runs"),
    ("compare.html",         "compare",   "Benchmarks", "Compare"),
    ("diff.html",            "diff",      "Benchmarks", "Run Diff"),
    ("eval-suite-new.html",  "evalsuite", "Evals",      "Suite create"),
    ("evals.html",           "evals",     "Evals",      "Results"),
    ("judgments.html",       "judgments", "Evals",      "Judgments"),
    ("reports.html",         "reports",   "Evals",      "Reports"),
    ("models.html",          "models",    "Settings",   "Models"),
    ("baselines.html",       "baselines", "Settings",   "Baselines"),
]

# Ordered nav model: section -> [(page-id, href, label)]
NAV_MODEL = [
    ("Benchmarks", [
        ("kickoff", "/kickoff", "Kick off"),
        ("runs",    "/",        "Runs"),
        ("compare", "/compare", "Compare"),
        ("diff",    "/diff",    "Run diff"),
    ]),
    ("Evals", [
        ("evalsuite", "/evals/suites/new", "Suite create"),
        ("evals",     "/evals",           "Results"),
        ("judgments", "/judgments",       "Judgments"),
        ("reports",   "/reports",         "Reports"),
    ]),
    ("Settings", [
        ("models",    "/models",    "Models"),
        ("baselines", "/baselines", "Baselines"),
    ]),
]

FOOT = '<div class="sidebar-foot">\n      <button class="theme-toggle-inline" data-theme-toggle>☾</button>\n    </div>'


def sidebar_html() -> str:
    parts = ['  <aside class="sidebar">\n',
             '    <a class="brand" href="/">Arcturos</a>\n',
             '    <span class="brand-sub">LLM benchmark &amp; eval dashboard</span>\n',
             '    <nav>\n']
    for section, links in NAV_MODEL:
        parts.append(f'      <div class="nav-section" data-nav-section="{section}">\n')
        parts.append(f'        <div class="nav-section-head">{section}</div>\n')
        for page_id, href, label in links:
            parts.append(f'        <a href="{href}" data-page="{page_id}">{label}</a>\n')
        parts.append('      </div>\n')
    parts.append('    </nav>\n')
    parts.append('    ' + FOOT + '\n')
    parts.append('  </aside>\n')
    return ''.join(parts)


def build_pages() -> None:
    sidebar = sidebar_html()
    for fname, page, section, _title in PAGES:
        src = STATIC / fname
        if not src.exists():
            print(f"SKIP (missing): {fname}")
            continue
        t = src.read_text()
        orig = t

        # ---- 1) stylesheet wiring: nav.css right after theme.css ----
        if '/static/nav.css' not in t:
            t = t.replace('  <link rel="stylesheet" href="/static/theme.css">',
                          '  <link rel="stylesheet" href="/static/theme.css">\n'
                          '  <link rel="stylesheet" href="/static/nav.css">', 1)

        # ---- 2) scripts: nav.js after theme.js -----------------------
        if '/static/nav.js' not in t:
            t = t.replace('  <script src="/static/theme.js"></script>',
                          '  <script src="/static/theme.js"></script>\n'
                          '  <script src="/static/nav.js"></script>', 1)

        # ---- 3) body identity ---------------------------------------
        t = re.sub(r'<body([^>]*)>',
                   rf'<body\1 class="shell" data-page="{page}" '
                   rf'data-section="{section}">', t, count=1)

        # ---- 4) sidebar injection right after <body …> --------------
        if 'class="sidebar"' not in t:
            m = re.search(r'<body[^>]*>', t)
            assert m, f"{fname}: <body> not found"
            t = t[:m.end()] + "\n" + sidebar + t[m.end():]

        # ---- 5) retire the legacy header (brand + old nav) ----------
        t = re.sub(r"\s*<header>\s*<h1>Arcturos</h1>.*?</header>",
                   "", t, count=1, flags=re.S)

        # ---- 6) retire legacy per-page highlighters ------------------
        t = re.sub(
            r"\n?\s*<script>\s*// Grouped-nav active highlight.*?</script>",
            "", t, flags=re.S)

        # ---- 7) retire legacy anchor-routing scripts -----------------
        t = re.sub(
            r"\n?\s*<script>\s*// Grouped-nav anchor routing.*?</script>",
            "", t, flags=re.S)

        # ---- 8) idempotency marker -----------------------------------
        if f'data-page="{page}"' not in t:
            print(f"WARN: {fname} missing data-page={page}")

        if t != orig:
            src.write_text(t)
            print(f"built {fname} ({page} / {section})")
        else:
            print(f"unchanged: {fname}")


if __name__ == "__main__":
    build_pages()
    print("done")