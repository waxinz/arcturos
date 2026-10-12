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
    # Owner renames (2026-10-11): "Kick off" -> "Start", "Runs" -> "Compare",
    # Run Diff retired as a page (its A/B tables live under the charts on
    # the Compare page). Page ids / routes / file names are stable — only
    # labels change.
    ("bench-kickoff.html",   "kickoff",   "Benchmarks", "Start"),
    # /create (legacy alias) serves bench-kickoff.html straight from
    # main.py — no separate file, so no shell drift between the two
    # routes (create.html deleted 2026-10-11 after its copy went stale).
    # Compare page (2026-10-11): the former /runs list merged into
    # /compare — index.html retired, / and /runs redirect (main.py).
    # Owner rename (same day): the page is "Compare".
    ("run_detail.html",      "runsdetail", "Benchmarks", "Run detail"),
    # Run detail is a drill-down, not a nav destination: it carries the
    # shell identity (Benchmarks section lit, no link highlighted) but
    # gets no NAV_MODEL entry.
    ("compare.html",         "compare",   "Benchmarks", "Compare"),
    # diff.html retired from the rail (owner call 2026-10-11): the A/B
    # diff tables render on the Compare page under the charts. The file
    # and /diff route stay alive (run detail deep-links ?run_a=…), but
    # the page is no longer a nav destination — it is a legacy alias
    # that redirects regular visitors to the integrated view.
    ("diff.html",            "diff",      "Benchmarks", "Run diff"),
    ("eval-suite-new.html",  "evalsuite", "Evals",      "Create"),
    ("evals.html",           "evals",     "Evals",      "Results"),
    ("judgments.html",       "judgments", "Evals",      "Judgments"),
    ("reports.html",         "reports",   "Evals",      "Reports"),
    ("models.html",          "models",    "Settings",   "Models"),
    ("baselines.html",       "baselines", "Settings",   "Baselines"),
]

# Ordered nav model: section -> [(page-id, href, label)]
NAV_MODEL = [
    ("Benchmarks", [
        ("kickoff", "/kickoff", "Start"),
        ("compare", "/compare", "Compare"),
    ]),
    ("Evals", [
        ("evalsuite", "/evals/suites/new", "Create"),
        ("evals",     "/evals",           "Results"),
        ("judgments", "/judgments",       "Judgments"),
        ("reports",   "/reports",         "Reports"),
    ]),
    ("Settings", [
        ("models",    "/models",    "Models"),
        ("baselines", "/baselines", "Baselines"),
    ]),
]


def sidebar_html() -> str:
    # 2026-10-11 FIFTH-ROUND REWRITE (owner: left rail is wrong, period).
    # Two-level TOP navigation: row 1 = brand + section tabs + theme
    # toggle; row 2 = the active section's pages as pills. The section
    # divs keep data-nav-section and the links keep data-page, so nav.js
    # keeps painting the active states.
    parts = ['  <header class="topnav">\n',
             '    <div class="topnav-row">\n'
             '      <a class="brand-tile" href="/" aria-hidden="true">A</a>\n'
             '      <a class="brand" href="/">Arcturos</a>\n'
             '      <nav class="topnav-tabs">\n']
    for section, links in NAV_MODEL:
        first = links[0][1]
        parts.append(f'        <div class="nav-section" data-nav-section="{section}">\n')
        parts.append(f'          <a class="nav-section-head" href="{first}">{section}</a>\n')
        parts.append('        </div>\n')
    parts.append('      </nav>\n'
                 '      <div class="sidebar-foot">\n'
                 '        <button class="theme-toggle-inline" data-theme-toggle>☾</button>\n'
                 '      </div>\n'
                 '    </div>\n'
                 '    <nav class="topnav-sub">\n')
    for section, links in NAV_MODEL:
        for page_id, href, label in links:
            parts.append(f'      <a href="{href}" data-page="{page_id}" '
                         f'data-nav-section="{section}">{label}</a>\n')
    parts.append('    </nav>\n'
                 '  </header>\n')
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
        # Idempotent: strip ALL previously injected class/data-* attrs
        # from the body tag (older runs of this script stacked them),
        # then re-add exactly one canonical set.
        t = re.sub(
            r'<body[^>]*>',
            lambda m: '<body class="shell" data-page="%s" data-section="%s"%s>'
            % (page, section,
               re.sub(r'\s+(?:class|data-page|data-section)="[^"]*"',
                      '', m.group(0)[len('<body'):-1]).rstrip()),
            t, count=1)

        # ---- 4) nav injection / refresh right after <body …> ---------
        # Replace-based (2026-10-11): the nav model is owned HERE, so a
        # changed NAV_MODEL must rewrite the nav on every page, not just
        # inject into pages missing one. The topnav container is matched
        # FIRST — matching only the legacy aside made every re-run after
        # the top-nav round inject a second header.
        if '<header class="topnav">' in t:
            t = re.sub(r'[ \t]*<header class="topnav">.*?</header>\n',
                       sidebar, t, count=1, flags=re.S)
        elif 'class="sidebar"' in t:
            t = re.sub(r'  <aside class="sidebar">.*?</aside>\n',
                       sidebar, t, count=1, flags=re.S)
        else:
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