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
    # v0.2 (2026-10-15): replay got its own async job page (live per-item
    # progress + operator stop) — the Run pill.
    ("eval-run.html",        "evalrun",   "Evals",      "Run"),
    ("evals.html",           "evals",     "Evals",      "Results"),
    ("judgments.html",       "judgments", "Evals",      "Judgments"),
    ("reports.html",         "reports",   "Evals",      "Reports"),
    # Drill-downs (2026-10-15): shell identity, no NAV_MODEL entry.
    ("eval-run-detail.html", "evalrundetail", "Evals", "Run detail"),
    ("judgment-detail.html", "jpair",         "Evals", "Judgment detail"),
    # Single-judgment drill-down (2026-10-25): /judgments/<id> — one
    # stored judgment addressed by its own primary key.
    ("judgment.html",        "judgment",      "Evals", "Judgment"),
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
        ("evalrun",   "/evals/run",        "Run"),
        ("evals",     "/evals",            "Results"),
        ("judgments", "/judgments",        "Judgments"),
        ("reports",   "/reports",          "Reports"),
    ]),
    ("Settings", [
        ("models",    "/models",    "Models"),
        ("baselines", "/baselines", "Baselines"),
    ]),
]


# Section landing pills (hand-curated shape of the shipped nav partial):
# each section's FIRST pill and its head link are the page the operator
# lands on — Evals lands on Judgments, not the alphabetically-first
# Create. Without this map a regen silently reordered the pills and
# dropped the separators (the partial had drifted from the generator).
LANDING: dict[str, str] = {
    "Benchmarks": "kickoff",
    "Evals": "judgments",
    "Settings": "models",
}


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
        land = LANDING.get(section)
        first = next((href for pid, href, _ in links if pid == land),
                     links[0][1])
        parts.append(f'        <div class="nav-section" data-nav-section="{section}">\n')
        parts.append(f'          <a class="nav-section-head" href="{first}">{section}</a>\n')
        parts.append('        </div>\n')
    parts.append('      </nav>\n'
                 '      <div class="sidebar-foot">\n'
                 '        <button class="theme-toggle-inline" data-theme-toggle>☾</button>\n'
                 '      </div>\n'
                 '    </div>\n'
                 '    <nav class="topnav-sub">\n')
    for si, (section, links) in enumerate(NAV_MODEL):
        if si:
            # one thin separator between sections (test-pinned count: 2)
            parts.append('      <span class="sub-sep" aria-hidden="true"></span>\n')
        land = LANDING.get(section)
        ordered = sorted(links, key=lambda l: 0 if l[0] == land else 1)
        for page_id, href, label in ordered:
            parts.append(f'      <a href="{href}" data-page="{page_id}" '
                         f'data-nav-section="{section}">{label}</a>\n')
    parts.append('    </nav>\n'
                 '  </header>\n')
    return ''.join(parts)


def build_pages() -> None:
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

        # ---- 4) nav placeholder ---------------------------------------
        # 2026-10-15: pages carry a `<!--NAV-->` placeholder; the header
        # is ONE canonical partial (partials/nav.html, written below from
        # NAV_MODEL) injected server-side at request time (pagelib).
        # Stamping full markup into 13 files let copies drift (a regen
        # once silently deleted the Run pill).
        if '<!--NAV-->' in t:
            pass                                    # already wired
        elif '<header class="topnav">' in t:
            t = re.sub(r'[ \t]*<header class="topnav">.*?</header>\n',
                       '  <!--NAV-->\n', t, count=1, flags=re.S)
        elif 'class="sidebar"' in t:
            t = re.sub(r'  <aside class="sidebar">.*?</aside>\n',
                       '  <!--NAV-->\n', t, count=1, flags=re.S)
        else:
            m = re.search(r'<body[^>]*>', t)
            assert m, f"{fname}: <body> not found"
            t = t[:m.end()] + "\n  <!--NAV-->\n" + t[m.end():]

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

    # The menu itself: ONE canonical partial injected server-side by
    # pagelib at request time. This file is the single source of truth.
    partials = STATIC / "partials"
    partials.mkdir(exist_ok=True)
    out = partials / "nav.html"
    out.write_text(sidebar_html())
    print(f"built partials/nav.html ({len(NAV_MODEL)} sections)")


if __name__ == "__main__":
    build_pages()
    print("done")