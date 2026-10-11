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
    # Legacy alias (2026-09-30 split): /create serves the same bench
    # start card as /kickoff — same page identity, kept in the shell
    # sync so its rail never drifts.
    ("create.html",          "kickoff",   "Benchmarks", "Start"),
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

FOOT = '<div class="sidebar-foot">\n      <button class="theme-toggle-inline" data-theme-toggle>☾</button>\n    </div>'


# Inline stroke icons per page (16px, currentColor — they inherit the
# row's text colour, so hover/active states recolour them for free).
NAV_ICONS = {
    "kickoff":   '<path d="M7 4.5l12 7.5-12 7.5z"/>',                      # play
    "compare":   '<path d="M5 20v-9M12 20V4M19 20v-5"/>',                  # bars
    "evalsuite": '<rect x="3.5" y="3.5" width="17" height="17" rx="3.5"/>'
                 '<path d="M12 8.5v7M8.5 12h7"/>',                         # plus-square
    "evals":     '<path d="M8.5 6H21M8.5 12H21M8.5 18H21"/>'
                 '<path d="M3.5 6h1M3.5 12h1M3.5 18h1"/>',                 # list
    "judgments": '<path d="M12 3.5l2.7 5.4 6 .9-4.3 4.2 1 6L12 17.2l-5.4 2.8 1-6L3.3 9.8l6-.9z"/>',  # star
    "reports":   '<path d="M14 3.5H7a2 2 0 00-2 2v13a2 2 0 002 2h10a2 2 0 002-2V8.5z"/>'
                 '<path d="M14 3.5v5h5M9.5 13h5M9.5 16.5h5"/>',            # document
    "models":    '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>'
                 '<path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21'
                 'M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21"/>',      # chip
    "baselines": '<path d="M12 3.5l8.5 4.75L12 13 3.5 8.25z"/>'
                 '<path d="M4.5 12.5l7.5 4.25 7.5-4.25"/>'
                 '<path d="M4.5 16.5l7.5 4.25 7.5-4.25"/>',                # layers
}


def _icon(page_id: str) -> str:
    body = NAV_ICONS.get(page_id, '')
    if not body:
        return ''
    return ('<svg class="nav-ico" viewBox="0 0 24 24" width="16" height="16"'
            ' fill="none" stroke="currentColor" stroke-width="1.8"'
            ' stroke-linecap="round" stroke-linejoin="round"'
            ' aria-hidden="true">' + body + '</svg>')


def sidebar_html() -> str:
    # 2026-10-11 full redesign (owner: "needs a full complete rethink"):
    # brand monogram tile + name row; plain uppercase section labels;
    # icon + text link rows. No dots, no decorative bars.
    parts = ['  <aside class="sidebar">\n',
             '    <div class="brand-row">\n'
             '      <span class="brand-tile" aria-hidden="true">A</span>\n'
             '      <span class="brand-text">\n'
             '        <a class="brand" href="/">Arcturos</a>\n'
             '        <span class="brand-sub">LLM benchmark &amp; eval dashboard</span>\n'
             '      </span>\n'
             '    </div>\n'
             '    <nav>\n']
    for section, links in NAV_MODEL:
        parts.append(f'      <div class="nav-section" data-nav-section="{section}">\n')
        parts.append(f'        <div class="nav-section-head">{section}</div>\n')
        for page_id, href, label in links:
            parts.append(f'        <a href="{href}" data-page="{page_id}">'
                         f'{_icon(page_id)}<span class="nav-label">{label}</span></a>\n')
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

        # ---- 4) sidebar injection / refresh right after <body …> ------
        # Replace-based (2026-10-11): the nav model is owned HERE, so a
        # changed NAV_MODEL (e.g. the Runs+Compare merge) must rewrite the
        # rail on every page, not just inject into pages missing one.
        if 'class="sidebar"' in t:
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