"""pagelib.py — server-side page assembly (2026-10-15).

The top-nav menu is ONE canonical partial (static/partials/nav.html)
injected into every page at request time; pages carry a ``<!--NAV-->``
placeholder where the header sits. This kills the drift class where the
stamped copies in 13 files diverged (the 2026-10-15 round where a
generator regen silently deleted the Run pill because the pill lived in
hand-maintained page markup).

Routes serve pages through `page_response()` instead of FileResponse.
Direct /static/*.html fetches bypass injection and show the placeholder
— routes are the canonical entry, that is acceptable.

Reads are per-request (files are small; always fresh, no cache
invalidation bugs). The placeholder must appear EXACTLY once per page.
"""

from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_NAV_PLACEHOLDER = "<!--NAV-->"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def nav_partial() -> str:
    """The canonical top-nav header markup (single source of truth)."""
    return _read(_STATIC_DIR / "partials" / "nav.html")


def render_page(name: str) -> str:
    """One static page with the nav partial injected."""
    html = _read(_STATIC_DIR / name)
    count = html.count(_NAV_PLACEHOLDER)
    if count != 1:
        raise ValueError(
            f"{name}: expected exactly one {_NAV_PLACEHOLDER}, found {count}")
    return html.replace(_NAV_PLACEHOLDER, nav_partial())
