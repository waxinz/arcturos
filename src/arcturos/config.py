"""Local-config loader (2026-10-01 publish round).

Single source for operator-specific defaults so the public repo ships none.

Lookup precedence (first hit wins per key):
    1. environment        ARCTUROS_BENCH__SERVER_URL, ARCTUROS_BENCH__API_KEY, …
    2. arcturos.local.toml  gitignored; the operator's private overlay
    3. arcturos.toml        committed template (neutral placeholders only)

Sections (both files share the shape):
    [bench]  server_url / model / api_key   — Kick off page + dispatch defaults
    [demo]   live_targets / judge_url       — seed_demo + ab_judge defaults

Secrets note: api_key values originating here flow through memory to the
outbound client and are never persisted (matches the store-wide rule).
"""
from functools import lru_cache
from pathlib import Path
import os
import tomllib

_CANDIDATE_FILES = ("arcturos.local.toml", "arcturos.toml")


def _candidate_paths():
    """Where files may live: the service CWD plus every parent of the
    package (so the repo root is found whether run as a checkout, an
    installed package, or from an arbitrary working directory)."""
    here = Path(__file__).resolve()          # .../src/arcturos/config.py
    roots = [Path.cwd(), *here.parents]
    seen = []
    for root in roots:
        for name in _CANDIDATE_FILES:
            p = root / name
            if p.exists() and p not in seen:
                seen.append(p)
    return seen


def _flatten(d, prefix=()):
    """{bench: {server_url: x}} -> {('bench','server_url'): x}.
    Tuple keys — the same shape the ARCTUROS_* env parser produces."""
    flat = {}
    for k, v in d.items():
        key = prefix + (str(k),)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


@lru_cache(maxsize=1)
def _load_all() -> dict:
    """merged triples: env > local overlay > committed template."""
    merged: dict[tuple, object] = {}

    # 3) committed template
    for path in _candidate_paths():
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for key, value in _flatten(data).items():
            merged.setdefault(key, value)      # local file seen first wins

    # 1) environment — highest precedence, ARCTUROS_BENCH__SERVER_URL shape
    for env_name, env_value in os.environ.items():
        if env_name.startswith("ARCTUROS_") and "__" in env_name:
            key = tuple(env_name[len("ARCTUROS_"):].lower().split("__"))
            merged[key] = env_value

    return {k: v for k, v in merged.items() if v not in ("", None)}


def get(section: str, key: str, default=None):
    """Public accessor: get('bench', 'server_url', 'http://localhost:8000')."""
    return _load_all().get((section.lower(), key.lower()), default)


def get_section(section: str) -> dict:
    """Every key in one section (locals merged over template)."""
    prefix = (section.lower(),)
    return {k[-1]: v for k, v in _load_all().items() if k[:-1] == prefix}