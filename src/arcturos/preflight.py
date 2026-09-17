"""Health preflight (J6) — verify a target before dispatching work.

Three checks, each with an actionable failure message:
  reachable   — HTTP GET succeeds
  tokenize    — POST /tokenize with a tiny body returns 200 (llama.cpp native)
  chat        — POST /v1/chat/completions returns 200 (OpenAI-compatible)

The bench path needs tokenize + completion-native; the eval path needs chat.
Power visibility is checked separately (power.py) — never here.

Pure httpx with injectable transport for tests. Never raises: returns a
PreflightResult with per-check detail; dispatch calls ok() first.
"""

from __future__ import annotations

import httpx


class PreflightResult(dict):
    """dict with a convenience ok() — checks carry name + ok + detail."""

    def ok(self) -> bool:
        return all(c["ok"] for c in self["checks"])


def _health_url(base_url: str) -> str:
    """/health lives at the service root, not under /v1 — strip that suffix
    so OpenAI-compatible bases (http://host:4000/v1) probe the right place."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/health"


def _check_reachable(base_url: str, client: httpx.Client) -> dict:
    try:
        resp = client.get(_health_url(base_url))
        ok = resp.status_code == 200
        detail = "GET /health -> 200" if ok else f"GET /health -> HTTP {resp.status_code}"
        if not ok:
            detail += " (server up but /health unhappy; may still work)"
    except httpx.HTTPError as exc:
        ok = False
        detail = f"GET /health failed: {exc}"
    return {"name": "reachable", "ok": ok, "detail": detail}


def _check_tokenize(base_url: str, client: httpx.Client) -> dict:
    try:
        resp = client.post(base_url.rstrip("/") + "/tokenize",
                           json={"content": "preflight probe"})
        ok = resp.status_code == 200
        detail = f"POST /tokenize -> {resp.status_code}" if ok else \
            f"POST /tokenize -> HTTP {resp.status_code} (native llama.cpp endpoint missing?)"
        try:
            tokens = resp.json().get("tokens")
            detail += f" · {len(tokens) if isinstance(tokens, list) else '?'} tokens"
        except ValueError:
            detail += " · non-JSON body"
        return {"name": "tokenize", "ok": ok, "detail": detail}
    except httpx.HTTPError as exc:
        return {"name": "tokenize", "ok": False, "detail": f"POST /tokenize failed: {exc}"}


def _chat_url(base_url: str) -> str:
    """Chat endpoint, tolerant of a /v1 suffix: http://host:4000/v1 ->
    http://host:4000/v1/chat/completions (never /v1/v1/...)."""
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def _check_chat(base_url: str, client: httpx.Client, model: str | None) -> dict:
    try:
        body: dict = {"messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1}
        if model:
            body["model"] = model
        resp = client.post(_chat_url(base_url), json=body)
        ok = resp.status_code == 200
        if ok:
            detail = f"POST /v1/chat/completions -> 200 (model={model or 'server-default'})"
        else:
            detail = (f"POST /v1/chat/completions -> HTTP {resp.status_code}"
                      f" {resp.text[:120]!r}")
            if resp.status_code in (400, 404) and model:
                detail += " (model name may be wrong for this target)"
        return {"name": "chat", "ok": ok, "detail": detail}
    except httpx.HTTPError as exc:
        return {"name": "chat", "ok": False, "detail": f"POST /v1/chat/completions failed: {exc}"}


def preflight(base_url: str, kind: str = "bench", model: str | None = None,
              transport=None, timeout: float = 20.0) -> PreflightResult:
    """Run the checks relevant to the dispatch kind.

    kind='bench' (native llama.cpp): reachable + tokenize.
    kind='eval'  (OpenAI-compatible): reachable + chat.
    """
    if kind not in ("bench", "eval"):
        raise ValueError(f"unknown preflight kind: {kind!r}")
    with httpx.Client(timeout=timeout,
                      transport=transport) as client:
        checks = [_check_reachable(base_url, client)]
        checks.append(_check_tokenize(base_url, client) if kind == "bench"
                      else _check_chat(base_url, client, model))
    return PreflightResult(target=base_url, kind=kind, checks=checks)
