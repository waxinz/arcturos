"""J3 multi-turn replay — ordered message lists with assistant placeholders.

A multi-turn item carries turns like:
    [{"role": "user", "content": "..."},
     {"role": "assistant", "content": None},   <- placeholder, runner fills
     {"role": "user", "content": "..."}]

The runner fills each placeholder from the model's reply to the turns so far,
then continues. One stored eval_result per assistant turn, item_id suffixed
(turn 1 = base id, turn 2 = base#t2, ...).
"""

from __future__ import annotations

from typing import Any

from urllib import request as urlreq
import json


def validate_multiturn_item(item: dict[str, Any]) -> None:
    """Raise ValueError unless the item is a well-formed multi-turn item."""
    turns = item.get("turns")
    if not turns or not isinstance(turns, list):
        raise ValueError("multi-turn item needs a non-empty 'turns' list")
    if item.get("type") != "multi-turn":
        raise ValueError("item type must be 'multi-turn'")
    if not any(t.get("role") == "assistant" for t in turns):
        raise ValueError("turns need at least one assistant placeholder")
    for t in turns:
        if t.get("role") not in ("user", "assistant"):
            raise ValueError(f"bad role: {t.get('role')!r}")
        if t["role"] == "user" and not isinstance(t.get("content"), str):
            raise ValueError("user turns need string 'content'")


def post_openai_chat(target: str, messages: list[dict], api_key: str | None = None) -> dict:
    """POST /v1/chat/completions; base URL may or may not include /v1."""
    base = target.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlreq.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "test", "messages": messages,
            "max_tokens": 512, "temperature": 0.0, "stream": False,
        }).encode(),
        headers=headers, method="POST",
    )
    with urlreq.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def extract_output(resp: dict) -> tuple[str, int, int]:
    """Return (content-with-reasoning-fallback, prompt_tokens, completion_tokens).

    Reasoning models (DeepSeek-V4-Flash, qwen3.8) put text in
    reasoning_content and leave content empty; keep the trail auditable.
    """
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip():
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning:
            content = f"[reasoning] {reasoning}"
    usage = resp.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def replay_multiturn(target: str, item: dict[str, Any],
                     api_key: str | None = None) -> list[dict[str, Any]]:
    """Replay one multi-turn item; return one record per assistant turn.

    Record: {turn_index, item_id (suffixed), output, prompt_tokens,
             completion_tokens, latency_ms}.
    """
    validate_multiturn_item(item)
    base_id = item["id"]
    turns = item["turns"]
    messages: list[dict] = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    records: list[dict[str, Any]] = []
    turn_no = 0
    import time

    for t in turns:
        if t["role"] == "user":
            messages.append({"role": "user", "content": t["content"]})
        elif t["role"] == "assistant":
            turn_no += 1
            t0 = time.monotonic()
            resp = post_openai_chat(target, messages, api_key)
            latency_ms = (time.monotonic() - t0) * 1000
            content, ptok, ctok = extract_output(resp)
            messages.append({"role": "assistant", "content": content})
            item_id = base_id if turn_no == 1 else f"{base_id}#t{turn_no}"
            records.append({
                "turn_index": turn_no, "item_id": item_id,
                "output": content, "prompt_tokens": ptok,
                "completion_tokens": ctok, "latency_ms": round(latency_ms, 1),
            })
    return records
