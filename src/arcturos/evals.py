"""J3 eval core (v0.2) — item validation, deterministic reference checks,
image content parts, single-turn replay with system prompts.

The eval suite is a versioned JSON collection of items. Each item:

    {"id": "core-reasoning-001",
     "type": "single-turn",                # or "multi-turn"
     "dimension": "reasoning",             # the quality axis (ADR 007)
     "category": "algebra",                # freeform sub-category
     "system": "optional system prompt",
     "prompt": "...",
     "image": "images/foo.png",            # optional: URL or suite-relative path
     "tags": ["..."],
     "reference": {"check": "numeric", "value": 42}}   # optional

Reference items are scored at REPORT time (reports.reference_scores) —
never at replay time, so an improved reference spec applies retroactively
without touching the append-only store.

Multi-turn items keep the multiturn.py shape: ordered turns with assistant
placeholders, one stored result per assistant turn (item_id suffixed #t2,
#t3, ...).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

# The quality axes (ADR 007 §1). Fixed allowlist: reports group by these;
# freeform sub-classification stays in `category`. v1.1 (2026-10-15): grew
# from 7 to 13 for the Evals 101 community set — accuracy, factuality,
# tool-calling, agentic, conciseness, extraction join; `knowledge` stays
# for arcturos-core-v1 back-compat (its items report under knowledge).
DIMENSIONS = (
    "reasoning",
    "knowledge",
    "instruction-following",
    "writing",
    "coding",
    "vision",
    "safety",
    "accuracy",
    "factuality",
    "tool-calling",
    "agentic",
    "conciseness",
    "extraction",
)

CHECK_KINDS = ("exact", "numeric", "contains", "one_of", "regex")

# Suite assets (images) resolve against this directory — the canonical home
# of suite JSON files — then the repo root. Absolute paths and http(s) URLs
# pass through as-is.
SUITE_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "qa" / "suites"


class ItemValidationError(ValueError):
    """An eval item failed validation — the suite needs fixing (422)."""


# --------------------------------------------------------------- validation

def validate_item(item: dict[str, Any]) -> None:
    """Raise ItemValidationError unless the item is well-formed (v0.2).

    Multi-turn items delegate shape checks to multiturn's validator.
    Single-turn items need an id, a non-empty prompt, a known dimension,
    and (when present) a valid reference spec / image reference.
    """
    item_id = item.get("id") or item.get("item_id")
    if not item_id:
        raise ItemValidationError(f"item missing id: {item!r}")
    if item.get("type") == "multi-turn":
        from arcturos import multiturn as mt
        try:
            mt.validate_multiturn_item(item)
        except ValueError as exc:
            raise ItemValidationError(f"item {item_id}: {exc}") from exc
    else:
        if not (item.get("prompt") or item.get("content")):
            raise ItemValidationError(f"item {item_id!r} missing prompt")
    dimension = item.get("dimension")
    if dimension is not None:
        if dimension not in DIMENSIONS:
            raise ItemValidationError(
                f"item {item_id!r}: unknown dimension {dimension!r} — "
                f"expected one of {', '.join(DIMENSIONS)}")
    if item.get("reference") is not None:
        validate_reference(item_id, item["reference"])
    if item.get("image") is not None:
        if not isinstance(item["image"], str) or not item["image"].strip():
            raise ItemValidationError(
                f"item {item_id!r}: image must be a non-empty URL or path")


def validate_reference(item_id: str, reference: Any) -> None:
    """Raise ItemValidationError unless the reference check spec is valid."""
    if not isinstance(reference, dict):
        raise ItemValidationError(
            f"item {item_id!r}: reference must be an object with a 'check' key")
    kind = reference.get("check")
    if kind not in CHECK_KINDS:
        raise ItemValidationError(
            f"item {item_id!r}: unknown reference check {kind!r} — "
            f"expected one of {', '.join(CHECK_KINDS)}")
    if kind == "numeric":
        if not isinstance(reference.get("value"), (int, float)) \
                or isinstance(reference.get("value"), bool):
            raise ItemValidationError(
                f"item {item_id!r}: numeric reference needs a numeric 'value'")
        tol = reference.get("tolerance", 0)
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol < 0:
            raise ItemValidationError(
                f"item {item_id!r}: numeric tolerance must be a number >= 0")
        return
    if kind == "one_of":
        values = reference.get("values")
        if not isinstance(values, list) or not values \
                or not all(isinstance(v, str) for v in values):
            raise ItemValidationError(
                f"item {item_id!r}: one_of reference needs a non-empty "
                "list of strings in 'values'")
        return
    # exact / contains / regex all key off a string 'value' (regex may
    # instead carry 'pattern' — accept either spelling)
    value = reference.get("value", reference.get("pattern"))
    if kind == "regex":
        pattern = reference.get("pattern", reference.get("value"))
        if not isinstance(pattern, str) or not pattern:
            raise ItemValidationError(
                f"item {item_id!r}: regex reference needs a 'pattern'")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ItemValidationError(
                f"item {item_id!r}: regex reference does not compile: {exc}")
        return
    if not isinstance(value, str):
        raise ItemValidationError(
            f"item {item_id!r}: {kind} reference needs a string 'value'")


def validate_suite(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a whole suite payload; returns its items (empty is an error
    at replay time, not validation time — empty suites are creatable)."""
    if not isinstance(suite, dict):
        raise ItemValidationError("suite must be a JSON object")
    items = suite.get("items")
    if items is None:
        return []
    if not isinstance(items, list):
        raise ItemValidationError("suite 'items' must be a list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ItemValidationError(f"item is not an object: {item!r}")
        validate_item(item)
        item_id = item.get("id") or item.get("item_id")
        assert item_id is not None  # validate_item raises when absent
        if item_id in seen:
            raise ItemValidationError(f"duplicate item id: {item_id!r}")
        seen.add(item_id)
    return items


# --------------------------------------------------- deterministic checks

_MD_EMPHASIS_RE = re.compile(r"\*\*|__|`")


def _normalize(text: str) -> str:
    """Exact/one_of normalization: trim outer whitespace, casefold, and
    strip markdown emphasis (**) and backticks.

    The emphasis strip is the reasoning-005 lesson from the first real
    run: GLM answered '**Monopoly**' — correct content, but the literal
    asterisks failed a one_of check that accepted only bare 'monopoly'.
    Models wrap answers in markdown emphasis far more often than an
    emphasis character is itself the expected answer; the deterministic
    scorer should judge the words, not the wrapper. (The qwen '\\n\\n'
    preamble is already handled by strip.)
    """
    return _MD_EMPHASIS_RE.sub("", text).strip().casefold()


# LAST number in the text (GSM8K convention: the final answer carries the
# number; earlier numbers are usually working-out). Matches ints, decimals,
# negatives, and thousands separators; currency symbols/% around it are fine.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def build_reference_checker(
        reference: dict[str, Any]) -> Callable[[str], tuple[bool, str]]:
    """Compile a reference spec into a checker(output) -> (passed, detail).

    The checker never raises on odd output text — unparseable output is a
    failed check with an honest detail string, not a crash.
    """
    kind = reference["check"]

    if kind == "exact":
        expected = _normalize(str(reference["value"]))

        def check_exact(output: str) -> tuple[bool, str]:
            ok = _normalize(output or "") == expected
            return ok, f"exact match on {expected!r}"

        return check_exact

    if kind == "numeric":
        expected = float(reference["value"])
        tolerance = float(reference.get("tolerance", 0))

        def check_numeric(output: str) -> tuple[bool, str]:
            found = _last_number(output or "")
            if found is None:
                return False, "no number found in output"
            ok = abs(found - expected) <= tolerance
            return ok, f"expected number {expected} (±{tolerance}), saw {found}"

        return check_numeric

    if kind == "contains":
        needle = str(reference["value"])
        case_sensitive = bool(reference.get("case_sensitive", False))

        def check_contains(output: str) -> tuple[bool, str]:
            hay = output or ""
            if not case_sensitive:
                hay, needle_ = hay.casefold(), needle.casefold()
            else:
                needle_ = needle
            ok = needle_ in hay
            return ok, f"contains {needle!r}"

        return check_contains

    if kind == "one_of":
        accepted = {_normalize(v) for v in reference["values"]}

        def check_one_of(output: str) -> tuple[bool, str]:
            ok = _normalize(output or "") in accepted
            return ok, f"one of {sorted(accepted)!r}"

        return check_one_of

    # regex (validated at suite time; compile defensively anyway)
    pattern = reference.get("pattern", reference.get("value", ""))
    flags_str = str(reference.get("flags", "") or "")
    flags = re.IGNORECASE if "i" in flags_str.lower() else 0

    def check_regex(output: str) -> tuple[bool, str]:
        try:
            ok = re.search(pattern, output or "", flags) is not None
        except re.error as exc:
            return False, f"regex error: {exc}"
        return ok, f"matches /{pattern}/"

    return check_regex


def score_output(output: str, reference: dict[str, Any]) -> tuple[bool, str]:
    """Score one stored output against a reference spec. Never raises."""
    try:
        checker = build_reference_checker(reference)
        return checker(output or "")
    except Exception as exc:  # noqa: BLE001 — a broken spec is a failed check
        return False, f"reference spec error: {exc}"


# ------------------------------------------------------------- image parts

def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def load_image_part(image_ref: str, timeout: float = 15.0) -> dict[str, Any]:
    """Build an OpenAI image_url content part from a suite image reference.

    http(s) URLs pass through untouched. Local paths resolve against
    SUITE_ASSETS_DIR then the repo root (then cwd) and are inlined as
    base64 data URIs — the suite stays self-contained and no operator
    filesystem path leaks to the target server.
    Raises FileNotFoundError/ValueError with an actionable message.
    """
    ref = (image_ref or "").strip()
    if not ref:
        raise ValueError("image reference is empty")
    if ref.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": ref}}
    candidates = [Path(ref)] if Path(ref).is_absolute() else [
        SUITE_ASSETS_DIR / ref,
        SUITE_ASSETS_DIR.parent / ref,
        Path(ref),
    ]
    for candidate in candidates:
        if candidate.is_file():
            data = candidate.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return {"type": "image_url",
                    "image_url": {"url": f"data:{_guess_mime(candidate)};base64,{b64}"}}
    raise FileNotFoundError(
        f"image not found: {image_ref!r} (tried {SUITE_ASSETS_DIR} and repo root)")


def fetch_image_b64(image_ref: str, timeout: float = 15.0) -> str:
    """Fetch a remote image and return its base64 payload (no data: prefix).

    Used by the judge path when a suite references remote images; local
    images reuse load_image_part directly.
    """
    req = urlreq.Request(image_ref)
    with urlreq.urlopen(req, timeout=timeout) as resp:
        mime = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
        data = resp.read()
    return f"data:{mime or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"


def image_data_url(image_ref: str) -> str:
    """Image reference -> data URL string (pass-through for http URLs)."""
    part = load_image_part(image_ref)
    return part["image_url"]["url"]


# ----------------------------------------------------------------- replay

def build_user_content(prompt: str, image_ref: str | None) -> str | list[dict]:
    """OpenAI chat content for one user turn: plain string, or a text+image
    part list when the item carries an image."""
    if not image_ref:
        return prompt
    return [
        {"type": "text", "text": prompt},
        load_image_part(image_ref),
    ]


def replay_single_turn(
    target: str,
    item: dict[str, Any],
    api_key: str | None = None,
    model: str = "test",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_budget: int | None = None,
) -> dict[str, Any]:
    """Replay one single-turn item; return one eval record dict.

    v0.2: honors the item's `system` prompt (pre-v0.2 the runner silently
    dropped it) and carries image items as image_url content parts.
    ``reasoning_effort``/``reasoning_budget`` (2026-10-15, optional) pass
    through to the chat call — per-model reasoning depth for eval runs.
    """
    item_id = item.get("id") or item.get("item_id")
    prompt = item.get("prompt") or item.get("content")
    assert prompt is not None  # validate_item raises when absent
    messages: list[dict] = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user",
                     "content": build_user_content(prompt, item.get("image"))})

    from arcturos import multiturn as mt
    import time
    t0 = time.monotonic()
    resp = mt.post_openai_chat(target, messages, api_key=api_key, model=model,
                               max_tokens=max_tokens, temperature=temperature,
                               reasoning_effort=reasoning_effort,
                               reasoning_budget=reasoning_budget)
    latency_ms = (time.monotonic() - t0) * 1000
    content, ptok, ctok = mt.extract_output(resp)
    return {
        "item_id": item_id,
        "output": content,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "latency_ms": round(latency_ms, 1),
    }


def load_suite_file(path: str | Path) -> dict[str, Any]:
    """Load + validate a suite JSON file (operator/seed helper)."""
    suite = json.loads(Path(path).read_text())
    validate_suite(suite)
    return suite
