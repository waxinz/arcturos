"""Committed suite files stay valid (2026-10-15).

The suite JSON files under qa/suites/ are first-class artifacts: the
registry imports them, the public mirror ships them, and eval reports
key off their ids. This test runs the REAL validator over every
committed suite and pins the Evals 101 contract (100 items, 12
dimensions, image assets present on disk) so asset rot or accidental
edits surface in CI, not at replay time.
"""

import json
from pathlib import Path

import pytest

from arcturos import evals as evals_mod

SUITES_DIR = Path(__file__).resolve().parent.parent / "qa" / "suites"


def _load(name: str) -> dict:
    path = SUITES_DIR / name
    assert path.exists(), f"missing committed suite: {path}"
    return json.loads(path.read_text())


def test_committed_suites_pass_validator():
    suites = sorted(SUITES_DIR.glob("*.json"))
    assert suites, "no committed suites found"
    for path in suites:
        payload = json.loads(path.read_text())
        try:
            items = evals_mod.validate_suite(payload)
        except evals_mod.ItemValidationError as exc:
            pytest.fail(f"{path.name}: {exc}")
        assert isinstance(items, list)


def test_evals101_contract():
    suite = _load("evals-101-v1.json")
    assert suite["suite"] == "evals-101"
    items = suite["items"]
    assert len(items) == 100
    dims = {i["dimension"] for i in items}
    expected = {"accuracy", "factuality", "reasoning", "instruction-following",
                "writing", "coding", "tool-calling", "agentic", "conciseness",
                "extraction", "vision", "safety"}
    assert dims == expected, dims - expected
    # coverage: every dimension has >= 5 items
    from collections import Counter
    counts = Counter(i["dimension"] for i in items)
    assert all(c >= 5 for c in counts.values()), counts
    # mix: multi-turn present, text-dominant with a real image slice,
    # and a meaningful deterministic-reference share
    assert sum(1 for i in items if i["type"] == "multi-turn") >= 10
    imgs = [i["image"] for i in items if i.get("image")]
    assert 5 <= len(imgs) <= 15
    assert sum(1 for i in items if i.get("reference")) >= 30
    # every referenced image resolves to a real file in the repo
    root = SUITES_DIR
    for img in imgs:
        assert not img.startswith("http"), \
            "committed suites should use repo assets, not remote URLs"
        assert (root / img).exists(), f"missing suite asset: {img}"
    # ids are unique and prefixed consistently
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("e101-") for i in ids)


def test_core_v1_still_validates_with_extended_enum():
    """The 2026-10-15 enum extension must not break the legacy suite."""
    suite = _load("arcturos-core-v1.json")
    items = evals_mod.validate_suite(suite)
    assert len(items) > 0
    assert all(i.get("dimension") in evals_mod.DIMENSIONS for i in items
               if i.get("dimension"))
