"""J3 multi-turn replay tests — validation, placeholder filling, reasoning fallback."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arcturos.multiturn import (  # noqa: E402
    extract_output, replay_multiturn, validate_multiturn_item,
)


def _item(**over):
    base = {
        "id": "mt-1",
        "type": "multi-turn",
        "system": "You are helpful.",
        "turns": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Q2"},
        ],
    }
    base.update(over)
    return base


def test_validate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        validate_multiturn_item({"type": "multi-turn", "id": "x"})
    with pytest.raises(ValueError):
        validate_multiturn_item({"type": "multi-turn", "id": "x",
                                 "turns": [{"role": "user"}]})
    with pytest.raises(ValueError):
        validate_multiturn_item({"type": "multi-turn", "id": "x",
                                 "turns": [{"role": "user"},
                                           {"role": "assistant", "content": None}]})
    validate_multiturn_item(_item())  # happy shape passes


def _fake_resp(content, ptok=10, ctok=5):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}}


def test_replay_fills_placeholders_and_suffixes_ids():
    # MagicMock keeps a reference to the same mutable messages list, so
    # call_args_list inspection shows the FINAL state. Snapshot per call.
    snapshots: list[list] = []

    def spy(target, messages, api_key=None, **_kw):
        snapshots.append(list(messages))
        responses = [_fake_resp("A1"), _fake_resp("A2")]
        return responses[len(snapshots) - 1]

    with patch("arcturos.multiturn.post_openai_chat", side_effect=spy):
        item = _item(turns=[
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": None},
        ])
        recs = replay_multiturn("http://fake:8000", item)
    assert [r["item_id"] for r in recs] == ["mt-1", "mt-1#t2"]
    assert [r["output"] for r in recs] == ["A1", "A2"]
    # first call: system + Q1; second call carries A1 forward
    assert snapshots[0] == [{"role": "system", "content": "You are helpful."},
                           {"role": "user", "content": "Q1"}]
    assert snapshots[1] == [{"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Q1"},
                            {"role": "assistant", "content": "A1"},
                            {"role": "user", "content": "Q2"}]
    assert recs[0]["latency_ms"] >= 0


def test_extract_output_reasoning_fallback():
    resp = {"choices": [{"message": {"content": "", "reasoning_content": "thinking..."}}],
            "usage": {}}
    out, pt, ct = extract_output(resp)
    assert out == "[reasoning] thinking..."
    assert pt == 0
    out2, _, _ = extract_output(_fake_resp("plain"))
    assert out2 == "plain"


def test_extract_output_null_usage():
    """tabbyAPI returns usage: null (key present, value None) — extract must
    not explode and must fall back to 0 tokens (regression from live eval
    dispatch against ruapehu GLM-5.3-Flash exl3, 2026-09-21)."""
    resp = {"choices": [{"message": {"content": "ok", "reasoning_content": None}}],
            "usage": None}
    out, pt, ct = extract_output(resp)
    assert out == "ok"
    assert pt == 0
    assert ct == 0


def test_replay_multi_assistant_placeholder_count():
    with patch("arcturos.multiturn.post_openai_chat") as mp:
        mp.side_effect = [_fake_resp("R1"), _fake_resp("R2"), _fake_resp("R3")]
        item = _item(turns=[
            {"role": "user"}, {"role": "assistant", "content": None},
            {"role": "user"}, {"role": "assistant", "content": None},
            {"role": "user"}, {"role": "assistant", "content": None},
            {"role": "user"},
        ])
        # shorthand dicts need content keys
        item["turns"] = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "d"},
        ]
        recs = replay_multiturn("http://fake:8000", item)
    assert len(recs) == 3
    assert recs[-1]["item_id"] == "mt-1#t3"
