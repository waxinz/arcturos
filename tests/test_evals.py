"""evals.py unit tests (v0.2) — item validation, replay, reference checks."""

import pytest

from arcturos import evals


# ----------------------------------------------------------- validate_item --

def test_validate_item_happy_single_turn():
    item = {"id": "q1", "dimension": "reasoning", "prompt": "What is 2+2?"}
    evals.validate_item(item)  # raises when invalid


def test_validate_item_missing_id():
    with pytest.raises(evals.ItemValidationError, match="missing id"):
        evals.validate_item({"prompt": "hi", "dimension": "reasoning"})


def test_validate_item_missing_prompt():
    with pytest.raises(evals.ItemValidationError, match="missing prompt"):
        evals.validate_item({"id": "q1", "dimension": "reasoning"})


def test_validate_item_unknown_dimension():
    with pytest.raises(evals.ItemValidationError, match="dimension"):
        evals.validate_item({"id": "q1", "dimension": "vibes", "prompt": "x"})


def test_validate_item_dimension_free():
    evals.validate_item({"id": "q1", "prompt": "x"})  # dimension optional


def test_validate_item_reference_requires_valid_spec():
    evals.validate_item({"id": "q1", "prompt": "x",
                         "reference": {"check": "numeric", "value": 42}})
    with pytest.raises(evals.ItemValidationError, match="reference"):
        evals.validate_item({"id": "q1", "prompt": "x",
                             "reference": {"check": "bogus"}})
    with pytest.raises(evals.ItemValidationError, match="numeric"):
        evals.validate_item({"id": "q1", "prompt": "x",
                             "reference": {"check": "numeric"}})


def test_validate_item_image_must_be_url_or_path():
    with pytest.raises(evals.ItemValidationError, match="image"):
        evals.validate_item({"id": "q1", "prompt": "x", "image": 42})


def test_validate_item_multiturn_delegates_shape_checks():
    item = {"id": "mt1", "type": "multi-turn", "dimension": "reasoning",
            "turns": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None},
            ]}
    evals.validate_item(item)
    bad = {"id": "mt2", "type": "multi-turn", "dimension": "reasoning",
           "turns": [{"role": "user", "content": 42}]}
    with pytest.raises(evals.ItemValidationError):
        evals.validate_item(bad)
    no_assistant = {"id": "mt3", "type": "multi-turn",
                    "dimension": "reasoning",
                    "turns": [{"role": "user", "content": "hi"}]}
    with pytest.raises(evals.ItemValidationError, match="assistant"):
        evals.validate_item(no_assistant)


def test_validate_suite_rejects_non_dict():
    with pytest.raises(evals.ItemValidationError, match="JSON object"):
        evals.validate_suite(["not", "a", "dict"])


def test_validate_suite_empty_items_ok():
    assert evals.validate_suite({"items": []}) == []


def test_validate_suite_duplicate_ids_rejected():
    suite = {"items": [{"id": "q1", "prompt": "a", "dimension": "reasoning"},
                       {"id": "q1", "prompt": "b", "dimension": "reasoning"}]}
    with pytest.raises(evals.ItemValidationError, match="duplicate"):
        evals.validate_suite(suite)


# --------------------------------------------------------------- replay ----

def _chat(content="ok", prompt_tokens=5, completion_tokens=3):
    return {"choices": [{"message": {"content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens}}


def test_replay_single_turn_posts_messages_and_records(monkeypatch):
    captured = {}

    def fake_post(target, messages, api_key=None, model="test",
                  max_tokens=512, temperature=0.0, **kw):
        captured["messages"] = messages
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        return _chat()

    monkeypatch.setattr("arcturos.multiturn.post_openai_chat", fake_post)
    item = {"id": "q1", "dimension": "reasoning", "prompt": "hello",
            "system": "be terse"}
    rec = evals.replay_single_turn("http://t:8000/v1", item, model="m1",
                                   max_tokens=1024)
    assert rec["item_id"] == "q1"
    assert rec["output"] == "ok"
    assert rec["prompt_tokens"] == 5
    assert rec["completion_tokens"] == 3
    assert rec["latency_ms"] is not None
    assert captured["model"] == "m1"
    assert captured["max_tokens"] == 1024
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["messages"][1] == {"role": "user", "content": "hello"}


def test_replay_single_turn_empty_content_stored_honestly(monkeypatch):
    """Reasoning-model pitfall (ADR 007): some servers return empty content
    with no reasoning fallback. The record stores the empty output as-is —
    honest nulls, never a fabricated answer."""
    monkeypatch.setattr("arcturos.multiturn.post_openai_chat",
                        lambda *a, **k: _chat(content=""))
    item = {"id": "q1", "dimension": "reasoning", "prompt": "hello"}
    rec = evals.replay_single_turn("http://t:8000/v1", item)
    assert rec["output"] == ""


# ------------------------------------------------------- reference checks --

def test_score_output_numeric_last_number_wins():
    passed, detail = evals.score_output(
        "Let me work it out… 3 + 4 = 7", {"check": "numeric", "value": 7})
    assert passed is True
    assert "7" in detail


def test_score_output_numeric_tolerance():
    ref = {"check": "numeric", "value": 100, "tolerance": 5}
    assert evals.score_output("about 104", ref)[0] is True
    assert evals.score_output("about 120", ref)[0] is False
    assert evals.score_output("no digits", ref)[0] is False


def test_score_output_contains():
    ref = {"check": "contains", "value": "Forty-Two"}
    assert evals.score_output("the answer is forty-two", ref)[0] is True
    assert evals.score_output("nope", ref)[0] is False
    ref_cs = {"check": "contains", "value": "Forty", "case_sensitive": True}
    assert evals.score_output("the answer is forty", ref_cs)[0] is False


def test_score_output_exact():
    ref = {"check": "exact", "value": " 42 "}
    assert evals.score_output("42", ref)[0] is True
    assert evals.score_output("42x", ref)[0] is False


def test_score_output_one_of():
    ref = {"check": "one_of", "values": ["yes", "no"]}
    assert evals.score_output("YES", ref)[0] is True
    assert evals.score_output("maybe", ref)[0] is False


def test_score_output_strips_markdown_emphasis():
    """reasoning-005 lesson: '**Monopoly**' is a correct answer to a
    one_of that accepts 'monopoly' — the scorer judges the words, not
    the markdown wrapper."""
    ref = {"check": "one_of", "values": ["monopoly"]}
    assert evals.score_output("**Monopoly**", ref)[0] is True
    ref2 = {"check": "exact", "value": "Monopoly"}
    assert evals.score_output("**Monopoly**", ref2)[0] is True
    assert evals.score_output("`Monopoly`", ref2)[0] is True
    # emphasis is NOT itself part of the expected answer here
    assert evals.score_output("Monopoly**Go**", ref2)[0] is False


def test_score_output_regex():
    ref = {"check": "regex", "pattern": r"\d{1,3}(,\d{3})+"}
    assert evals.score_output("result: 1,024 tokens", ref)[0] is True
    assert evals.score_output("no match", ref)[0] is False


def test_score_output_broken_spec_is_failure_not_crash():
    passed, detail = evals.score_output("x", {"check": "numeric"})
    assert passed is False
    assert "error" in detail.lower() or "reference" in detail.lower()


# ------------------------------------------------------------ image parts --

def test_load_image_part_passthrough_url():
    part = evals.load_image_part("https://example.com/img.png")
    assert part == {"type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"}}


def test_load_image_part_inlines_local_file(tmp_path):
    png = tmp_path / "tiny.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    part = evals.load_image_part(str(png))
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_load_image_part_suite_assets_relative(tmp_path, monkeypatch):
    asset = tmp_path / "pic.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    monkeypatch.setattr(evals, "SUITE_ASSETS_DIR", tmp_path)
    part = evals.load_image_part("pic.png")
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_load_image_part_missing_raises():
    with pytest.raises(FileNotFoundError, match="image not found"):
        evals.load_image_part("definitely-not-here.png")


def test_image_data_url_passthrough():
    assert evals.image_data_url("http://h/i.jpg") == "http://h/i.jpg"
