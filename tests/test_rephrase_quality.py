"""The rephrase's gates: its shape, the exact terms it must keep, and the soft warning
for an answer that was not rephrased."""

import pytest
from fastapi import HTTPException

from claude_exam.api.rephrase import _build_system_message, _validate_shape, quality_warnings

ITEM = {
    "correct_response": ["b"],
    "prompt": {
        "question": "<p>Which tool_choice setting forces a tool call?</p>",
        "answers": ["Set `tool_choice` to any so a tool is always called.", "Use plan mode, which only reads files."],
        "feedbacks": ["no", "yes"],
        "explanation": "Rule: scope the tools.",
    },
}

GOOD = {
    "question": "Which tool_choice setting forces a tool call?",
    "answers": ["Pick tool_choice any.", "Use plan mode."],
}


def test_the_shape_must_match_the_original():
    _validate_shape(GOOD, 2)
    for bad in ({**GOOD, "answers": ["one"]}, {**GOOD, "answers": ["Same", "same"]}, {**GOOD, "question": " "}):
        with pytest.raises(HTTPException) as exc:
            _validate_shape(bad, 2)
        assert exc.value.status_code == 502


def test_missing_identifier_is_a_warning():
    parsed = {"question": "Which setting forces a tool call?", "answers": ["Pick any.", "Use plan mode."]}
    assert any("tool_choice" in w for w in quality_warnings(ITEM, parsed))


def test_identifier_kept_gives_no_warning():
    assert quality_warnings(ITEM, GOOD) == []


def test_near_copy_answer_is_a_warning():
    parsed = {**GOOD, "answers": [ITEM["prompt"]["answers"][0], "Use plan mode."]}
    assert any("not rephrased" in w for w in quality_warnings(ITEM, parsed))


def test_system_message_asks_for_question_and_answers_only():
    system = _build_system_message(2)
    assert "{{" not in system
    assert "You are not told which answer is correct" in system
    assert '"feedbacks"' not in system and '"explanation"' not in system
    assert '"<rephrase of answer B' in system and '"<rephrase of answer C' not in system
