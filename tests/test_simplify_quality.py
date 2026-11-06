import pytest
from fastapi import HTTPException

from claude_exam.api.simplify import (
    _build_system_message,
    _validate_shape,
    quality_warnings,
)

ITEM = {
    "correct_response": ["b"],
    "prompt": {
        "question": "<p>Which tool_choice setting forces a tool call?</p>",
        "answers": ["Set `tool_choice` to any so a tool is always called.", "Use plan mode, which only reads files."],
    },
}

GOOD = {
    "question": "Which tool_choice setting forces a tool call?",
    "answers": ["Pick tool_choice any.", "Use plan mode."],
    "feedbacks": ["Incorrect. It does not pick the tool.", "Correct. It blocks edits."],
    "explanation": "Rule: scope the tools.",
}


def test_verdict_must_match_correct_answer():
    _validate_shape(GOOD, 2, ["b"])
    swapped = {**GOOD, "feedbacks": [GOOD["feedbacks"][1], GOOD["feedbacks"][0]]}
    with pytest.raises(HTTPException) as exc:
        _validate_shape(swapped, 2, ["b"])
    assert exc.value.status_code == 502


def test_verdict_check_is_skipped_without_correct_letters():
    _validate_shape({**GOOD, "feedbacks": ["no", "yes"]}, 2)


def test_missing_identifier_is_a_warning():
    parsed = {**GOOD, "question": "Which setting forces a tool call?",
              "answers": ["Pick any.", "Use plan mode."]}
    warnings = quality_warnings(ITEM, parsed)
    assert any("tool_choice" in w for w in warnings)


def test_identifier_kept_gives_no_warning():
    assert quality_warnings(ITEM, GOOD) == []


def test_near_copy_answer_is_a_warning():
    parsed = {**GOOD, "answers": [ITEM["prompt"]["answers"][0], "Use plan mode."]}
    assert any("not simplified" in w for w in quality_warnings(ITEM, parsed))


def test_system_message_has_new_rules_and_no_placeholders():
    system = _build_system_message(2)
    assert "{{" not in system
    assert 'start each feedback with "Correct."' in system
    assert "evaluator-optimizer" in system
