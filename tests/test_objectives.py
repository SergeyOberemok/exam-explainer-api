"""The objectives step: the catalog, the prompt, and parsing the model's tags."""

import copy

import pytest
from bson import ObjectId
from fastapi import HTTPException
from test_entities import FakeIca, reply

from claude_exam.api.objective_catalog import (
    BY_KEY,
    OBJECTIVES,
    for_section,
    known,
    normalize_section,
)
from claude_exam.api.objectives import (
    _build_system_message,
    _build_user_message,
    parse_objectives,
    tag_question,
)

pytestmark = pytest.mark.anyio

SECTION = "Domain 3: Claude Code Configuration & Workflows"

DOC = {
    "_id": ObjectId("6ae4891ec11925a6a80c076b"),
    "section": SECTION,
    "correct_response": ["b"],
    "prompt": {
        "question": "Which mode reads but never edits?",
        "answers": ["acceptEdits mode", "plan mode", "a PreToolUse hook"],
        "feedbacks": ["It edits.", "It blocks edits.", ""],
        "explanation": "Plan mode blocks edits.",
    },
}

GOOD = {
    "primary": "claude-code-planning",
    "secondary": ["agents-hooks"],
    "level": "apply",
    "tempting": ["C"],
    "distractors": [
        {"misconception": "Accepting edits automatically still keeps files unchanged.", "confused_with": None},
        None,
        {"misconception": "A hook is the way to choose a working mode.", "confused_with": "agents-hooks"},
    ],
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def parse(**changes):
    return parse_objectives({**copy.deepcopy(GOOD), **changes}, SECTION, 3, [1])


def test_catalog_is_consistent():
    assert len(BY_KEY) == len(OBJECTIVES) >= 20
    assert all(o.title and o.statement and o.section for o in OBJECTIVES)
    assert known("tools-errors") == "tools-errors"
    assert known("nope") is None and known(None) is None


def test_sections_match_with_or_without_the_domain_prefix():
    assert normalize_section(SECTION) == "Claude Code Configuration & Workflows"
    assert for_section(SECTION) == for_section("Claude Code Configuration & Workflows") != []
    assert for_section("Unknown") == [] and for_section(None) == []


def test_parse_returns_the_fields_to_store():
    assert parse() == {
        "objectives": ["claude-code-planning", "agents-hooks"],
        "level": "apply",
        "tempting": [2],
        "difficulty": 1,
        "distractors": [
            {"confused_with": None, "misconception": GOOD["distractors"][0]["misconception"]},
            None,
            {"confused_with": "agents-hooks", "misconception": GOOD["distractors"][2]["misconception"]},
        ],
    }


def test_parse_trims_only_the_optional_extras():
    wrong = {"misconception": "  x  ", "confused_with": "not-a-key"}
    same = {"misconception": "y", "confused_with": "claude-code-planning"}
    result = parse(
        secondary=["nope", "claude-code-planning", "tools-errors", "tools-errors", "agents-hooks", "context-facts"],
        distractors=[wrong, None, same],
    )
    # Unknown, repeated and main keys go; at most two secondary objectives stay.
    assert result["objectives"] == ["claude-code-planning", "tools-errors", "agents-hooks"]
    # An unknown key, or the main objective itself, is no confusion.
    assert result["distractors"] == [
        {"confused_with": None, "misconception": "x"},
        None,
        {"confused_with": None, "misconception": "y"},
    ]


def test_parse_reads_distractors_keyed_by_answer_letter():
    first, _, last = GOOD["distractors"]
    assert parse(distractors={"C": last, "A": first}) == parse()
    # A null entry for the correct answer is the same as none.
    assert parse(distractors={"A": first, "B": None, "C": last}) == parse()
    for broken in ({"A": first}, {"A": first, "B": first, "C": last}, {"A": first, "C": last, "D": last}):
        with pytest.raises(HTTPException) as exc:
            parse(distractors=broken)
        assert exc.value.status_code == 502


def test_main_objective_may_belong_to_another_section():
    assert parse(primary="tools-errors")["objectives"] == ["tools-errors", "agents-hooks"]


@pytest.mark.parametrize(
    "changes",
    [
        {"primary": "nope"},
        {"primary": None},
        {"secondary": "agents-hooks"},
        {"level": "create"},
        {"tempting": "C"},
        {"tempting": None},
        {"distractors": GOOD["distractors"][:2]},
        {"distractors": "none"},
        # The correct answer must not get a misconception...
        {"distractors": [GOOD["distractors"][0], GOOD["distractors"][0], GOOD["distractors"][2]]},
        # ...and every wrong answer needs one.
        {"distractors": [None, None, GOOD["distractors"][2]]},
        {"distractors": [{"misconception": " "}, None, GOOD["distractors"][2]]},
    ],
)
def test_parse_rejects_what_would_mislabel_the_question(changes):
    with pytest.raises(HTTPException) as exc:
        parse(**changes)
    assert exc.value.status_code == 502


def test_system_message_separates_the_sections_objectives_from_the_rest():
    message = _build_system_message(SECTION, 3, [1])
    assert "{{" not in message
    own, others = message.split("OBJECTIVES OF OTHER SECTIONS")
    assert "- claude-code-planning: " in own and "- tools-errors: " not in own
    assert "- tools-errors: " in others
    assert all(f"- {o.key}: {o.statement}" in message for o in OBJECTIVES)
    assert "exactly these keys: A, C." in message
    # The schema has no entry for the correct answer.
    assert '"<for A>"' in message and '"<for B>"' not in message and '"<for C>"' in message


def test_user_message_carries_the_answer_key():
    message = _build_user_message(DOC, [1])
    assert "Section: Claude Code Configuration & Workflows" in message
    assert "CORRECT ANSWER: B" in message and "  C. a PreToolUse hook" in message
    assert "  B. It blocks edits." in message and "Plan mode blocks edits." in message


async def test_tag_question_returns_fields_and_meta():
    ica = FakeIca(reply(GOOD))
    fields, meta = await tag_question(ica, DOC, "claude-haiku-5-5")
    assert fields["objectives"][0] == "claude-code-planning" and meta.total_tokens == 42
    assert "learning objective" in ica.calls[0]["system"]


async def test_tag_question_rejects_a_truncated_reply():
    with pytest.raises(HTTPException) as exc:
        await tag_question(FakeIca(reply(GOOD, finish_reason="length")), DOC, "claude-haiku-5-5")
    assert exc.value.status_code == 502


@pytest.mark.parametrize(
    "doc",
    [
        {**DOC, "section": "A section without objectives"},
        {**DOC, "correct_response": []},
        {**DOC, "prompt": {"question": "q", "answers": []}},
    ],
)
async def test_untaggable_question_is_refused_without_calling_the_model(doc):
    ica = FakeIca(reply(GOOD))
    with pytest.raises(HTTPException) as exc:
        await tag_question(ica, doc, "claude-haiku-5-5")
    assert exc.value.status_code == 422 and ica.calls == []
