"""The icon step: the catalog, parsing the model's picks and its fallbacks."""

import pytest
from fastapi import HTTPException
from test_entities import DOC, GOOD_ICONS, VIEW, FakeIca, reply

from claude_exam.api.icon_catalog import ICON_KEYS, ICONS, KIND_FALLBACK, known
from claude_exam.api.icons import (
    _build_system_message,
    _build_user_message,
    parse_icons,
    pick_icons,
)
from claude_exam.api.schemas import PromptEntities

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


CODE = {"name": "Claude Code", "kind": "actor", "aliases": ["the CLI"]}
EDITS = {"name": "acceptEdits", "kind": "object", "aliases": []}
PLAN = {"name": "plan mode", "kind": "object", "aliases": []}


def entities(question=(CODE,), answers=((EDITS,), (PLAN,))):
    return PromptEntities.model_validate(
        {"question": list(question), "answers": [{"entities": list(a)} for a in answers]}
    )


def icons(result):
    return (
        result.question_icon,
        [e.icon for e in result.question],
        [(a.icon, [e.icon for e in a.entities]) for a in result.answers],
    )


def test_catalog_is_consistent():
    assert len(ICON_KEYS) == len(ICONS) > 100
    assert all(icon.meaning and icon.lucide for icon in ICONS)
    assert set(KIND_FALLBACK) == {"actor", "object", "action"}
    assert known("subagent") == "subagent"
    assert known("nope") is None and known(None) is None and known(3) is None


def test_parse_assigns_the_picked_keys():
    result = parse_icons(GOOD_ICONS, entities(), 2)
    assert icons(result) == ("mode", ["terminal"], [("write", ["write"]), ("plan", ["plan"])])


def test_parse_leaves_the_input_untouched_and_matches_names_ignoring_case():
    source = entities()
    reply_ = {**GOOD_ICONS, "entities": {"claude code": "terminal"}}
    result = parse_icons(reply_, source, 2)
    assert result.question[0].icon == "terminal"
    assert source.question[0].icon is None and source.answers[0].icon is None


def test_unknown_or_missing_entity_pick_falls_back_to_the_kind():
    reply_ = {**GOOD_ICONS, "entities": {"Claude Code": "not-a-key", "plan mode": 7}}
    result = parse_icons(reply_, entities(), 2)
    assert result.question[0].icon == "actor"
    assert [a.entities[0].icon for a in result.answers] == ["object", "object"]


def test_bad_answer_pick_falls_back_to_the_least_shared_entity():
    shared = entities(answers=((CODE, EDITS), (CODE, PLAN), ()))
    reply_ = {**GOOD_ICONS, "question": "nope", "answers": ["nope", None, "nope"]}
    result = parse_icons(reply_, shared, 3)
    # Claude Code is on two answers, so the other entity tells each apart.
    assert [a.icon for a in result.answers] == ["write", "plan", None]
    assert result.question_icon == "terminal"  # the question's first entity


def test_question_without_entities_still_gets_icons():
    result = parse_icons({"entities": {}, "question": "mode", "answers": ["write", "plan"]}, PromptEntities(), 2)
    assert icons(result) == ("mode", [], [("write", []), ("plan", [])])


@pytest.mark.parametrize(
    "reply_",
    [
        {**GOOD_ICONS, "entities": ["terminal"]},
        {**GOOD_ICONS, "answers": ["write"]},
        {**GOOD_ICONS, "answers": "write"},
        {"question": "mode"},
    ],
)
def test_parse_rejects_bad_structure(reply_):
    with pytest.raises(HTTPException) as exc:
        parse_icons(reply_, entities(), 2)
    assert exc.value.status_code == 502


def test_system_message_lists_the_catalog_and_the_answer_count():
    message = _build_system_message(3)
    assert "{{" not in message
    assert all(f"- {icon.key}: {icon.meaning}" in message for icon in ICONS)
    assert '"<key for C>"' in message and '"<key for D>"' not in message
    assert "exactly 3 keys" in message


def test_user_message_lists_each_entity_once_and_hides_the_answer_key():
    message = _build_user_message(VIEW, entities(answers=((CODE, EDITS), (PLAN,))))
    assert message.count("- Claude Code (actor) - also: the CLI") == 1
    assert "- plan mode (object)\n" in message + "\n"
    assert "A. acceptEdits mode" in message and "B. plan mode" in message
    assert DOC["prompt"]["explanation"] not in message and "Section: Domain 3" in message


async def test_pick_icons_returns_entities_with_icons_and_meta():
    ica = FakeIca(reply(GOOD_ICONS))
    result, meta = await pick_icons(ica, VIEW, entities(), "claude-haiku-5-5")
    assert result.answers[1].icon == "plan" and meta.total_tokens == 42
    assert "icon catalog" in ica.calls[0]["system"]


async def test_pick_icons_rejects_a_truncated_reply():
    ica = FakeIca(reply(GOOD_ICONS, finish_reason="length"))
    with pytest.raises(HTTPException) as exc:
        await pick_icons(ica, VIEW, entities(), "claude-haiku-5-5")
    assert exc.value.status_code == 502
