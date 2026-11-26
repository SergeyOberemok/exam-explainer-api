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
    assert message.count("- Claude Code: Claude Code (actor) - also: the CLI") == 1
    assert "- plan mode: plan mode (object)\n" in message
    assert "A. acceptEdits mode" in message and "B. plan mode" in message
    assert DOC["prompt"]["explanation"] not in message and "Section: Domain 3" in message


async def test_pick_icons_returns_entities_with_icons_and_meta():
    ica = FakeIca(reply(GOOD_ICONS))
    result, picked, meta = await pick_icons(ica, VIEW, entities(), "claude-haiku-5-5")
    assert result.answers[1].icon == "plan" and meta.total_tokens == 42
    assert picked.answers == ["write", "plan"] and picked.question == "mode"
    assert "icon catalog" in ica.calls[0]["system"]


async def test_pick_icons_rejects_a_truncated_reply():
    ica = FakeIca(reply(GOOD_ICONS, finish_reason="length"))
    with pytest.raises(HTTPException) as exc:
        await pick_icons(ica, VIEW, entities(), "claude-haiku-5-5")
    assert exc.value.status_code == 502


def test_icons_are_not_picked_for_a_decomposition_of_an_earlier_text():
    import asyncio

    from fastapi import HTTPException

    from claude_exam.api.icons import assign_and_store

    class Store:
        async def find_one(self, query):
            return {"question_id": query["question_id"], "text_hash": "old", "entities": {}}

    class Ica:
        calls = 0

    doc = {"_id": "q1", "prompt": {"question": "Which mode?", "answers": ["a", "b"]}}
    try:
        asyncio.run(assign_and_store(Ica(), Store(), doc, "claude-haiku-5-5"))
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("expected 404")


def test_the_question_icon_may_not_point_at_an_answer():
    from claude_exam.api.question_view import key_free

    # The same key as an answer card: the question's picture would favour that answer.
    result = parse_icons({**GOOD_ICONS, "question": "plan"}, entities(), 2)
    assert result.question_icon == "terminal"
    # A key only an answer talks about ("plan" for "plan mode"; the stem says "mode").
    reply_ = {**GOOD_ICONS, "question": "plan", "answers": ["write", "checklist"]}
    assert parse_icons(reply_, entities(), 2).question_icon == "plan"
    assert parse_icons(reply_, entities(), 2, key_free(DOC)).question_icon == "terminal"
    # A key the stem itself names stays.
    assert parse_icons(GOOD_ICONS, entities(), 2, key_free(DOC)).question_icon == "mode"


# --- picking by decomposition id: every entity (concepts too) and every action ---------


def _decomposed():
    from test_jobs import TAGGABLE, decomposition_with_plan_mode, in_chapter

    from claude_exam.api.question_view import key_free
    from claude_exam.api.schemas import QuestionDecomposition

    d = decomposition_with_plan_mode()
    d["entities"].append(
        {"id": "e4", "name": "Review Gate", "type": "system", "aliases": [], "description": "Holds edits",
         "source_status": "inferred", "evidence": ""}
    )
    # The one entity the scenario paragraph names.
    d["entities"].append(
        {"id": "e5", "name": "Claude Code", "type": "actor", "aliases": [], "description": "Runs in CI",
         "source_status": "inferred", "evidence": ""}
    )

    def event(id, name, sequence, actor=None, target=None):
        return {"id": id, "name": name, "actor_id": actor, "target_id": target, "input_ids": [],
                "output_ids": [], "sequence": sequence, "source_status": "inferred", "evidence": ""}

    d["events"] = [
        event("v2", "applies the edits", 2, actor="e5", target="e3"),
        event("v1", "opens the pull request", 1, actor="e5"),
        event("v3", "reviews before editing", None, actor="e4"),
    ]
    return key_free(in_chapter(section=TAGGABLE)), QuestionDecomposition.model_validate(d)


def test_the_prompt_lists_every_entity_and_the_actions_that_happened_by_id():
    from claude_exam.api.icons import listed_actions

    q, d = _decomposed()
    message = _build_user_message(q, PromptEntities(), d)
    assert "- e2: Ambiguity (concept)" in message and "- e3: plan mode (object)" in message
    # In order; the one without a sequence (what should happen) is left out, and a
    # reference to an entity only an answer names is gone (plan mode is answer B's).
    assert "- v1: opens the pull request (by Claude Code)\n- v2: applies the edits (by Claude Code)" in message
    assert "v3" not in message
    assert [e.id for e in listed_actions(d, q)] == ["v1", "v2"]


def test_concepts_and_actions_get_the_picked_keys_or_fall_back_by_type():
    from claude_exam.api.icons import parse_picks

    q, d = _decomposed()
    reply_ = {
        "entities": {"e1": "agent", "e2": "rule", "e3": "plan", "e4": "nope"},
        "actions": {"v1": "pull-request", "v2": 5},
        "question": "mode",
        "answers": ["write", "plan"],
    }
    derived = PromptEntities.model_validate(
        {"question": [{"name": "Agent", "kind": "actor", "ref": "e1"}],
         "answers": [{"entities": []}, {"entities": [{"name": "plan mode", "kind": "object", "ref": "e3"}]}]}
    )
    with_icons, picked = parse_picks(reply_, derived, 2, q, d)
    # The concept gets the model's pick, the system with an unknown one its type's.
    assert picked.entities == {"e1": "agent", "e2": "rule", "e3": "plan", "e4": "server", "e5": "actor"}
    assert picked.actions == {"v1": "pull-request", "v2": "action"}
    # /v1's visual entities carry the same keys, by ref.
    assert with_icons.question[0].icon == "agent" and with_icons.answers[1].entities[0].icon == "plan"
    # Without a pick at all a concept is drawn as an idea.
    _, bare = parse_picks({"entities": {}, "question": "mode", "answers": ["write", "plan"]}, derived, 2, q, d)
    assert bare.entities["e2"] == "idea" and bare.actions == {"v1": "action", "v2": "action"}


def test_actions_must_be_an_object():
    q, d = _decomposed()
    with pytest.raises(HTTPException) as exc:
        parse_icons({**GOOD_ICONS, "actions": ["run"]}, entities(), 2, q, d)
    assert exc.value.status_code == 502
