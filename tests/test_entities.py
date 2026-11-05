import json

import pytest
from bson import ObjectId
from fastapi import HTTPException

from claude_exam.api.entities import (
    _build_system_message,
    _build_user_message,
    enrich_and_store,
    extract_entities,
    parse_entities,
)
from claude_exam.api.schemas import PromptEntities, SimplifiedPrompt
from claude_exam.api.simplify import simplify_and_store

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


QUESTION_ID = ObjectId("6ae4891ec11925a6a80c076b")

DOC = {
    "_id": QUESTION_ID,
    "section": "Domain 3",
    "assessment_type": "multiple-choice",
    "source": "x.json",
    "correct_response": ["b"],
    "related_lectures": [],
    "prompt": {
        "question": "<p>Which mode?</p>",
        "answers": ["acceptEdits mode", "plan mode"],
        "feedbacks": ["no", "yes"],
        "explanation": "Plan mode blocks edits.",
    },
}

SIMPLIFIED = SimplifiedPrompt(
    question="Which mode lets Claude Code read but not edit?",
    answers=["acceptEdits mode", "plan mode"],
    feedbacks=["wrong", "right"],
    explanation="Plan mode blocks edits.",
)

GOOD_ENTITIES = {
    "entities": [
        {"id": "e1", "name": "Claude Code", "kind": "actor", "aliases": []},
        {"id": "e2", "name": "acceptEdits", "kind": "object", "aliases": ["acceptEdits mode"]},
        {"id": "e3", "name": "plan mode", "kind": "object"},
    ],
    "question_entities": ["e1"],
    "answers_entities": [["e2"], ["e3"]],
}

GOOD_SIMPLIFIED = {
    "question": SIMPLIFIED.question,
    "answers": SIMPLIFIED.answers,
    "feedbacks": SIMPLIFIED.feedbacks,
    "explanation": SIMPLIFIED.explanation,
}


def names(entities):
    return [e.name for e in entities]


def test_old_catalog_shape_is_still_readable():
    old = {
        "entities": [{"id": "e1", "name": "A", "kind": "actor", "aliases": []}],
        "question": ["e1"],
        "answers": [["e1"], []],
    }
    read = PromptEntities(**old)
    assert names(read.question) == ["A"]
    assert names(read.answers[0].entities) == ["A"] and read.answers[1].entities == []


def reply(payload, finish_reason="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"total_tokens": 42},
    }


class FakeIca:
    """Returns the queued replies in order; records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def chat_model(self, model_id, prompt, system=None, max_tokens=None):
        self.calls.append({"model_id": model_id, "prompt": prompt, "system": system})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeCollection:
    """Just enough of AsyncCollection for the simplified-question upsert."""

    def __init__(self, *docs):
        self.docs = list(docs)

    def _match(self, query):
        return next(
            (d for d in self.docs if all(d.get(k) == v for k, v in query.items())), None
        )

    async def find_one(self, query):
        doc = self._match(query)
        return None if doc is None else {**doc}  # a copy, like a real driver

    async def find_one_and_update(self, query, update, upsert=False, return_document=None):
        doc = self._match(query)
        if doc is None:
            if not upsert:
                return None
            doc = {**query, **update.get("$setOnInsert", {})}
            self.docs.append(doc)
        doc.update(update.get("$set", {}))
        return doc


# --- parse_entities -------------------------------------------------------------


def test_parse_valid_output():
    result = parse_entities(GOOD_ENTITIES, 2)
    assert names(result.question) == ["Claude Code"]
    assert [names(a.entities) for a in result.answers] == [["acceptEdits"], ["plan mode"]]
    assert result.answers[0].entities[0].aliases == ["acceptEdits mode"]
    assert result.answers[0].entities[0].kind == "object"


def test_parse_ignores_unreferenced_entity():
    data = {**GOOD_ENTITIES, "entities": GOOD_ENTITIES["entities"] + [
        {"id": "e4", "name": "Orphan", "kind": "object"}
    ]}
    result = parse_entities(data, 2)
    assert "Orphan" not in names(result.question) + [
        n for a in result.answers for n in names(a.entities)
    ]


def test_parse_repeats_entity_where_mentioned_and_drops_self_alias():
    data = {
        "entities": [{"id": "e1", "name": "MCP", "kind": "object", "aliases": ["mcp", "Model Context Protocol", "MCP"]}],
        "question_entities": ["e1"],
        "answers_entities": [["e1"], []],
    }
    result = parse_entities(data, 2)
    assert names(result.question) == ["MCP"] == names(result.answers[0].entities)
    assert result.answers[0].entities[0].aliases == ["Model Context Protocol"]
    assert result.answers[1].entities == []


def test_parse_dedupes_repeated_refs():
    data = {**GOOD_ENTITIES, "question_entities": ["e1", "e1"]}
    assert names(parse_entities(data, 2).question) == ["Claude Code"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {**d, "answers_entities": [["e2"]]},  # wrong answer count
        lambda d: {**d, "question_entities": "e1"},  # not a list
        lambda d: {**d, "entities": "nope"},
        lambda d: {**d, "entities": [{"id": "e1", "name": "A", "kind": "thing"}]},  # bad kind
        lambda d: {**d, "entities": d["entities"] + [{"id": "e1", "name": "B", "kind": "actor"}]},
        lambda d: {**d, "entities": [{"id": "e1", "name": " ", "kind": "actor"}]},
    ],
)
def test_parse_rejects_bad_structure(mutate):
    with pytest.raises(HTTPException) as exc:
        parse_entities(mutate(GOOD_ENTITIES), 2)
    assert exc.value.status_code == 502


def test_parse_drops_dangling_refs_instead_of_failing():
    data = {**GOOD_ENTITIES, "answers_entities": [["e2", "e9"], ["e3"]]}
    assert [names(a.entities) for a in parse_entities(data, 2).answers] == [["acceptEdits"], ["plan mode"]]


def test_parse_trims_overflow():
    many = [{"id": f"e{i}", "name": f"N{i}", "kind": "object"} for i in range(1, 31)]
    data = {
        "entities": many,
        "question_entities": ["e1", "e17", "e30"],  # beyond 16 entities: trimmed away
        "answers_entities": [[f"e{i}" for i in range(1, 9)], ["e2"]],
    }
    result = parse_entities(data, 2)
    assert names(result.question) == ["N1"]  # e30 was trimmed away
    assert names(result.answers[0].entities) == ["N1", "N2", "N3", "N4"]
    assert all(n not in names(result.answers[1].entities) for n in ["N30"])


# --- prompts --------------------------------------------------------------------


def test_system_message_has_exact_answer_count_and_no_placeholders():
    system = _build_system_message(3)
    assert "{{" not in system
    assert "exactly 3 lists" in system
    assert system.count("<ids for answer") == 3


def test_user_message_labels_both_sources_and_cleans_html():
    message = _build_user_message(DOC, SIMPLIFIED)
    assert "ORIGINAL QUESTION:" in message and "SIMPLIFIED QUESTION:" in message
    assert "<p>" not in message


def test_user_message_puts_original_first_and_keeps_both_wordings():
    simplified = SimplifiedPrompt(
        question="Which setting stops edits?",
        answers=["The mode that auto-approves edits", "The read-only planning mode"],
        feedbacks=["", ""],
        explanation="",
    )
    message = _build_user_message(DOC, simplified)
    original, _, simple = message.partition("SIMPLIFIED QUESTION:")
    assert message.index("ORIGINAL QUESTION:") < message.index("SIMPLIFIED QUESTION:")
    assert "ORIGINAL ANSWERS:\n  A. acceptEdits mode\n  B. plan mode" in original
    assert "Which mode?" in original and "Which setting stops edits?" in simple
    assert "  B. The read-only planning mode" in simple
    assert "acceptEdits" not in simple


def test_system_message_names_original_as_primary_and_refs_either_version():
    system = _build_system_message(2)
    assert "ORIGINAL text is the primary source" in system
    assert "ORIGINAL or the SIMPLIFIED question" in system


# --- extract_entities -----------------------------------------------------------


async def test_extract_entities_parses_fenced_reply_and_returns_meta():
    ica = FakeIca(reply("```json\n" + json.dumps(GOOD_ENTITIES) + "\n```"))
    entities, meta = await extract_entities(ica, DOC, SIMPLIFIED, "claude-haiku-5-5")
    assert [names(a.entities) for a in entities.answers] == [["acceptEdits"], ["plan mode"]]
    assert meta.total_tokens == 42 and meta.model_id == "claude-haiku-5-5"


async def test_extract_entities_rejects_truncated_reply():
    ica = FakeIca(reply(GOOD_ENTITIES, finish_reason="length"))
    with pytest.raises(HTTPException) as exc:
        await extract_entities(ica, DOC, SIMPLIFIED, "claude-haiku-5-5")
    assert exc.value.status_code == 502


async def test_extract_entities_rejects_non_json():
    ica = FakeIca(reply("no entities here"))
    with pytest.raises(HTTPException) as exc:
        await extract_entities(ica, DOC, SIMPLIFIED, "claude-haiku-5-5")
    assert exc.value.status_code == 502


# --- simplify_and_store ---------------------------------------------------------


async def test_simplify_and_store_saves_entities():
    ica = FakeIca(reply(GOOD_SIMPLIFIED), reply(GOOD_ENTITIES))
    simplified = FakeCollection()
    stored = await simplify_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    assert len(ica.calls) == 2
    assert names(stored.entities.question) == ["Claude Code"]
    assert stored.entities_ica.total_tokens == 42


async def test_entity_failure_is_non_fatal():
    ica = FakeIca(reply(GOOD_SIMPLIFIED), reply("garbage"))
    simplified = FakeCollection()
    stored = await simplify_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    assert stored.prompt.question == SIMPLIFIED.question
    assert stored.entities.question == [] and stored.entities_ica is None


async def test_simplify_and_store_can_skip_entities():
    ica = FakeIca(reply(GOOD_SIMPLIFIED))
    stored = await simplify_and_store(
        ica, FakeCollection(), DOC, "claude-haiku-5-5", with_entities=False
    )
    assert len(ica.calls) == 1 and stored.entities.question == []


# --- enrich_and_store (backfill) -------------------------------------------------


async def test_backfill_updates_only_entity_fields():
    ica = FakeIca(reply(GOOD_SIMPLIFIED), reply("garbage"), reply(GOOD_ENTITIES))
    simplified = FakeCollection()
    first = await simplify_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    assert first.entities.question == []

    enriched = await enrich_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    assert [names(a.entities) for a in enriched.entities.answers] == [["acceptEdits"], ["plan mode"]]
    assert enriched.prompt == first.prompt
    assert enriched.created_at == first.created_at
    assert enriched.ica == first.ica


async def test_resimplify_failure_keeps_entities_only_if_text_unchanged():
    simplified = FakeCollection()
    ica = FakeIca(reply(GOOD_SIMPLIFIED), reply(GOOD_ENTITIES))
    await simplify_and_store(ica, simplified, DOC, "claude-haiku-5-5")

    same = FakeIca(reply(GOOD_SIMPLIFIED), reply("garbage"))
    kept = await simplify_and_store(same, simplified, DOC, "claude-haiku-5-5")
    assert names(kept.entities.question) == ["Claude Code"] and kept.entities_ica is not None

    changed = {**GOOD_SIMPLIFIED, "question": "A different retelling."}
    other = FakeIca(reply(changed), reply("garbage"))
    wiped = await simplify_and_store(other, simplified, DOC, "claude-haiku-5-5")
    assert wiped.entities.question == [] and wiped.entities_ica is None


async def test_entity_failure_of_any_kind_is_non_fatal():
    ica = FakeIca(reply(GOOD_SIMPLIFIED), RuntimeError("network"))
    stored = await simplify_and_store(ica, FakeCollection(), DOC, "claude-haiku-5-5")
    assert stored.entities.question == []


async def test_backfill_conflicts_when_record_changed_meanwhile():
    simplified = FakeCollection()
    await simplify_and_store(
        FakeIca(reply(GOOD_SIMPLIFIED), reply("garbage")), simplified, DOC, "claude-haiku-5-5"
    )

    class Racing(FakeIca):
        async def chat_model(self, *args, **kwargs):
            simplified.docs[0]["updated_at"] = "rewritten"  # re-simplified meanwhile
            return await super().chat_model(*args, **kwargs)

    with pytest.raises(HTTPException) as exc:
        await enrich_and_store(Racing(reply(GOOD_ENTITIES)), simplified, DOC, "claude-haiku-5-5")
    assert exc.value.status_code == 409


async def test_backfill_errors_propagate():
    simplified = FakeCollection()
    with pytest.raises(HTTPException) as exc:
        await enrich_and_store(FakeIca(), simplified, DOC, "claude-haiku-5-5")
    assert exc.value.status_code == 404  # not simplified yet

    ica = FakeIca(reply(GOOD_SIMPLIFIED), reply(GOOD_ENTITIES), reply("garbage"))
    await simplify_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    with pytest.raises(HTTPException) as exc:
        await enrich_and_store(ica, simplified, DOC, "claude-haiku-5-5")
    assert exc.value.status_code == 502
