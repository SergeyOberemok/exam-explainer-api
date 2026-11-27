import json

import pytest
from bson import ObjectId
from fastapi import HTTPException

"""The visual entities: parsing a catalog reply (grounding, stable actors, limits) and
deriving them from a decomposition, with no model call."""

from claude_exam.api.entities import derive_entities, parse_entities
from claude_exam.api.question_view import key_free
from claude_exam.api.schemas import PromptEntities, QuestionDecomposition

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

# The key-free view of DOC: what the icon and entity steps read.
VIEW = key_free(DOC)

GOOD_ENTITIES = {
    "entities": [
        {"id": "e1", "name": "Claude Code", "kind": "actor", "aliases": []},
        {"id": "e2", "name": "acceptEdits", "kind": "object", "aliases": ["acceptEdits mode"]},
        {"id": "e3", "name": "plan mode", "kind": "object"},
    ],
    "question_entities": ["e1"],
    "answers_entities": [["e2"], ["e3"]],
}

GOOD_ICONS = {
    "entities": {"Claude Code": "terminal", "acceptEdits": "write", "plan mode": "plan"},
    "question": "mode",
    "answers": ["write", "plan"],
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

    async def chat_model(self, model_id, prompt, system=None, max_tokens=None, json_object=False):
        self.calls.append({"model_id": model_id, "prompt": prompt, "system": system})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeCollection:
    """Just enough of AsyncCollection for the decomposition record upserts."""

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


def test_parse_grounding_drops_entity_not_in_that_answer():
    # "MCP Server" is named by the model for answer B, but B never mentions it.
    data = {
        "entities": [
            {"id": "e1", "name": "MCP Server", "kind": "object", "aliases": ["MCP"]},
            {"id": "e2", "name": "plan mode", "kind": "object"},
        ],
        "question_entities": ["e1"],
        "answers_entities": [["e1"], ["e1", "e2"]],
    }
    texts = ("Tools over MCP", ["MCP tools", "plan mode blocks edits"])
    result = parse_entities(data, 2, texts)
    assert names(result.question) == ["MCP Server"]
    assert names(result.answers[0].entities) == ["MCP Server"]
    assert names(result.answers[1].entities) == ["plan mode"]


def test_parse_grounding_prunes_aliases_not_in_text():
    data = {
        "entities": [
            {"id": "e1", "name": "process_refund", "kind": "object",
             "aliases": ["refund requests", "process refund"]},
        ],
        "question_entities": ["e1"],
        "answers_entities": [[], []],
    }
    texts = ("call process refund first", ["", ""])
    result = parse_entities(data, 2, texts)
    assert result.question[0].aliases == ["process refund"]


def test_parse_prunes_generic_one_word_aliases_but_keeps_phrases():
    data = {
        "entities": [
            {"id": "e1", "name": "Refund Decision", "kind": "object",
             "aliases": ["refund", "proposed refund decision"]},
        ],
        "question_entities": ["e1"],
        "answers_entities": [[], []],
    }
    texts = ("the proposed refund decision", ["", ""])
    assert parse_entities(data, 2, texts).question[0].aliases == ["proposed refund decision"]


def test_parse_keeps_actor_grounded_only_by_agent_alias():
    data = {
        "entities": [{"id": "e1", "name": "Main Agent", "kind": "actor", "aliases": ["agent"]}],
        "question_entities": [],
        "answers_entities": [["e1"], []],
    }
    texts = ("", ["The agent gives a score.", ""])
    assert names(parse_entities(data, 2, texts).answers[0].entities) == ["Main Agent"]


def test_parse_drops_generic_entities():
    data = {
        "entities": [
            {"id": "e1", "name": "Tools", "kind": "object", "aliases": ["tool set"]},
            {"id": "e2", "name": "Order", "kind": "object", "aliases": []},
        ],
        "question_entities": ["e1", "e2"],
        "answers_entities": [["e1"], []],
    }
    texts = ("the tool set and the Order", ["tool set", ""])
    result = parse_entities(data, 2, texts)
    assert names(result.question) == ["Order"]
    assert result.answers[0].entities == []


def test_parse_adds_subagent_missing_from_a_place_that_names_it():
    data = {
        "entities": [
            {"id": "e1", "name": "Subagent", "kind": "actor", "aliases": ["test-gap-analyzer"]},
            {"id": "e2", "name": "Hook", "kind": "object"},
        ],
        "question_entities": ["e1"],
        "answers_entities": [["e2"], []],
    }
    texts = (
        "A test-gap-analyzer subagent reviews PRs.",
        ["Add a hook for the subagent", "Nothing here"],
    )
    result = parse_entities(data, 2, texts)
    assert names(result.question) == ["Subagent"]
    assert names(result.answers[0].entities) == ["Subagent", "Hook"]
    assert result.answers[0].entities[0].aliases == ["test-gap-analyzer"]  # catalog entity reused
    assert result.answers[1].entities == []


def test_parse_creates_main_agent_when_catalog_lacks_it():
    data = {
        "entities": [{"id": "e1", "name": "Hook", "kind": "object"}],
        "question_entities": [],
        "answers_entities": [[], ["e1"]],
    }
    texts = ("The main agent collects logs.", ["", "a hook"])
    result = parse_entities(data, 2, texts)
    assert [(e.name, e.kind) for e in result.question] == [("Main Agent", "actor")]


def test_parse_without_texts_skips_grounding():
    assert names(parse_entities(GOOD_ENTITIES, 2).answers[0].entities) == ["acceptEdits"]


# --- derived from the decomposition -------------------------------------------------


def _decomposition(**entities_by_id):
    from test_jobs import decomposition_reply

    d = decomposition_reply(2)
    d["entities"] = [
        {"id": i, "name": name, "type": kind, "aliases": [], "description": "", "source_status": "inferred", "evidence": ""}
        for i, (name, kind) in entities_by_id.items()
    ]
    d["answer_options"][0]["entity_ids"] = ["e2"]
    d["answer_options"][1]["entity_ids"] = ["e3"]
    return QuestionDecomposition.model_validate(d)


def test_derived_entities_keep_their_ids_and_follow_the_answer_options():
    d = _decomposition(
        e1=("mode", "concept"), e2=("acceptEdits mode", "object"), e3=("plan mode", "object")
    )
    result = derive_entities(d, VIEW)
    # Each answer lists what its option relies on; the concept is not in the scene.
    assert [(e.name, e.ref) for e in result.answers[0].entities] == [("acceptEdits mode", "e2")]
    assert [(e.name, e.ref) for e in result.answers[1].entities] == [("plan mode", "e3")]
    # Grounded in the original question's words: "Which mode?" names neither.
    assert result.question == []


def test_derived_entities_need_no_game_text_and_read_no_key():
    d = _decomposition(e2=("acceptEdits mode", "object"), e3=("plan mode", "object"))
    assert len(derive_entities(d, VIEW).answers) == len(DOC["prompt"]["answers"])
    assert not hasattr(VIEW, "correct_response") and not hasattr(VIEW, "feedbacks")


def test_answer_keeps_its_options_entities_without_the_text_naming_them():
    d = _decomposition(
        e1=("Shared Runner", "system"),
        e2=("--continue Flag", "concept"),
        e3=("Session ID", "object"),
    )
    d.answer_options[0].entity_ids = ["e2", "e1"]
    d.answer_options[1].entity_ids = ["e3"]
    result = derive_entities(d, VIEW)
    # The option's entities stay though the answer's words never name them, concept
    # included; the scene's other entities still need the text.
    assert [e.ref for e in result.answers[0].entities] == ["e2", "e1"]
    assert [e.ref for e in result.answers[1].entities] == ["e3"]
    # The question names drawable entities only.
    assert all(e.ref != "e2" for e in result.question)


def test_rederive_script_keeps_icons_and_adds_the_options_entities():
    import importlib.util
    from pathlib import Path

    from claude_exam.api.scenario_catalog import text_hash

    path = Path(__file__).resolve().parents[1] / "scripts" / "rederive_entities.py"
    spec = importlib.util.spec_from_file_location("rederive_entities", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    d = _decomposition(e1=("Shared Runner", "system"), e2=("--continue Flag", "concept"))
    d.answer_options[0].entity_ids = ["e2"]
    d.answer_options[1].entity_ids = ["e1"]
    record = {
        "text_hash": text_hash(DOC),
        "semantic": d.model_dump(),
        # The old derivation: nothing for the answers; icons by id from the icons step.
        "entities": {"question_icon": "workflow", "question": [], "answers": [{"entities": [], "icon": "a"}, {"entities": [], "icon": "b"}]},
        "icons": {"entities": {"e1": "server", "e2": "cli-flag"}},
    }
    new = script.rederived(record, DOC)
    assert [(e["ref"], e["icon"]) for e in new["answers"][0]["entities"]] == [("e2", "cli-flag")]
    assert [(e["ref"], e["icon"]) for e in new["answers"][1]["entities"]] == [("e1", "server")]
    assert (new["question_icon"], [a["icon"] for a in new["answers"]]) == ("workflow", ["a", "b"])
    # A decomposition written for another text is left alone.
    assert script.rederived({**record, "text_hash": "other"}, DOC) is None
