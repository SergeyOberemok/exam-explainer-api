"""The game document (`compose.py`): put together by the last step of a "full" job from
the stored records, only when everything is there, with nothing of the answer key in
what is sent before answering; current exactly while composing again gives the same
document. Also the resolve gate on the per-answer consequence and impact."""

import copy
import json

import pytest
from fastapi import HTTPException
from test_jobs import (
    TAGGABLE,
    FakeMongo,
    ScriptedIca,
    decomposition_reply,
    decomposition_with_plan_mode,
    flow_reply,
    in_chapter,
    objectives_reply,
    reply,
    resolution_reply,
    run_job,
)

from claude_exam.api import batch, compose, flow
from claude_exam.api.batch import Stores
from claude_exam.api.schemas import GameCard, GameCardKey, QuestionDecomposition

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)


def stores_of(mongo) -> Stores:
    db = mongo.get_default_database()
    return Stores(
        db["examination"],
        db["examination_simplified"],
        db["question_objectives"],
        db["question_story"],
        db["question_decomposition"],
        db["game_question"],
    )


async def composed(**ica):
    q = in_chapter(section=TAGGABLE)
    mongo = FakeMongo([q])
    ica = ScriptedIca(decompose=[reply(decomposition_with_plan_mode())], **ica)
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert (job.failed, job.errors) == (0, [])
    return q, mongo, mongo.get_default_database()["game_question"].docs


async def test_a_full_job_composes_the_game_document():
    q, _, docs = await composed()
    assert len(docs) == 1
    doc = docs[0]
    card, key = GameCard.model_validate(doc["public"]), GameCardKey.model_validate(doc["key"])
    assert (doc["question_id"], doc["held"], doc["count"]) == (str(q["_id"]), False, 2)
    assert (doc["scenario"], doc["stage"]) == (card.scenario, card.stage) == ("pipeline", "build")

    # 1. The briefing in play order, then the question's own ask, apart.
    roles = [p.role for p in card.briefing]
    assert "ask" not in roles and roles == sorted(roles, key=compose.BRIEFING_ORDER.index)
    assert card.ask.text == "Which mode?"
    # 2. One tool per answer: its text, what it proposes and what it is made of. The
    #    concept the icon step does not draw is in the cast too, with an icon by type.
    assert [t.text for t in card.tools] == ["acceptEdits mode", "plan mode"]
    assert [t.label for t in card.tools] == ["a proposal", "a proposal"]
    assert card.tools[1].components == ["e3"]
    cast = {c.ref: c for c in card.cast}
    assert cast["e2"].type == "concept" and cast["e2"].icon == "idea"
    # Only answer B names plan mode: it is not part of the scene.
    assert cast["e3"].scene is False
    # 3. The map is the pre-answer flow.
    assert [n.id for n in card.map.nodes] == ["n1", "n2", "n3"]
    # 4. The key: an outcome per answer, from the resolution and the objectives.
    assert key.correct == 1
    assert [(o.verdict, o.impact) for o in key.outcomes] == [("incorrect", "failure"), ("correct", "success")]
    assert key.outcomes[1].consequence == "The review goes well."
    assert key.outcomes[0].misconception == "Auto-accepting edits keeps files unchanged."
    assert key.outcomes[1].misconception == ""
    assert (key.win, key.fail) == ("Reviews happen before edits.", "Edits land unreviewed.")
    assert any(n.added for n in key.resolution_flow.nodes)


async def test_nothing_of_the_key_is_in_what_is_sent_before_answering():
    q = in_chapter(section=TAGGABLE)
    q["prompt"]["explanation"] = "SENTINEL-EXPLANATION"
    q["prompt"]["feedbacks"] = ["SENTINEL-FEEDBACK-A", "SENTINEL-FEEDBACK-B"]
    tagged = objectives_reply()
    tagged["distractors"][0]["misconception"] = "SENTINEL-MISCONCEPTION"
    resolved = resolution_reply()
    for option in resolved["options"]:
        option["consequence"] = f"SENTINEL-CONSEQUENCE {option['index']}"
        option["reason"] = f"SENTINEL-REASON {option['index']}"
    resolved["rule"] = "SENTINEL-RULE"
    mongo = FakeMongo([q])
    ica = ScriptedIca(objectives=[reply(tagged)], resolve=[reply(resolved)])
    await run_job(mongo, ica, [q["_id"]], "full")
    doc = mongo.get_default_database()["game_question"].docs[0]
    public = json.dumps(doc["public"])
    assert "SENTINEL" not in public
    assert not {"correct", "verdict", "impact", "outcomes"} & set(doc["public"])
    # ... and all of it is in the key.
    key = json.dumps(doc["key"])
    for sentinel in ("EXPLANATION", "FEEDBACK-B", "MISCONCEPTION", "CONSEQUENCE 1", "REASON 0", "RULE"):
        assert f"SENTINEL-{sentinel}" in key


async def test_a_question_is_composed_only_when_everything_is_there():
    q = in_chapter(section=TAGGABLE)
    assert compose.missing(q, None, None, None) == ["decomposition", "story", "objectives"]

    # The resolution fails: no game document, and the job says what is missing.
    mongo = FakeMongo([q])
    ica = ScriptedIca(resolve=[HTTPException(status_code=400, detail="x")])
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert job.stages_failed == {"resolve": 1, "compose": 1}
    assert "resolution" in job.errors[-1].detail and job.errors[-1].status_code == 422
    assert mongo.get_default_database()["game_question"].docs == []


async def test_a_resolution_without_consequences_is_not_enough():
    # One stored before the resolve step wrote consequences (the defaults read it).
    q, mongo, _ = await composed()
    db = mongo.get_default_database()
    record = db["question_decomposition"].docs[0]
    for option in record["resolution"]["options"]:
        option.pop("consequence")
        option.pop("impact")
    assert compose.missing(q, *[db[c].docs[0] for c in ("question_objectives", "question_story")], record) == [
        "resolution with a consequence per answer"
    ]


async def test_an_edited_question_loses_its_game_document():
    q, mongo, docs = await composed()
    db = mongo.get_default_database()
    db["examination"].docs[0]["prompt"]["question"] += " Pick one."
    job = await run_job(mongo, ScriptedIca(), [q["_id"]], "compose")
    assert job.stages_failed == {"compose": 1} and "decomposition" in job.errors[0].detail
    assert db["game_question"].docs == []


async def test_the_document_is_current_while_composing_again_gives_the_same():
    q, mongo, docs = await composed()
    stores = stores_of(mongo)
    assert await batch.is_current("compose", stores, q)
    db = mongo.get_default_database()

    # What the game does not show (the decomposition's own context text) changes nothing.
    db["question_decomposition"].docs[0]["semantic"]["context"]["description"] = "Another reading."
    assert await batch.is_current("compose", stores, q)

    # A new beat shows: the document is outdated, and a compose job writes it again.
    db["question_story"].docs[0]["beat"] = "A new line before the question."
    assert not await batch.is_current("compose", stores, q)
    job = await run_job(mongo, ScriptedIca(), [q["_id"]], "compose", only_missing=True)
    assert job.succeeded == 1
    assert db["game_question"].docs[0]["public"]["beat"] == "A new line before the question."
    assert await batch.is_current("compose", stores, q)
    again = await run_job(mongo, ScriptedIca(), [q["_id"]], "compose", only_missing=True)
    assert again.skipped == 1


# --- the resolve gate on consequences ----------------------------------------------------


def _resolution_errors(**option_changes):
    q = in_chapter(section=TAGGABLE)
    parsed = resolution_reply()
    for index, changes in option_changes.items():
        parsed["options"][int(index[1:])].update(changes)
    return flow.resolution_errors(q, QuestionDecomposition.model_validate(decomposition_reply()), parsed)


def test_the_resolution_gives_each_option_a_consequence_and_an_impact():
    assert _resolution_errors() == []
    assert _resolution_errors(o0={"impact": "partial"}) == []
    assert any("impact" in e for e in _resolution_errors(o0={"impact": "success"}))
    assert any("impact" in e for e in _resolution_errors(o1={"impact": "partial"}))
    assert any("consequence" in e for e in _resolution_errors(o0={"consequence": " "}))
    long_line = " ".join(["word"] * (flow.MAX_CONSEQUENCE_WORDS + 1))
    assert any("at most" in e for e in _resolution_errors(o1={"consequence": long_line}))


def test_the_resolution_is_set_in_the_question_s_place_in_the_story():
    q = in_chapter(section=TAGGABLE)
    d = QuestionDecomposition.model_validate(decomposition_reply())
    drawn = flow.DecisionFlow.model_validate(copy.deepcopy(flow_reply()))
    _, user = flow.resolve_messages(q, d, drawn, {"scenario": "pipeline", "stage": "build"})
    assert "Chapter" in user and "Stage" in user
    # Another place is another input: the stored resolution is outdated.
    assert flow.resolve_input_key(q, d, drawn, {"scenario": "pipeline", "stage": "build"}) != (
        flow.resolve_input_key(q, d, drawn, {"scenario": "pipeline", "stage": "scale"})
    )


async def test_the_cast_and_the_actions_carry_the_icons_the_model_picked():
    d = decomposition_with_plan_mode()
    d["entities"].append(
        {"id": "e5", "name": "Claude Code", "type": "actor", "aliases": [], "description": "Runs in CI",
         "source_status": "inferred", "evidence": ""}
    )

    def event(id, name, sequence, **refs):
        return {"id": id, "name": name, "actor_id": refs.get("actor"), "target_id": refs.get("target"),
                "input_ids": refs.get("inputs", []), "output_ids": [], "sequence": sequence,
                "source_status": "inferred", "evidence": ""}

    d["events"] = [
        event("v2", "applies the edits", 2, actor="e5", target="e3", inputs=["e5", "e3"]),
        event("v1", "starts the job", 1, actor="e5"),
        event("v9", "waits for a review", None, actor="e5"),
    ]
    picks = {"entities": {"e2": "rule", "e3": "plan", "e5": "terminal"}, "actions": {"v1": "run", "v2": "write"},
             "question": "mode", "answers": ["write", "plan"]}
    q = in_chapter(section=TAGGABLE)
    mongo = FakeMongo([q])
    job = await run_job(mongo, ScriptedIca(decompose=[reply(d)], icons=[reply(picks)]), [q["_id"]], "full")
    assert (job.failed, job.errors) == (0, [])
    card = GameCard.model_validate(mongo.get_default_database()["game_question"].docs[0]["public"])

    cast = {c.ref: c.icon for c in card.cast}
    # The concept has the model's icon now, not the lightbulb; e1 got no pick: its type's.
    assert (cast["e2"], cast["e3"], cast["e5"], cast["e1"]) == ("rule", "plan", "terminal", "actor")
    # The actions that happened, in order, with their verbs' icons; plan mode is only
    # answer B's, so the action does not point at it.
    assert [(a.ref, a.sequence, a.icon) for a in card.actions] == [("v1", 1, "run"), ("v2", 2, "write")]
    assert (card.actions[1].actor, card.actions[1].target, card.actions[1].inputs) == ("e5", "", ["e5"])
    assert [t.icon for t in card.tools] == ["write", "plan"] and card.icon == "mode"


async def test_icons_stored_only_on_the_visual_entities_are_not_enough():
    # Picked before the icon step stored its picks by id: the backfill picks them again.
    q, mongo, _ = await composed()
    db = mongo.get_default_database()
    record = db["question_decomposition"].docs[0]
    record.pop("icons")
    assert compose.missing(q, *[db[c].docs[0] for c in ("question_objectives", "question_story")], record) == ["icons"]
