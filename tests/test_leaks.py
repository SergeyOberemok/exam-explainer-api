"""One leak rule for every text shown before answering (`leaks.py`): the gates of the
decomposition, the flow, the beat and the rephrase, the semantic parts, and the flow
built in code (an experiment)."""

import pytest
from fastapi import HTTPException
from test_jobs import (
    FakeMongo,
    ScriptedIca,
    SIMPLE,
    decomposition_reply,
    decomposition_with_plan_mode,
    flow_reply,
    question,
    reply,
    run_job,
    story_reply,
)

from claude_exam.api import batch
from claude_exam.api.beat import beat_and_store, beat_errors, beat_item
from claude_exam.api.decompose import decomposition_errors
from claude_exam.api.flow import flow_errors, flow_from
from claude_exam.api.leaks import AnswerOnly, answer_only, leak, leak_errors
from claude_exam.api.pipeline_keys import Records, is_current
from claude_exam.api.question_view import key_free, the_ask
from claude_exam.api.rephrase import rephrase_item
from claude_exam.api.scenario_catalog import text_hash
from claude_exam.api.schemas import QuestionDecomposition

pytestmark = pytest.mark.anyio
MODEL = "claude-haiku-5-5"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)


LONG = "Route uncovered findings to a human reviewer"


def long_question():
    """A question with a long answer (tested by its text) and the stem's own words."""
    q = question()
    q["prompt"] = {
        **q["prompt"],
        "question": "The review agent posts findings the guidelines do not cover. What now?",
        "answers": ["Lower temperature", LONG],
    }
    return q


def plan_mode():
    return QuestionDecomposition.model_validate(decomposition_with_plan_mode())


# --- the rule ---------------------------------------------------------------------


def test_a_text_leaks_when_it_names_an_answer_only_entity_or_repeats_an_answer():
    q = question()
    terms = answer_only(q, plan_mode())
    # "plan mode" is named only by answer B; "Agent" no text names (the model's word).
    assert {e.name for e in terms.entities} == {"Agent", "Ambiguity", "plan mode"}
    assert [e.name for e in terms.proposed] == ["plan mode"]
    assert leak("The team switches to Plan Mode.", terms) == "names 'plan mode', which only an answer option mentions"

    long = long_question()
    terms = answer_only(long)
    assert leak(f"Omar says: {LONG.lower()}.", terms) == "repeats answer B"
    # Short answers are words, not leaks; without a decomposition no entity is known.
    assert terms.answers[0] == "" and leak("Lower temperature", terms) == ""
    assert leak("The guidelines are silent.", terms) == ""


def test_an_answer_the_scenario_already_says_is_no_leak():
    q = long_question()
    q["prompt"]["question"] = f"Kim proposes: {LONG}. Is that enough?"
    assert answer_only(q).answers[1] == ""


def test_leak_errors_name_the_text_and_the_reason():
    terms = AnswerOnly(answers=("", LONG.lower()))
    assert leak_errors([("node n2", LONG), ("node n3", "fine")], terms, "show the scenario") == [
        "node n2 repeats answer B: show the scenario"
    ]


# --- the gates --------------------------------------------------------------------


def test_the_decomposition_gate_rejects_a_scenario_fact_that_repeats_an_answer():
    q = long_question()
    d = decomposition_reply()
    assert decomposition_errors(q, d) == []
    d["context"]["current_state"][0]["text"] = f"They {LONG.lower()} already."
    assert any("repeats answer B" in e for e in decomposition_errors(q, d))


def test_the_flow_gate_rejects_a_label_that_repeats_an_answer():
    q = long_question()
    d = QuestionDecomposition.model_validate(decomposition_reply())
    f = flow_reply()
    assert flow_errors(q, d, f) == []
    f["nodes"][1]["label"] = LONG
    assert any(e.startswith("node n2 repeats answer B") for e in flow_errors(q, d, f))


def test_the_beat_gate():
    q = key_free(question())
    assert beat_errors(q, {"beat": "Edits land before anyone reads them."}, plan_mode()) == []
    assert beat_errors(q, {"beat": ""}) == ['"beat" must be one sentence']
    assert "longer than" in beat_errors(q, {"beat": " ".join(["word"] * 30)})[0]
    assert "names 'plan mode'" in beat_errors(q, {"beat": "Kim wants plan mode."}, plan_mode())[0]
    # Without a decomposition only the answers' texts are known.
    assert beat_errors(q, {"beat": "Kim wants plan mode."}) == []
    # A beat keeps the question's scope: it says "all" where the question does not.
    assert "the beat says all" in beat_errors(q, {"beat": "All edits land before anyone reads them."})[0]


async def test_the_models_own_word_is_no_leak_in_texts_written_from_the_question():
    # "Agent" is an entity no text names (the model's word, not an answer's): the beat
    # and the rephrase, written from the raw question, may use it.
    d = QuestionDecomposition.model_validate(decomposition_reply())
    q = question()
    assert beat_errors(key_free(q), {"beat": "The agent edits too much."}, d) == []
    ica = ScriptedIca(rephrase=[reply({**SIMPLE, "question": "Which mode should the agent use?"})])
    rephrase, _meta = await rephrase_item(ica, q, MODEL, decomposition=d)
    assert rephrase.question == "Which mode should the agent use?" and ica.calls == ["rephrase"]


async def test_a_leaking_beat_is_sent_back_and_repaired():
    ica = ScriptedIca(beat=[reply({"beat": "Kim wants plan mode."})])
    place = {"scenario": "support", "stage": "build"}
    text, meta = await beat_item(ica, question(), place, MODEL, decomposition=plan_mode())
    assert text == "The team cannot agree on a working mode." and ica.calls == ["beat", "beat"]
    assert meta.total_tokens == 14


async def test_a_beat_needs_a_stage():
    with pytest.raises(HTTPException) as exc:
        await beat_item(ScriptedIca(), question(), {"scenario": "", "stage": ""}, MODEL)
    assert exc.value.status_code == 422


async def test_a_rephrased_question_naming_what_an_answer_proposes_is_sent_back():
    good = {**SIMPLE, "question": "Which mode only reads and never edits?"}
    ica = ScriptedIca(rephrase=[reply(SIMPLE), reply(good)])
    rephrase, _meta = await rephrase_item(ica, question(), MODEL, decomposition=plan_mode())
    assert rephrase.question == good["question"] and ica.calls == ["rephrase", "rephrase"]
    # Without a decomposition "plan mode" is not known to be an answer's.
    ica = ScriptedIca(rephrase=[reply(SIMPLE)])
    rephrase, _meta = await rephrase_item(ica, question(), MODEL)
    assert rephrase.question == SIMPLE["question"] and ica.calls == ["rephrase"]


# --- the beat step in jobs ----------------------------------------------------------


async def test_the_beat_is_written_on_the_story_record_and_a_seed_beat_is_current():
    q = question()
    q["prompt"]["question"] = (
        "SCENARIO: You are integrating Claude Code into your Continuous Integration pipeline. "
        "QUESTION: Which mode?"
    )
    mongo, ica = FakeMongo([q]), ScriptedIca()
    stories = mongo.get_default_database()["question_story"]
    # No story record: nothing to set the beat in.
    with pytest.raises(HTTPException) as exc:
        await beat_and_store(ica, stories, q, MODEL)
    assert exc.value.status_code == 404 and ica.calls == []

    await run_job(mongo, ica, [q["_id"]], "full")
    record = await stories.find_one({"question_id": str(q["_id"])})
    assert record["beat"] == "The team cannot agree on a working mode."
    assert is_current("beat", q, Records(story=record))

    seed = {**record, "origin": "seed", "beat": "Hand-written.", "beat_input_key": None}
    assert is_current("beat", q, Records(story=seed))
    # A seed record without a beat still gets one written.
    assert not is_current("beat", q, Records(story={**seed, "beat": ""}))


# --- the flow built in code (experiment) --------------------------------------------


def test_the_flow_built_in_code_passes_the_flow_gate_and_ends_at_the_ask():
    q = question()
    d = plan_mode()
    built = flow_from(d, q)
    assert flow_errors(q, d, built.model_dump()) == []
    last = built.nodes[-1]
    assert last.is_question and last.type == "decision" and last.label == the_ask(q)
    assert built.nodes[0].type == "start" and built.visualization.primary == "process_flow"
    assert [(e.source, e.target) for e in built.edges] == [
        (a.id, b.id) for a, b in zip(built.nodes, built.nodes[1:])
    ]


def test_the_flow_built_in_code_follows_the_events_and_leaves_out_what_leaks():
    q = question()
    d = decomposition_with_plan_mode()
    d["events"] = [
        {"id": "v2", "name": "The agent edits files", "actor_id": None, "target_id": None, "input_ids": [], "output_ids": [], "sequence": 2, "source_status": "inferred", "evidence": ""},
        {"id": "v1", "name": "A task arrives", "actor_id": None, "target_id": None, "input_ids": [], "output_ids": [], "sequence": 1, "source_status": "inferred", "evidence": ""},
        {"id": "v3", "name": "Someone turns on plan mode", "actor_id": None, "target_id": None, "input_ids": [], "output_ids": [], "sequence": 3, "source_status": "inferred", "evidence": ""},
    ]
    decomposition = QuestionDecomposition.model_validate(d)
    built = flow_from(decomposition, q)
    labels = [n.label for n in built.nodes]
    # "agent" names the model's own entity, which the flow gate also refuses.
    assert labels == ["A task arrives", "Edits land unreviewed.", "Which mode?"]
    assert [n.type for n in built.nodes] == ["start", "output", "decision"]
    assert flow_errors(q, decomposition, built.model_dump()) == []


# --- what the leak gate read is in the input keys ------------------------------------


def test_beat_and_rephrase_keys_cover_the_answer_only_entities_their_gate_tested():
    from claude_exam.api import beat, rephrase

    q = question()
    q["prompt"]["question"] = (
        "SCENARIO: You are integrating Claude Code into your Continuous Integration pipeline. "
        "QUESTION: Which mode?"
    )
    story = {"question_id": str(q["_id"]), "scenario": "pipeline", "stage": "design", "text_hash": text_hash(q)}
    place = {"scenario": "pipeline", "stage": "design"}
    d = plan_mode()
    # Written before the decomposition existed, the gate knew no entity: a later
    # decomposition that names one outdates both.
    assert beat.input_key(q, story) != beat.input_key(q, story, d)
    assert rephrase.input_key(q, place) != rephrase.input_key(q, place, d)
    # A decomposition whose entities no answer names leaves the keys as they were.
    plain = d.model_copy(deep=True)
    plain.entities = [e for e in plain.entities if e.name != "plan mode"]
    assert answer_only(q, plain).narrow().entities == ()
    assert beat.input_key(q, story) == beat.input_key(q, story, plain)
    assert rephrase.input_key(q, place) == rephrase.input_key(q, place, plain)

    record = {"question_id": str(q["_id"]), "text_hash": text_hash(q), "semantic": d.model_dump()}
    stored = {**story, "origin": "ica", "beat": "A line.", "beat_input_key": beat.input_key(q, story)}
    assert is_current("beat", q, Records(story=stored))
    assert not is_current("beat", q, Records(story=stored, decomposition=record))


def test_the_seed_loader_refuses_a_beat_that_names_what_only_an_answer_proposes():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "load_story_seed.py"
    spec = importlib.util.spec_from_file_location("load_story_seed", path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)

    q = question()
    q["prompt"]["question"] = (
        "SCENARIO: You are integrating Claude Code into your Continuous Integration pipeline. "
        "QUESTION: Which mode?"
    )
    item = {
        "hash": text_hash(q), "scenario": "pipeline", "stage": "design",
        "beat": "The team wants plan mode for every run.",
        "win": "Runs stay safe.", "fail": "Runs still edit files.",
    }
    record = {"question_id": str(q["_id"]), "text_hash": text_hash(q), "semantic": plan_mode().model_dump()}
    updates, problems = loader.seed_updates([item], [q], {str(q["_id"]): record})
    assert updates == [] and "names 'plan mode'" in problems[0]

    # Without the entity the beat loads, and drops a beat the model wrote earlier.
    updates, problems = loader.seed_updates([{**item, "beat": "Runs keep editing files."}], [q])
    assert problems == []
    assert updates[0][1]["beat_ica"] is None and updates[0][1]["beat_input_key"] is None
