"""Input keys (`step_key.py`, `pipeline_keys.py`): a step's stored result is current
while the messages it would send today are the ones it was written from. A changed
prompt, frame or upstream result makes exactly the steps that read it outdated, with
nothing bumped by hand; retries do not change a key."""

import pytest
from test_jobs import (
    TAGGABLE,
    FakeMongo,
    ScriptedIca,
    client_for,
    decomposition_reply,
    in_chapter,
    reply,
    run_job,
)

from claude_exam.api import batch, decompose, objectives, rephrase, story
from claude_exam.api.batch import STAGES, Stores

pytestmark = pytest.mark.anyio
ALL = list(STAGES["full"])


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
    )


async def outdated(mongo, doc) -> list[str]:
    stores = stores_of(mongo)
    return [s for s in ALL if not await batch.is_current(s, stores, doc)]


async def done():
    q = in_chapter(section=TAGGABLE)
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert job.failed == 0
    return q, mongo, ica


async def test_a_finished_question_is_current_in_every_step():
    q, mongo, _ = await done()
    assert await outdated(mongo, q) == []


async def test_a_changed_prompt_or_frame_outdates_only_the_steps_that_read_it(monkeypatch):
    q, mongo, _ = await done()
    # The company's intro is in the story's and the rephrase's messages (and the check's
    # background, which this module does not patch).
    monkeypatch.setattr(story, "COMPANY_INTRO", story.COMPANY_INTRO + " Now in Lisbon.")
    assert await outdated(mongo, q) == ["story"]
    monkeypatch.setattr(rephrase, "COMPANY_INTRO", rephrase.COMPANY_INTRO + " Now in Lisbon.")
    assert await outdated(mongo, q) == ["story", "rephrase"]


async def test_a_changed_upstream_result_outdates_what_was_built_from_it():
    q, mongo, _ = await done()
    db = mongo.get_default_database()
    question_id = {"question_id": str(q["_id"])}

    # Other entities: the icons were picked for the old ones.
    record = await db["question_decomposition"].find_one(question_id)
    entities = record["entities"]
    entities["question"] = [{"name": "Another thing", "kind": "object", "aliases": []}]
    await db["question_decomposition"].update_one(question_id, {"$set": {"entities": entities}})
    assert await outdated(mongo, q) == ["icons"]

    # Another rephrase: its check, not the rephrase itself (its own inputs did not change).
    await db["examination_simplified"].update_one(
        question_id, {"$set": {"prompt.question": "Which mode never edits?"}}
    )
    assert await outdated(mongo, q) == ["icons", "check"]


async def test_another_decomposition_outdates_the_flow_and_its_resolution():
    q, mongo, _ = await done()
    record = mongo.get_default_database()["question_decomposition"].docs[0]
    record["semantic"]["context"]["description"] = "Another reading."
    # The flow's messages show the decomposition; the icons read the stored entities and
    # the decomposition was asked from the same messages, so both stay current.
    assert await outdated(mongo, q) == ["flow", "resolve"]


async def test_a_repaired_reply_keeps_the_key_of_the_first_messages():
    q = in_chapter(section=TAGGABLE)
    bad = decomposition_reply()
    bad["entities"] = []  # fails the gate: the answer options reference missing ids
    mongo, ica = FakeMongo([q]), ScriptedIca(decompose=[reply(bad)])
    await run_job(mongo, ica, [q["_id"]], "decompose")
    assert ica.calls.count("decompose") == 2  # rejected once, then repaired
    record = mongo.get_default_database()["question_decomposition"].docs[0]
    assert record["input_key"] == decompose.input_key(q)


async def test_the_objectives_and_story_backfills_pick_what_the_keys_call_outdated(monkeypatch):
    q, mongo, ica = await done()
    client = client_for(mongo, ica)
    assert client.post("/examination/objectives/backfill").json()["total"] == 0
    assert client.post("/examination/story/backfill").json()["total"] == 0
    # A changed prompt (nothing bumped) is found by the routes, not only by the job.
    monkeypatch.setattr(objectives, "_SYSTEM_MESSAGE", objectives._SYSTEM_MESSAGE + "\nNew rule.")
    monkeypatch.setattr(story, "_SYSTEM_MESSAGE", story._SYSTEM_MESSAGE + "\nNew rule.")
    assert client.post("/examination/objectives/backfill").json()["total"] == 1
    assert client.post("/examination/story/backfill").json()["total"] == 1
