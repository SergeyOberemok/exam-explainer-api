"""The whole ingest: a repeatable import, the "full" job with resume, the chapter the
model picks for a question without a scenario paragraph."""

import pytest
from fastapi import HTTPException
from test_jobs import (
    TAGGABLE,
    FakeMongo,
    ScriptedIca,
    client_for,
    decomposed,
    in_chapter,
    outsider,
    question,
    reply,
    run_job,
    story_record,
    story_reply,
    tags,
    upload,
)

from claude_exam.api import batch
from claude_exam.api.game.domain import story_of
from claude_exam.api.ingest import plan_import
from claude_exam.api.normalization import normalize_result
from claude_exam.api.objectives import difficulty_of
from claude_exam.api.scenario_catalog import BY_KEY
from claude_exam.api.story import ICA_ORDER, _build_system_message as story_system

pytestmark = pytest.mark.anyio
# The ICA calls of a "full" job (the parts and entities are built from the
# decomposition in code and the game document is composed in code, so they make no call).
FULL = ["objectives", "story", "decompose", "beat", "icons", "flow", "resolve"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)


def ready(**kw):
    """A question every step can handle: a known chapter and a section with objectives."""
    return in_chapter(section=TAGGABLE, **kw)


# --- the full job --------------------------------------------------------------------


async def test_full_job_runs_every_step_in_order_and_stores_the_keys():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert ica.calls == FULL
    assert (job.mode, job.status, job.succeeded, job.failed) == ("full", "completed", 1, 0)

    # The last step composed the game document.
    game = mongo.get_default_database()["game_question"].docs
    assert [d["question_id"] for d in game] == [str(q["_id"])] and game[0]["input_key"]
    record = await decomposed(mongo, q)
    assert record["icons_ica"] is not None and record["parts"] and record["flow"]
    # The story is ordered after the seeded records, by the difficulty just tagged.
    assert (await story_record(mongo, q))["order"] == ICA_ORDER + (await tags(mongo, q))["difficulty"]


async def test_full_job_run_again_does_only_what_is_missing():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca(icons=[HTTPException(status_code=400, detail="x")])
    first = await run_job(mongo, ica, [q["_id"]], "full")
    # The icon step failed; the flow and resolution do not need it and ran, and the
    # game document could not be composed without the icons.
    assert ica.calls == FULL and (first.failed, first.icons_failed) == (1, 1)
    assert first.stages_failed == {"icons": 1, "compose": 1}
    assert [e.stage for e in first.errors] == ["icons", "compose"]
    assert "icons" in first.errors[1].detail
    assert mongo.get_default_database()["game_question"].docs == []

    ica.calls.clear()
    again = await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert ica.calls == ["icons"] and (again.succeeded, again.skipped) == (1, 0)

    ica.calls.clear()
    done = await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert ica.calls == [] and (done.status, done.skipped) == ("completed", 1)


async def test_a_failed_step_does_not_stop_the_steps_that_do_not_need_it():
    # No objectives for the section and no chapter for the paragraph: both are refused
    # without a call, and everything else is still done (the resolution in the company
    # only); the game document cannot be composed without them.
    q = outsider()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert ica.calls == ["decompose", "icons", "flow", "resolve"]
    assert (job.status, job.failed) == ("completed", 1)
    assert job.stages_failed == {"objectives": 1, "story": 1, "compose": 1}
    assert [e.status_code for e in job.errors] == [422, 422, 422]

    # Without a decomposition there are no icons, flow or resolution; the beat does not
    # need one.
    q2 = ready()
    mongo, ica = FakeMongo([q2]), ScriptedIca(decompose=[HTTPException(status_code=400, detail="x")])
    job = await run_job(mongo, ica, [q2["_id"]], "full")
    assert ica.calls == ["objectives", "story", "decompose", "beat"]
    assert job.stages_failed == {"decompose": 1, "compose": 1} and job.status == "completed"


async def test_questions_are_processed_side_by_side_and_all_counted():
    questions = [ready() for _ in range(7)]
    mongo, ica = FakeMongo(questions), ScriptedIca()
    jobs = mongo.get_default_database()["simplification_jobs"]
    job = await jobs.insert_one(batch.new_job("t", "claude-haiku-5-5", 7, "full"))
    import asyncio

    await batch.run_simplification_job(
        mongo, ica, asyncio.Semaphore(1), job.inserted_id, [q["_id"] for q in questions],
        "claude-haiku-5-5", "full", False, 3,
    )
    doc = await jobs.find_one({"_id": job.inserted_id})
    assert (doc["status"], doc["processed"], doc["succeeded"]) == ("completed", 7, 7)
    assert sorted(ica.calls) == sorted(FULL * 7)


# --- import --------------------------------------------------------------------------


def test_import_skips_what_the_bank_holds_and_reports_what_needs_a_catalog():
    mongo, ica = FakeMongo(), ScriptedIca()
    client = client_for(mongo, ica)
    known = ready()

    dry = upload(client, known, dry_run="true").json()
    assert (dry["dry_run"], dry["inserted"], dry["new"], dry["job_id"]) == (True, 0, 1, None)
    assert dry["chapters"] == [{"name": "pipeline", "count": 1}]
    assert mongo.get_default_database()["examination"].docs == [] and ica.calls == []

    first = upload(client, known).json()
    assert (first["inserted"], first["duplicates"]) == (1, 0) and first["job_id"]
    ica.calls.clear()
    second = upload(client, known).json()
    assert (second["inserted"], second["duplicates"], second["job_id"]) == (0, 1, None)
    assert ica.calls == [] and len(mongo.get_default_database()["examination"].docs) == 1

    report = upload(client, outsider(section="Brand new domain"), dry_run="true").json()
    assert report["unknown_scenarios"] == [{"name": "You run a bakery.", "count": 1}]
    assert report["unknown_sections"] == [{"name": "Brand new domain", "count": 1}]


def test_plan_counts_duplicates_inside_one_file_and_unplayable_questions():
    def doc(text, **kw):
        item = {**question(**kw), "prompt": {**question()["prompt"], "question": text}}
        return normalize_result({k: v for k, v in item.items() if k != "_id"}, "x.json")

    one, same = doc("First?"), doc("  first? ")
    multi = doc("Pick two?", correct_response=["a", "b"])
    new, report = plan_import([one, same, multi], set())
    assert new == [one, multi]
    assert (report["new"], report["duplicates"], report["unplayable"]) == (2, 1, 1)
    assert report["without_scenario"] == 2
    assert plan_import([one], {one["text_hash"]})[1]["duplicates"] == 1


def test_pipeline_backfill_starts_only_for_questions_that_need_something():
    done, todo = ready(), ready()
    todo["prompt"]["question"] += " Really?"
    mongo, ica = FakeMongo([done, todo]), ScriptedIca()
    client = client_for(mongo, ica)

    first = client.post("/examination/pipeline/backfill", params={"limit": 1})
    assert first.status_code == 202 and first.json()["total"] == 1
    job = client.get(f"/examination/jobs/{first.json()['job_id']}").json()
    assert (job["mode"], job["only_missing"], job["succeeded"]) == ("full", True, 1)

    rest = client.post("/examination/pipeline/backfill")
    assert rest.json()["total"] == 1
    assert client.post("/examination/pipeline/backfill").json() == {"job_id": None, "total": 0}
    assert client.post("/examination/pipeline/backfill", params={"only_missing": "false"}).json()["total"] == 2
    assert client.post("/examination/pipeline/backfill", params={"source": "other.json"}).json()["total"] == 0


# --- a chapter for a question without a scenario paragraph ----------------------------


async def test_the_model_picks_the_chapter_of_a_question_without_a_paragraph():
    q = question(section=TAGGABLE)
    system = story_system(q)
    assert "CHAPTERS" in system and '"scenario": "dev-tools" | "support"' in system
    assert all(f"- {key}: " in system for key in BY_KEY)
    assert "CHAPTERS" not in story_system(in_chapter()) and '"scenario"' not in story_system(in_chapter())

    # A reply without a chapter is rejected and asked again.
    picked = reply({**story_reply(), "scenario": "research"})
    mongo, ica = FakeMongo([q]), ScriptedIca(story=[reply(story_reply()), picked])
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert job.succeeded == 1 and ica.calls.count("story") == 2

    record = await story_record(mongo, q)
    assert (record["scenario"], record["scenario_origin"]) == ("research", "model")
    # The game then serves that chapter.
    assert story_of(q, record)["scenario"] == "research" and story_of(q, record)["stem"] == ""
    # A record for another text of the question says nothing about this one.
    assert story_of(q, {**record, "text_hash": "other"})["scenario"] == ""
    # And a paragraph no chapter claims is never overruled by a record.
    assert story_of(outsider(), {**record, "text_hash": "x"})["scenario"] == ""


# --- difficulty -----------------------------------------------------------------------


def test_difficulty_follows_the_tempting_answers_and_the_level():
    assert difficulty_of("remember", 0) == 1 and difficulty_of("apply", 1) == 1
    assert difficulty_of("apply", 2) == 2 and difficulty_of("analyze", 1) == 2
    assert difficulty_of("analyze", 2) == 3 and difficulty_of("apply", 3) == 3


def test_a_rewrite_may_not_widen_a_fact():
    from claude_exam.api.exam_terms import widened

    assert widened("The agent posts these as definitive.", "The agent posts every finding as definitive.") == ["every"]
    assert widened("It reads all files.", "It reads all the files.") == []
    # "never" is how a rewrite says "does not": not a wider scope.
    assert widened("It does not edit.", "It never edits.") == []
