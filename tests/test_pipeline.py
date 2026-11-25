"""The whole ingest: a repeatable import, the "full" job with resume, the chapter the
model picks for a question without a scenario paragraph, the gates on the game text and
the check step."""

import pytest
from fastapi import HTTPException
from test_jobs import (
    SIMPLE,
    TAGGABLE,
    FakeMongo,
    ScriptedIca,
    client_for,
    in_chapter,
    outsider,
    question,
    reply,
    run_job,
    stored,
    story_record,
    story_reply,
    tags,
    upload,
)

from claude_exam.api import batch
from claude_exam.api.check import PROMPT_VERSION as CHECK_VERSION
from claude_exam.api.check import parse_pick
from claude_exam.api.exam_terms import identifiers, lost_numbers, lost_terms
from claude_exam.api.game.domain import simplified_matches, story_of
from claude_exam.api.ingest import plan_import
from claude_exam.api.normalization import normalize_result
from claude_exam.api.objectives import difficulty_of
from claude_exam.api.scenario_catalog import BY_KEY, chapter_version, frame_version
from claude_exam.api.simplify import (
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    _build_system_message,
    _build_user_message,
    fidelity_errors,
    opening,
    simplify_item,
)
from claude_exam.api.story import ICA_ORDER, _build_system_message as story_system

pytestmark = pytest.mark.anyio
# The ICA calls of a "full" job: the entities are derived from the decomposition, so
# that step makes no call.
FULL = ["objectives", "story", "decompose", "simplify", "icons", "flow", "resolve", "check"]


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


async def test_full_job_runs_every_step_in_order_and_stores_the_versions():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert ica.calls == FULL
    assert (job.mode, job.status, job.succeeded, job.failed) == ("full", "completed", 1, 0)

    record = await stored(mongo, q)
    assert record["prompt_version"] == PROMPT_VERSION
    assert record["frame_version"] == frame_version("pipeline")
    assert record["check"]["verdict"] == "same" and record["check"]["picked"] == 1
    assert record["check"]["prompt_version"] == CHECK_VERSION
    assert record["icons_ica"] is not None
    # The story is ordered after the seeded records, by the difficulty just tagged.
    assert (await story_record(mongo, q))["order"] == ICA_ORDER + (await tags(mongo, q))["difficulty"]


async def test_full_job_run_again_does_only_what_is_missing():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca(icons=[HTTPException(status_code=400, detail="x")])
    first = await run_job(mongo, ica, [q["_id"]], "full")
    # The icon step failed; the check does not need it and ran.
    assert ica.calls == FULL and (first.failed, first.icons_failed) == (1, 1)
    assert first.stages_failed == {"icons": 1} and first.errors[0].stage == "icons"
    assert (await stored(mongo, q))["check"]["verdict"] == "same"

    ica.calls.clear()
    again = await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert ica.calls == ["icons"] and (again.succeeded, again.skipped) == (1, 0)

    ica.calls.clear()
    done = await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert ica.calls == [] and (done.status, done.skipped) == ("completed", 1)


async def test_a_new_prompt_or_story_frame_redoes_the_text_and_what_hangs_on_it():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "full")
    db = mongo.get_default_database()
    await db["examination_simplified"].update_one(
        {"question_id": str(q["_id"])}, {"$set": {"frame_version": "older"}}
    )
    ica.calls.clear()
    # Same text comes back, so its entities and its check still fit and are kept.
    await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert ica.calls == ["simplify"]

    await db["examination_simplified"].update_one(
        {"question_id": str(q["_id"])}, {"$set": {"prompt_version": "older"}}
    )
    ica.calls.clear()
    text = "Which mode, plan mode or acceptEdits mode, never edits?"
    other = {**{k: v for k, v in SIMPLE.items() if k != "question"}, "parts": [{"role": "ask", "text": text, "entity_ids": []}]}
    ica.queues["simplify"].append(reply(other))
    await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    # The entities are derived again (no call); the icons and the check follow the text.
    assert ica.calls == ["simplify", "icons", "check"]
    assert (await stored(mongo, q))["prompt"]["question"] == text


async def test_a_failed_step_does_not_stop_the_steps_that_do_not_need_it():
    # No objectives for the section and no chapter for the paragraph: both are refused
    # without a call, and the game text is still written, in the company only.
    q = outsider()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "full")
    assert ica.calls == ["decompose", "simplify", "icons", "flow", "resolve", "check"]
    assert (job.status, job.failed) == ("completed", 1)
    assert job.stages_failed == {"objectives": 1, "story": 1}
    assert [e.status_code for e in job.errors] == [422, 422]
    assert (await stored(mongo, q))["frame_version"] == frame_version(None)

    # Without a game text there is nothing to extract from or to check.
    q2 = ready()
    mongo, ica = FakeMongo([q2]), ScriptedIca(simplify=[HTTPException(status_code=400, detail="x")])
    job = await run_job(mongo, ica, [q2["_id"]], "full")
    # The flow and its resolution need only the decomposition, so they still run.
    assert ica.calls == ["objectives", "story", "decompose", "simplify", "flow", "resolve"]
    assert job.stages_failed == {"simplify": 1} and job.status == "completed"


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
    assert record["catalog_version"] == chapter_version("research")
    # The game then serves that chapter, and the game text was set in it.
    assert story_of(q, record)["scenario"] == "research" and story_of(q, record)["stem"] == ""
    assert (await stored(mongo, q))["frame_version"] == frame_version("research")
    # A record for another text of the question says nothing about this one.
    assert story_of(q, {**record, "text_hash": "other"})["scenario"] == ""
    # And a paragraph no chapter claims is never overruled by a record.
    assert story_of(outsider(), {**record, "text_hash": "x"})["scenario"] == ""


# --- the game text --------------------------------------------------------------------

ITEM = {
    "correct_response": ["b"],
    "section": "Domain 1",
    "prompt": {
        "question": (
            "SCENARIO: You are building a customer support resolution agent. QUESTION: "
            "The `PostToolUse` hook fires on 12% of refunds above $500. What should change?"
        ),
        "answers": [
            "Set `tool_choice` to any in the system prompt of the subagent.",
            "Move the limit into a PreToolUse hook that blocks process_refund.",
        ],
        "feedbacks": ["no", "yes"],
        "explanation": "Hooks enforce.",
    },
}
GOOD = {
    "question": "Mara shows you the logs. The PostToolUse hook fires on 12% of refunds above $500. What should change?",
    "answers": [
        "Set tool_choice to any in the subagent's system prompt.",
        "Put the limit in a PreToolUse hook that blocks process_refund.",
    ],
    "feedbacks": ["Incorrect. A prompt cannot enforce.", "Correct. A hook can block."],
    "explanation": "A hook enforces. Rule: enforce in code.",
}


def test_exact_terms_are_found_by_form_and_by_name():
    text = (
        "Run `claude -p` with --output-format json, edit .claude/rules/api.md and ~/.claude.json, "
        "call mcp__billing__refund from a PreToolUse hook, set allowedTools, use /compact and CLAUDE.md."
    )
    found = identifiers(text)
    for term in ("claude -p", "--output-format", ".claude/rules/api.md", "~/.claude.json",
                 "mcp__billing__refund", "PreToolUse", "allowedTools", "/compact", "CLAUDE.md"):
        assert term in found, term
    assert "and/or" not in " ".join(identifiers("first-contact and/or the @-mention at src/billing/x"))

    assert lost_terms("Use a subagent in plan mode for the PR.", "Use sub-agents in plan mode on the pull request.") == []
    assert lost_terms("Use a subagent in plan mode.", "Use a helper in read-only mode.") == ["subagent", "plan mode"]
    assert lost_numbers("stuck at 64% after 5,000 runs", "stuck at 64 percent after 5,000 runs") == []
    assert lost_numbers("stuck at 64%", "stuck at about two thirds") == ["64"]


def test_a_faithful_rewrite_passes_and_each_loss_is_named():
    assert fidelity_errors(ITEM, GOOD) == []

    def errors(**changes):
        return " | ".join(fidelity_errors(ITEM, {**GOOD, **changes}))

    assert "answer A must keep these exact terms: tool_choice" in errors(
        answers=["Force a tool call in the subagent's system prompt.", GOOD["answers"][1]]
    )
    # The term has to stay in its own answer, not somewhere in the reply.
    assert "answer B must keep these exact terms: PreToolUse" in errors(
        answers=[GOOD["answers"][0] + " Not a PreToolUse hook.", "Put the limit in a hook that blocks process_refund."]
    )
    assert "the question must keep these exact terms: PostToolUse" in errors(
        question="Mara shows you the logs. A hook fires on 12% of refunds above $500. What should change?"
    )
    assert "numbers as digits: 12" in errors(
        question="The PostToolUse hook fires on some refunds above $500. What should change?"
    )
    assert "answer A is longer than the original" in errors(
        answers=[GOOD["answers"][0] + " " + "This adds a long explanation nobody asked for." * 2, GOOD["answers"][1]]
    )
    # A named concept may leave one answer as long as the question still names it...
    assert errors(answers=[GOOD["answers"][0], "Put the limit in a PreToolUse check that blocks process_refund."]) == ""
    # ...but not be paraphrased away everywhere.
    assert "must still name: hook" in errors(
        question="Mara shows you the logs. PostToolUse fires on 12% of refunds above $500. What should change?",
        answers=[GOOD["answers"][0], "Put the limit in a PreToolUse check that blocks process_refund."],
    )
    # The scenario paragraph is background: its words need not come back.
    assert "customer support" not in GOOD["question"]


def test_messages_set_the_question_in_its_chapter():
    system = _build_system_message(2, "support")
    assert "{{" not in system and BY_KEY["support"].intro in system
    assert "Mara (head of support)" in system and 'start each feedback with "Correct."'.lower() in system.lower()
    assert "evaluator-optimizer" in system
    plain = _build_system_message(2)
    assert "{{" not in plain and "People of the chapter" not in plain and "THIS QUESTION'S CHAPTER" not in plain

    user = _build_user_message(ITEM, {"stage": "incident", "beat": "Refunds over the limit slip through."})
    assert "STAGE OF THE CHAPTER: Incident" in user
    assert "LINE SHOWN BEFORE THE QUESTION:\nRefunds over the limit slip through." in user
    assert "SCENARIO:\nYou are building a customer support resolution agent." in user
    assert "QUESTION:\nThe `PostToolUse` hook fires" in user and "CORRECT ANSWER: B" in user
    assert "STAGE OF THE CHAPTER" not in _build_user_message(ITEM)
    # Who opens the question is decided here, not by the model, so the same few people
    # do not open every question of a chapter.
    openings = {opening({"prompt": {"question": f"SCENARIO: a customer support resolution agent. QUESTION: q{i}"}}, "support") for i in range(40)}
    assert openings == {
        "OPENING: no person. Start with the situation itself.",
        "OPENING: Mara (head of support) brings this to you.",
        "OPENING: Omar (engineer on the support agent) brings this to you.",
    }
    assert opening(ITEM, None) == "OPENING: no person. Start with the situation itself."
    assert opening(ITEM, "support") in user


async def test_a_rejected_reply_is_sent_back_with_itself_and_what_was_wrong():
    lossy = {**GOOD, "answers": ["Force a tool call in the subagent's system prompt.", GOOD["answers"][1]]}

    class Ica:
        def __init__(self, *replies):
            self.replies, self.prompts = list(replies), []

        async def chat_model(self, model_id, prompt, system=None, max_tokens=None, json_object=False):
            self.prompts.append(prompt)
            return reply(self.replies.pop(0))

    ica = Ica(lossy, GOOD)
    prompt, meta = await simplify_item(ica, ITEM, "claude-haiku-5-5")
    assert prompt.answers == GOOD["answers"] and meta.total_tokens == 14
    assert "YOUR PREVIOUS REPLY" not in ica.prompts[0]
    # The model gets its own text back, so it can change one place and keep the rest.
    assert "Force a tool call in the subagent's system prompt." in ica.prompts[1]
    assert "answer A must keep these exact terms: tool_choice" in ica.prompts[1]

    stubborn = Ica(*[lossy] * MAX_ATTEMPTS)
    with pytest.raises(HTTPException) as exc:
        await simplify_item(stubborn, ITEM, "claude-haiku-5-5")
    assert exc.value.status_code == 502 and "tool_choice" in exc.value.detail
    assert len(stubborn.prompts) == MAX_ATTEMPTS


# --- the check ------------------------------------------------------------------------


async def run_check(*answers):
    q = ready()
    mongo = FakeMongo([q])
    ica = ScriptedIca(check=[reply({"reason": "r", "answer": a}) for a in answers])
    job = await run_job(mongo, ica, [q["_id"]], "full")
    return q, ica, job, (await stored(mongo, q))


async def test_check_asks_the_original_only_when_the_game_text_is_answered_wrongly():
    q, ica, job, record = await run_check("B")
    assert ica.calls.count("check") == 1
    assert (record["check"]["verdict"], record["check"]["picked_original"]) == ("same", None)
    assert simplified_matches(q, record)

    # Wrong on the game text, right on the original: the rewrite moved the answer.
    q, ica, job, record = await run_check("A", "B")
    assert ica.calls.count("check") == 2 and job.succeeded == 1
    assert (record["check"]["verdict"], record["check"]["picked"], record["check"]["picked_original"]) == ("changed", 0, 1)
    assert not simplified_matches(q, record)

    # Wrong on both: a hard question, nothing against the rewrite.
    q, ica, job, record = await run_check("A", "A")
    assert record["check"]["verdict"] == "hard" and simplified_matches(q, record)


async def test_check_job_and_route_work_on_stored_texts():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "simplify_only")
    assert (await stored(mongo, q))["check"] is None

    job = await run_job(mongo, ica, [q["_id"]], "check", only_missing=True)
    assert (job.mode, job.succeeded) == ("check", 1)
    again = await run_job(mongo, ica, [q["_id"]], "check", only_missing=True)
    assert again.skipped == 1 and ica.calls == ["simplify", "check"]

    client = client_for(mongo, ica)
    one = client.post(f"/examination/{q['_id']}/check")
    assert one.status_code == 200 and one.json()["check"]["verdict"] == "same"
    other = ready()
    assert client_for(FakeMongo([other]), ica).post(f"/examination/{other['_id']}/check").status_code == 404


def test_check_reply_must_name_an_answer():
    assert parse_pick({"answer": " b "}, 4) == 1
    for bad in ({"answer": "E"}, {"answer": "AB"}, {"answer": 2}, {}):
        with pytest.raises(HTTPException) as exc:
            parse_pick(bad, 4)
        assert exc.value.status_code == 502


# --- difficulty -----------------------------------------------------------------------


def test_difficulty_follows_the_tempting_answers_and_the_level():
    assert difficulty_of("remember", 0) == 1 and difficulty_of("apply", 1) == 1
    assert difficulty_of("apply", 2) == 2 and difficulty_of("analyze", 1) == 2
    assert difficulty_of("analyze", 2) == 3 and difficulty_of("apply", 3) == 3
