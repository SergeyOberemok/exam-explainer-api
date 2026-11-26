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
    decomposed,
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
from claude_exam.api.check import parse_pick
from claude_exam.api.exam_terms import identifiers, lost_numbers, lost_terms
from claude_exam.api.game.domain import simplified_matches, story_of
from claude_exam.api.ingest import plan_import
from claude_exam.api.normalization import normalize_result
from claude_exam.api.objectives import difficulty_of
from claude_exam.api.scenario_catalog import BY_KEY
from claude_exam.api.question_view import key_free
from claude_exam.api.rephrase import (
    MAX_ATTEMPTS,
    _build_system_message,
    _build_user_message,
    fidelity_errors,
    opening,
    rephrase_item,
)
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

    # No rephrase any more; the last step composed the game document.
    assert await stored(mongo, q) is None
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


async def test_a_new_prompt_or_story_frame_redoes_the_rephrase_and_what_hangs_on_it(monkeypatch):
    from claude_exam.api import rephrase

    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "full")
    await run_job(mongo, ica, [q["_id"]], "rephrase")
    # The company's intro changed (the frame the rephrase is written in; nothing bumped).
    monkeypatch.setattr(rephrase, "COMPANY_INTRO", rephrase.COMPANY_INTRO + " Now also in Lisbon.")
    ica.calls.clear()
    # Same text comes back, so its check still fits and is kept.
    await run_job(mongo, ica, [q["_id"]], "rephrase", only_missing=True)
    assert ica.calls == ["rephrase"]

    # The prompt changed, and another text comes back: it is checked again.
    monkeypatch.setattr(rephrase, "_SYSTEM_MESSAGE", rephrase._SYSTEM_MESSAGE + "\nNew rule.")
    ica.calls.clear()
    text = "Which mode, plan mode or acceptEdits mode, never edits?"
    ica.queues["rephrase"].append(reply({**SIMPLE, "question": text}))
    await run_job(mongo, ica, [q["_id"]], "rephrase", only_missing=True)
    assert ica.calls == ["rephrase", "check"]
    assert (await stored(mongo, q))["prompt"]["question"] == text


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
    # The game then serves that chapter, and the rephrase was set in it.
    assert story_of(q, record)["scenario"] == "research" and story_of(q, record)["stem"] == ""
    # A record for another text of the question says nothing about this one.
    assert story_of(q, {**record, "text_hash": "other"})["scenario"] == ""
    # And a paragraph no chapter claims is never overruled by a record.
    assert story_of(outsider(), {**record, "text_hash": "x"})["scenario"] == ""


# --- the rephrase --------------------------------------------------------------------

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
# A rephrase reply: the question and the answers only.
GOOD = {
    "question": "Mara shows you the logs. The PostToolUse hook fires on 12% of refunds above $500. What should change?",
    "answers": [
        "Set tool_choice to any in the subagent's system prompt.",
        "Put the limit in a PreToolUse hook that blocks process_refund.",
    ],
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
    # A rewrite may not sell its answer with a benefit the original does not state...
    assert "answer B adds a reason or benefit" in errors(
        answers=[GOOD["answers"][0], "Put the limit in a PreToolUse hook, so process_refund never runs."]
    )
    assert "answer A adds a reason or benefit" in errors(
        answers=["Set tool_choice to any in the subagent's prompt to ensure a call.", GOOD["answers"][1]]
    )


def test_a_purpose_the_original_states_may_stay():
    from claude_exam.api.rephrase import added_purpose

    assert not added_purpose("Raise max_tokens so that it can reason.", "Raise max_tokens so it can reason.")
    assert added_purpose("Raise max_tokens.", "Raise max_tokens so that it can reason.")
    # "so" alone inside a clause is not a purpose ("do so", "so far").
    assert not added_purpose("Log every call.", "Log every call and do so for each tool.")


def test_messages_set_the_question_in_its_chapter():
    system = _build_system_message(2, "support")
    assert "{{" not in system and BY_KEY["support"].intro in system
    assert "Mara (head of support)" in system and "feedback" not in system.lower()
    plain = _build_system_message(2)
    assert "{{" not in plain and "People of the chapter" not in plain and "THIS QUESTION'S CHAPTER" not in plain

    place = {"scenario": "support", "stage": "incident", "beat": "Refunds over the limit slip through."}
    user = _build_user_message(key_free(ITEM), place)
    assert "STAGE OF THE CHAPTER: Incident" in user
    assert "LINE SHOWN BEFORE THE QUESTION:\nRefunds over the limit slip through." in user
    assert "SCENARIO:\nYou are building a customer support resolution agent." in user
    assert "QUESTION:\nThe `PostToolUse` hook fires" in user
    # Written without the key: no correct letter, feedback or explanation.
    assert "CORRECT" not in user and "Hooks enforce." not in user and "FEEDBACK" not in user
    assert "STAGE OF THE CHAPTER" not in _build_user_message(key_free(ITEM))
    # Who opens the question is decided here, not by the model, so the same few people
    # do not open every question of a chapter.
    openings = {opening(key_free({"prompt": {"question": f"SCENARIO: a customer support resolution agent. QUESTION: q{i}"}}), "support") for i in range(40)}
    assert openings == {
        "OPENING: no person. Start with the situation itself.",
        "OPENING: Mara (head of support) brings this to you.",
        "OPENING: Omar (engineer on the support agent) brings this to you.",
    }
    assert opening(key_free(ITEM), None) == "OPENING: no person. Start with the situation itself."
    assert opening(key_free(ITEM), "support") in user


async def test_a_rejected_reply_is_sent_back_with_itself_and_what_was_wrong():
    lossy = {**GOOD, "answers": ["Force a tool call in the subagent's system prompt.", GOOD["answers"][1]]}

    class Ica:
        def __init__(self, *replies):
            self.replies, self.prompts = list(replies), []

        async def chat_model(self, model_id, prompt, system=None, max_tokens=None, json_object=False):
            self.prompts.append(prompt)
            return reply(self.replies.pop(0))

    ica = Ica(lossy, GOOD)
    prompt, meta = await rephrase_item(ica, ITEM, "claude-haiku-5-5")
    assert prompt.answers == GOOD["answers"] and meta.total_tokens == 14
    assert "YOUR PREVIOUS REPLY" not in ica.prompts[0]
    # The model gets its own text back, so it can change one place and keep the rest.
    assert "Force a tool call in the subagent's system prompt." in ica.prompts[1]
    assert "answer A must keep these exact terms: tool_choice" in ica.prompts[1]

    stubborn = Ica(*[lossy] * MAX_ATTEMPTS)
    with pytest.raises(HTTPException) as exc:
        await rephrase_item(stubborn, ITEM, "claude-haiku-5-5")
    assert exc.value.status_code == 502 and "tool_choice" in exc.value.detail
    assert len(stubborn.prompts) == MAX_ATTEMPTS


# --- the check ------------------------------------------------------------------------


async def run_check(*answers):
    q = ready()
    mongo = FakeMongo([q])
    ica = ScriptedIca(check=[reply({"reason": "r", "answer": a}) for a in answers])
    await run_job(mongo, ica, [q["_id"]], "story")
    job = await run_job(mongo, ica, [q["_id"]], "rephrase")
    return q, ica, job, (await stored(mongo, q))


async def test_check_asks_the_original_only_when_the_rephrase_is_answered_wrongly():
    q, ica, job, record = await run_check("B")
    assert ica.calls.count("check") == 1
    assert (record["check"]["verdict"], record["check"]["picked_original"]) == ("same", None)
    assert simplified_matches(q, record)

    # Wrong on the rephrase, right on the original: the rewrite moved the answer.
    q, ica, job, record = await run_check("A", "B")
    assert ica.calls.count("check") == 2 and job.succeeded == 1
    assert (record["check"]["verdict"], record["check"]["picked"], record["check"]["picked_original"]) == ("changed", 0, 1)
    assert not simplified_matches(q, record)

    # Wrong on both: a hard question, nothing against the rewrite.
    q, ica, job, record = await run_check("A", "A")
    assert record["check"]["verdict"] == "hard" and simplified_matches(q, record)


async def test_check_job_and_route_work_on_stored_rephrases():
    q = ready()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client_for(mongo, ica).post(f"/questions/{q['_id']}/rephrase")
    assert (await stored(mongo, q))["check"] is None

    job = await run_job(mongo, ica, [q["_id"]], "check", only_missing=True)
    assert (job.mode, job.succeeded) == ("check", 1)
    again = await run_job(mongo, ica, [q["_id"]], "check", only_missing=True)
    assert again.skipped == 1 and ica.calls == ["rephrase", "check"]

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


def test_a_rewrite_may_not_widen_a_fact():
    from claude_exam.api.exam_terms import widened

    assert widened("The agent posts these as definitive.", "The agent posts every finding as definitive.") == ["every"]
    assert widened("It reads all files.", "It reads all the files.") == []
    # "never" is how a rewrite says "does not": not a wider scope.
    assert widened("It does not edit.", "It never edits.") == []
    assert "the question says every" in " | ".join(
        fidelity_errors(ITEM, {**GOOD, "question": GOOD["question"].replace("on 12%", "on every one of 12%")})
    )
