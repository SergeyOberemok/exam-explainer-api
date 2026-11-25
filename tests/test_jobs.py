"""The ingest jobs: job modes (decomposition with its parts and entities, icons,
rephrase and check), the routes that start them, and the standalone steps.

The last test runs the same pipeline against a real MongoDB (opt-in, like
`tests/test_game_mongo.py`): TEST_MONGO_URL=mongodb://127.0.0.1:27017.
"""

import asyncio
import copy
import json
import re
import os
import secrets
from types import SimpleNamespace

import pytest
from bson import ObjectId
from fakes import MISSING, get_path, matches, set_path
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient

from claude_exam.api import batch
from claude_exam.api.batch import new_job, run_simplification_job
from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_decomposition_collection,
    get_async_jobs_collection,
    get_async_mongo_client,
    get_async_objectives_collection,
    get_async_simplified_collection,
    get_async_story_collection,
    get_ica_client,
    get_simplification_lock,
)
from claude_exam.api.main import app
from claude_exam.api.scenario_catalog import text_hash
from claude_exam.api.schemas import DecompositionRecord, PromptEntities, SimplificationJob
from claude_exam.api.story import ICA_ORDER as STORY_ORDER

pytestmark = pytest.mark.anyio
MONGO_URL = os.environ.get("TEST_MONGO_URL")
MODEL = "claude-haiku-5-5"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)


# --- fakes ------------------------------------------------------------------------


class _Cursor:
    def __init__(self, docs):
        self.docs = docs

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for doc in self.docs:
            yield doc


class Coll:
    """The collection operations the pipeline and its routes send (copies on read, like
    a real driver)."""

    def __init__(self, *docs):
        self.docs = [copy.deepcopy(d) for d in docs]

    def _first(self, query):
        return next((d for d in self.docs if matches(d, query)), None)

    @staticmethod
    def _apply(doc, update):
        for key, value in update.get("$set", {}).items():
            set_path(doc, key, copy.deepcopy(value))
        for key, value in update.get("$inc", {}).items():
            current = get_path(doc, key)
            set_path(doc, key, (0 if current is MISSING else current) + value)
        for key, value in update.get("$push", {}).items():
            items = value["$each"] if isinstance(value, dict) and "$each" in value else [value]
            doc.setdefault(key, []).extend(copy.deepcopy(items))

    async def find_one(self, query, projection=None):
        doc = self._first(query)
        return None if doc is None else copy.deepcopy(doc)

    def find(self, query=None, projection=None):
        return _Cursor([copy.deepcopy(d) for d in self.docs if matches(d, query or {})])

    async def insert_one(self, doc):
        doc = {"_id": ObjectId(), **copy.deepcopy(doc)}
        self.docs.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    async def insert_many(self, docs):
        return SimpleNamespace(inserted_ids=[(await self.insert_one(d)).inserted_id for d in docs])

    async def update_one(self, query, update):
        doc = self._first(query)
        if doc is not None:
            self._apply(doc, update)

    async def find_one_and_update(self, query, update, upsert=False, return_document=None):
        doc = self._first(query)
        if doc is None:
            if not upsert:
                return None
            doc = {"_id": ObjectId(), **{k: v for k, v in query.items() if not isinstance(v, dict)}}
            doc.update(copy.deepcopy(update.get("$setOnInsert", {})))
            self.docs.append(doc)
        self._apply(doc, update)
        return copy.deepcopy(doc)


class FakeMongo:
    def __init__(self, examination=()):
        self.db = {
            "examination": Coll(*examination),
            "examination_simplified": Coll(),
            "question_objectives": Coll(),
            "question_story": Coll(),
            "question_decomposition": Coll(),
            "simplification_jobs": Coll(),
        }

    def get_default_database(self):
        return self.db


def reply(payload):
    return {
        "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 7},
    }


# The default rephrase reply (`rephrase.py`): the question and answers only.
SIMPLE = {
    "question": "Which mode, plan mode or acceptEdits mode, reads but never edits?",
    "answers": ["acceptEdits mode", "plan mode"],
}


def icons_reply():
    return {"entities": {"plan mode": "plan"}, "question": "mode", "answers": ["write", "plan"]}


def objectives_reply():
    return {
        "primary": "claude-code-planning",
        "secondary": [],
        "level": "apply",
        "tempting": [],
        "distractors": [{"misconception": "Auto-accepting edits keeps files unchanged.", "confused_with": None}, None],
    }


def story_reply():
    return {
        "stage": "build",
        "beat": "The team cannot agree on a working mode.",
        "win": "Reviews happen before edits.",
        "fail": "Edits land unreviewed.",
    }


_LETTER_LINE = re.compile(r"^  ([A-Z])\. (.*)$", re.M)


def answer_letters(prompt):
    """The answer letters of the first answer list in a user message, and which of them
    is marked [CORRECT] (the resolution step's input)."""
    block = prompt.split("ANSWER", 1)[-1]
    lines = _LETTER_LINE.findall(block)
    letters = list(dict.fromkeys(letter for letter, _ in lines))
    correct = [ord(l) - 65 for l, text in lines if "[CORRECT]" in text]
    return letters, correct


def _fact(text):
    return {"text": text, "source_status": "inferred", "evidence": ""}


def decomposition_reply(n_answers=2):
    """A minimal decomposition that passes `decomposition_errors` for any question:
    every item is inferred, so no evidence has to be quoted."""
    return {
        "question_summary": "A team must choose a working mode.",
        "question_intent": {
            "primary": "comparison",
            "secondary": [],
            "question_task": "choose the mode",
            "target_of_question": "working mode",
            "success_criteria": [],
        },
        "context": {
            "description": "A team works with an agent.",
            "current_state": [_fact("Edits land unreviewed.")],
            "desired_state": [],
        },
        "entities": [
            {"id": "e1", "name": "Agent", "type": "actor", "aliases": [], "description": "Does the work", "source_status": "inferred", "evidence": ""},
            {"id": "e2", "name": "Ambiguity", "type": "concept", "aliases": [], "description": "Unclear rules", "source_status": "inferred", "evidence": ""},
        ],
        "events": [],
        "relationships": [],
        "problem": {"symptoms": [_fact("Edits land unreviewed.")], "root_causes": [], "consequences": [], "affected_entities": ["e1"]},
        "conditions": [],
        "goals_and_requirements": {"business_goals": [], "behavioral_requirements": [], "quality_requirements": [], "constraints": []},
        "answer_options": [{"index": i, "mechanism": "a proposal", "entity_ids": []} for i in range(n_answers)],
        "ambiguities_and_unknowns": [],
    }


def decomposition_with_plan_mode(n_answers=2):
    """`decomposition_reply` plus an entity only answer B names, so the derived
    entities have one to draw (the default ones are not in the question's words)."""
    d = decomposition_reply(n_answers)
    d["entities"].append(
        {"id": "e3", "name": "plan mode", "type": "object", "aliases": [], "description": "A mode", "source_status": "inferred", "evidence": ""}
    )
    d["answer_options"][1]["entity_ids"] = ["e3"]
    return d


def _node(id, type, label, **kw):
    return {"id": id, "type": type, "label": label, "entity_ids": [], "condition_id": "", "source_status": "inferred", "is_question": False, "added": False, **kw}


def _edge(source, target, **kw):
    return {"source": source, "target": target, "type": "next", "label": "", "source_status": "inferred", "added": False, **kw}


VIEW = {"primary": "process_flow", "secondary": [], "reason": "No branching."}


def flow_reply():
    return {
        "visualization": VIEW,
        "trigger": "Work starts",
        "nodes": [
            _node("n1", "start", "Work starts"),
            _node("n2", "process", "Edits happen"),
            _node("n3", "decision", "Which mode?", is_question=True),
        ],
        "edges": [_edge("n1", "n2"), _edge("n2", "n3")],
    }


def resolution_reply(n_answers=2, correct=(1,)):
    return {
        "answer_requirements": ["No edit before review"],
        "flow": {
            "visualization": VIEW,
            "trigger": "Work starts",
            "nodes": [
                _node("n1", "start", "Work starts"),
                _node("n2", "process", "Edits happen"),
                _node("n4", "process", "Plan before editing", added=True),
                _node("n5", "end", "Reviewed changes", added=True),
            ],
            "edges": [_edge("n1", "n2"), _edge("n2", "n4", added=True), _edge("n4", "n5", added=True)],
        },
        "options": [
            {"index": i, "verdict": "correct" if i in correct else "incorrect", "supported_requirements": [], "gaps": [], "reason": "Because."}
            for i in range(n_answers)
        ],
        "rule": "Plan first when edits need review.",
    }


class ScriptedIca:
    """Answers the calls of each pipeline step from separate queues (an
    Exception item is raised); defaults to good replies once a queue is empty. Records
    the call kinds."""

    DEFAULTS = {
        "rephrase": lambda: SIMPLE,
        "icons": icons_reply,
        "objectives": objectives_reply,
        "story": lambda: story_reply(),
        "check": lambda: {"reason": "It blocks edits.", "answer": "B"},
    }

    def __init__(
        self,
        rephrase=(),
        icons=(),
        objectives=(),
        story=(),
        check=(),
        decompose=(),
        flow=(),
        resolve=(),
    ):
        self.queues = {
            "rephrase": list(rephrase),
            "icons": list(icons),
            "objectives": list(objectives),
            "story": list(story),
            "check": list(check),
            "decompose": list(decompose),
            "flow": list(flow),
            "resolve": list(resolve),
        }
        self.calls = []
        self.max_tokens = []
        self.formats = []

    def default(self, kind, prompt, system):
        letters, correct = answer_letters(prompt)
        if kind == "decompose":
            return decomposition_reply(len(letters))
        if kind == "flow":
            return flow_reply()
        if kind == "resolve":
            return resolution_reply(len(letters), correct)
        payload = copy.deepcopy(self.DEFAULTS[kind]())
        if kind == "icons" and letters:
            # One answer icon per answer of the question asked about.
            payload["answers"] = (payload["answers"] + ["mode"] * len(letters))[: len(letters)]
        return payload

    async def chat_model(
        self, model_id, prompt, system=None, max_tokens=None, json_object=False, response_format=None
    ):
        system = system or ""
        self.formats.append((response_format or {}).get("type") or ("json_object" if json_object else None))
        kind = (
            "decompose" if "question decomposition engine" in system
            else "flow" if "You draw the flow" in system
            else "resolve" if "You explain the resolution" in system
            else "icons" if "icon catalog" in system
            else "objectives" if "learning objective" in system
            else "story" if "story of a quiz game" in system
            else "check" if "You answer one multiple-choice question" in system
            else "rephrase"
        )
        self.calls.append(kind)
        self.max_tokens.append(max_tokens)
        item = self.queues[kind].pop(0) if self.queues[kind] else None
        if isinstance(item, Exception):
            raise item
        return item or reply(self.default(kind, prompt, system))


def question(**kw):
    return {
        "_id": ObjectId(),
        "section": "Domain 3",
        "assessment_type": "multiple-choice",
        "source": "x.json",
        "correct_response": ["b"],
        "related_lectures": [],
        "prompt": {
            "question": "Which mode?",
            "answers": ["acceptEdits mode", "plan mode"],
            "feedbacks": ["no", "yes"],
            "explanation": "Plan mode blocks edits.",
        },
        **kw,
    }


async def run_job(mongo, ica, ids, mode, only_missing=False):
    jobs = mongo.get_default_database()["simplification_jobs"]
    job = await jobs.insert_one(new_job("t", MODEL, len(ids), mode, only_missing))
    await run_simplification_job(
        mongo, ica, asyncio.Semaphore(1), job.inserted_id, ids, MODEL, mode, only_missing
    )
    doc = await jobs.find_one({"_id": job.inserted_id})
    return SimplificationJob(**{**doc, "id": str(doc.pop("_id"))})


async def stored(mongo, doc):
    """The question's rephrase record."""
    db = mongo.get_default_database()
    return await db["examination_simplified"].find_one({"question_id": str(doc["_id"])})


async def decomposed(mongo, doc):
    """The question's decomposition record (with its parts, entities and icons)."""
    db = mongo.get_default_database()
    return await db["question_decomposition"].find_one({"question_id": str(doc["_id"])})


# --- job modes ---------------------------------------------------------------------


async def test_decompose_mode_stores_parts_entities_icons_flow_and_resolution():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca(decompose=[reply(decomposition_with_plan_mode())])
    job = await run_job(mongo, ica, [q["_id"]], "decompose")
    assert ica.calls == ["decompose", "icons", "flow", "resolve"]
    assert (job.mode, job.status, job.succeeded, job.failed) == ("decompose", "completed", 1, 0)
    record = await decomposed(mongo, q)
    # Parts and entities are built in code: no call of their own.
    assert record["parts"][-1] == {"role": "ask", "text": "Which mode?", "entity_ids": []}
    assert record["entities"]["answers"][1] == {
        "entities": [{"name": "plan mode", "kind": "object", "aliases": [], "ref": "e3", "icon": "plan"}],
        "icon": "plan",
    }
    assert record["entities"]["question_icon"] == "mode" and record["icons_ica"] is not None
    assert record["flow"] and record["resolution"]
    assert await stored(mongo, q) is None  # the rephrase is a step of its own


async def test_icons_failure_after_the_decomposition_is_partial_not_a_failed_job():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(icons=[HTTPException(status_code=400, detail="boom")])
    job = await run_job(mongo, ica, [q["_id"]], "decompose")
    assert (job.status, job.succeeded, job.failed, job.icons_failed) == ("completed", 0, 1, 1)
    assert job.errors[0].stage == "icons"
    record = await decomposed(mongo, q)
    assert record["icons_ica"] is None and record["flow"]  # the flow does not need icons


async def test_icons_job_picks_only_where_missing_and_keeps_the_rest():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(
        decompose=[reply(decomposition_with_plan_mode())],
        icons=[HTTPException(status_code=400, detail="x")],
    )
    await run_job(mongo, ica, [q["_id"]], "decompose")  # entities stored, icons failed
    before = await decomposed(mongo, q)
    ica.calls.clear()

    job = await run_job(mongo, ica, [q["_id"]], "icons", only_missing=True)
    after = await decomposed(mongo, q)
    assert ica.calls == ["icons"]
    assert (job.mode, job.status, job.succeeded) == ("icons", "completed", 1)
    assert after["entities"]["answers"][1]["entities"][0]["icon"] == "plan"
    for unchanged in ("_id", "semantic", "parts", "flow", "created_at"):
        assert after[unchanged] == before[unchanged]

    again = await run_job(mongo, ica, [q["_id"]], "icons", only_missing=True)
    assert ica.calls == ["icons"] and again.skipped == 1


async def test_a_new_decomposition_drops_the_old_icons():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca(decompose=[reply(decomposition_with_plan_mode())] * 2)
    await run_job(mongo, ica, [q["_id"]], "decompose")
    ica.queues["icons"].append(HTTPException(status_code=400, detail="x"))
    job = await run_job(mongo, ica, [q["_id"]], "decompose")
    record = await decomposed(mongo, q)
    assert job.icons_failed == 1
    assert record["icons_ica"] is None
    assert record["entities"]["answers"][1]["entities"][0]["icon"] is None


async def test_rephrase_mode_writes_the_rephrase_and_checks_it():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "rephrase")
    assert ica.calls == ["rephrase", "check"]
    assert (job.status, job.succeeded) == ("completed", 1)
    record = await stored(mongo, q)
    assert record["prompt"] == SIMPLE and record["correct_response"] == ["b"]
    assert record["check"]["verdict"] == "same" and record["input_key"]


async def test_a_failed_rephrase_is_reported_with_its_stage_and_nothing_is_checked():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(rephrase=[HTTPException(status_code=400, detail="bad key")])
    job = await run_job(mongo, ica, [q["_id"]], "rephrase")
    assert ica.calls == ["rephrase"]  # no check without a rephrase
    assert (job.status, job.failed) == ("failed", 1)
    assert job.errors[0].stage == "rephrase"
    assert await stored(mongo, q) is None


async def test_a_step_retries_transient_errors():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(rephrase=[HTTPException(status_code=429, detail="slow down")])
    job = await run_job(mongo, ica, [q["_id"]], "rephrase")
    assert ica.calls == ["rephrase", "rephrase", "check"]
    assert job.succeeded == 1


async def test_a_job_skips_what_another_job_did_meanwhile():
    """only_missing is checked when the job reaches a question, not when it was queued."""
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "rephrase")  # e.g. the import job it waited for
    ica.calls.clear()

    job = await run_job(mongo, ica, [q["_id"]], "rephrase", only_missing=True)
    assert ica.calls == []
    assert (job.status, job.succeeded, job.skipped, job.failed) == ("completed", 0, 1, 0)


async def test_an_icons_job_reports_a_question_without_a_decomposition():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "icons", only_missing=True)
    assert ica.calls == []
    assert (job.status, job.errors[0].stage, job.errors[0].status_code) == ("failed", "icons", 404)


# --- objectives --------------------------------------------------------------------

TAGGABLE = "Domain 3: Claude Code Configuration & Workflows"


async def tags(mongo, doc):
    return await mongo.get_default_database()["question_objectives"].find_one(
        {"question_id": str(doc["_id"])}
    )


async def test_objectives_job_tags_without_a_simplified_record():
    q = question(section=TAGGABLE)
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "objectives")
    assert ica.calls == ["objectives"]
    assert (job.mode, job.status, job.succeeded) == ("objectives", "completed", 1)
    record = await tags(mongo, q)
    assert record["objectives"] == ["claude-code-planning"] and record["section"] == TAGGABLE
    assert (record["level"], record["tempting"], record["difficulty"]) == ("apply", [], 1)
    assert record["distractors"][1] is None and record["distractors"][0]["misconception"]
    assert record["input_key"] and record["created_at"] == record["updated_at"]
    assert await stored(mongo, q) is None


async def test_objectives_job_skips_current_records_and_redoes_outdated_ones(monkeypatch):
    q = question(section=TAGGABLE)
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "objectives")
    first = await tags(mongo, q)

    job = await run_job(mongo, ica, [q["_id"]], "objectives", only_missing=True)
    assert (job.skipped, job.succeeded) == (1, 0) and ica.calls == ["objectives"]

    # The prompt changed since (no version bumped): the record is outdated, tagged again.
    from claude_exam.api import objectives

    monkeypatch.setattr(objectives, "_SYSTEM_MESSAGE", objectives._SYSTEM_MESSAGE + "\nNew rule.")
    db = mongo.get_default_database()
    job = await run_job(mongo, ica, [q["_id"]], "objectives", only_missing=True)
    assert (job.skipped, job.succeeded) == (0, 1) and ica.calls == ["objectives"] * 2
    again = await tags(mongo, q)
    assert again["input_key"] != first["input_key"] and again["created_at"] == first["created_at"]
    assert len(db["question_objectives"].docs) == 1


async def test_objectives_job_retries_a_rejected_reply_but_not_a_missing_catalog():
    good, other = question(section=TAGGABLE), question(section="Not in the catalog")
    bad = reply({**objectives_reply(), "primary": "not-in-the-catalog"})
    mongo, ica = FakeMongo([good, other]), ScriptedIca(objectives=[bad])
    job = await run_job(mongo, ica, [good["_id"], other["_id"]], "objectives")
    assert ica.calls == ["objectives", "objectives"]  # rejected, then accepted; none for `other`
    assert (job.status, job.succeeded, job.failed) == ("completed", 1, 1)
    assert (job.errors[0].stage, job.errors[0].status_code) == ("objectives", 422)
    assert await tags(mongo, other) is None


def test_objectives_backfill_and_single_question_route():
    a, b = question(section=TAGGABLE), question(section="Other")
    mongo, ica = FakeMongo([a, b]), ScriptedIca()
    client = client_for(mongo, ica)

    one = client.post("/examination/objectives/backfill", params={"section": TAGGABLE, "limit": 5})
    assert one.status_code == 202 and one.json()["total"] == 1
    job = client.get(f"/examination/jobs/{one.json()['job_id']}").json()
    assert (job["mode"], job["only_missing"], job["succeeded"]) == ("objectives", True, 1)

    # Only the untagged question is left; its section has no objectives yet.
    rest = client.post("/examination/objectives/backfill")
    assert rest.json()["total"] == 1
    job = client.get(f"/examination/jobs/{rest.json()['job_id']}").json()
    assert (job["status"], job["errors"][0]["status_code"]) == ("failed", 422)
    assert client.post("/examination/objectives/backfill", params={"only_missing": "false", "limit": 1}).json()["total"] == 1

    single = client.post(f"/examination/{a['_id']}/objectives")
    assert single.status_code == 200 and single.json()["objectives"] == ["claude-code-planning"]
    assert client.post(f"/examination/{b['_id']}/objectives").status_code == 422
    assert client.post(f"/examination/{ObjectId()}/objectives").status_code == 404


# --- story -------------------------------------------------------------------------


def in_chapter(**kw):
    """A question whose scenario paragraph the "pipeline" chapter claims."""
    q = question(**kw)
    q["prompt"]["question"] = (
        "SCENARIO: You are integrating Claude Code into your Continuous Integration pipeline. "
        "QUESTION: Which mode?"
    )
    return q


def outsider(**kw):
    """A question whose scenario paragraph no chapter claims."""
    q = question(**kw)
    q["prompt"]["question"] = "SCENARIO: You run a bakery. QUESTION: Which mode?"
    return q


async def story_record(mongo, doc):
    return await mongo.get_default_database()["question_story"].find_one(
        {"question_id": str(doc["_id"])}
    )


async def test_story_job_writes_a_record_and_skips_what_is_done(monkeypatch):
    q, plain = in_chapter(), outsider()
    mongo, ica = FakeMongo([q, plain]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"], plain["_id"]], "story")
    assert ica.calls == ["story"]  # none for the question of an unknown scenario
    assert (job.mode, job.succeeded, job.failed) == ("story", 1, 1)
    assert (job.errors[0].stage, job.errors[0].status_code) == ("story", 422)
    record = await story_record(mongo, q)
    assert (record["scenario"], record["stage"], record["origin"]) == ("pipeline", "build", "ica")
    assert (record["order"], record["text_hash"]) == (STORY_ORDER, text_hash(q))
    assert record["beat"] and record["ica"]["total_tokens"] == 7
    assert await story_record(mongo, plain) is None

    job = await run_job(mongo, ica, [q["_id"]], "story", only_missing=True)
    assert (job.skipped, job.succeeded) == (1, 0) and ica.calls == ["story"]

    # A changed prompt: redone, unless the record is a hand-written seed one.
    from claude_exam.api import story

    db = mongo.get_default_database()
    monkeypatch.setattr(story, "_SYSTEM_MESSAGE", story._SYSTEM_MESSAGE + "\nNew rule.")
    job = await run_job(mongo, ica, [q["_id"]], "story", only_missing=True)
    assert (job.skipped, job.succeeded) == (0, 1) and ica.calls == ["story"] * 2
    monkeypatch.setattr(story, "_SYSTEM_MESSAGE", story._SYSTEM_MESSAGE + "\nAnother rule.")
    await db["question_story"].update_one(
        {"question_id": str(q["_id"])}, {"$set": {"origin": "seed"}}
    )
    job = await run_job(mongo, ica, [q["_id"]], "story", only_missing=True)
    assert job.skipped == 1 and ica.calls == ["story"] * 2


async def test_story_job_retries_a_rejected_reply():
    q = in_chapter()
    bad = reply({**story_reply(), "stage": "finale"})
    mongo, ica = FakeMongo([q]), ScriptedIca(story=[bad])
    job = await run_job(mongo, ica, [q["_id"]], "story")
    assert ica.calls == ["story", "story"] and job.succeeded == 1


def test_story_backfill_and_single_question_route():
    a, b = in_chapter(), outsider()
    mongo, ica = FakeMongo([a, b]), ScriptedIca()
    client = client_for(mongo, ica)

    first = client.post("/examination/story/backfill")
    assert first.status_code == 202 and first.json()["total"] == 2
    job = client.get(f"/examination/jobs/{first.json()['job_id']}").json()
    assert (job["mode"], job["succeeded"], job["failed"]) == ("story", 1, 1)
    assert client.post("/examination/story/backfill").json()["total"] == 1  # only `b` is left
    assert client.post("/examination/story/backfill", params={"only_missing": "false", "limit": 1}).json()["total"] == 1

    single = client.post(f"/examination/{a['_id']}/story")
    assert single.status_code == 200 and single.json()["stage"] == "build"
    assert client.post(f"/examination/{b['_id']}/story").status_code == 422
    assert client.post(f"/examination/{ObjectId()}/story").status_code == 404


# --- routes ------------------------------------------------------------------------


def client_for(mongo, ica, lock=None):
    db = mongo.get_default_database()
    lock = lock or asyncio.Semaphore(1)
    app.dependency_overrides[get_async_collection] = lambda: db["examination"]
    app.dependency_overrides[get_async_simplified_collection] = lambda: db["examination_simplified"]
    app.dependency_overrides[get_async_objectives_collection] = lambda: db["question_objectives"]
    app.dependency_overrides[get_async_story_collection] = lambda: db["question_story"]
    app.dependency_overrides[get_async_decomposition_collection] = lambda: db["question_decomposition"]
    app.dependency_overrides[get_async_jobs_collection] = lambda: db["simplification_jobs"]
    app.dependency_overrides[get_async_mongo_client] = lambda: mongo
    app.dependency_overrides[get_ica_client] = lambda: ica
    app.dependency_overrides[get_simplification_lock] = lambda: lock
    return TestClient(app)  # background tasks run before the call returns


def upload(client, item=None, **params):
    item = {k: v for k, v in (item or in_chapter(section=TAGGABLE)).items() if k != "_id"}
    body = json.dumps({"results": [item]}).encode()
    return client.post(
        "/examination/import", params=params, files={"file": ("x.json", body, "application/json")}
    )


@pytest.mark.parametrize(
    ("params", "calls"),
    [
        ({}, ["objectives", "story", "decompose", "icons", "flow", "resolve", "rephrase", "check"]),
        ({"simplify": "false"}, []),
    ],
)
def test_import_runs_the_whole_pipeline_unless_disabled(params, calls):
    mongo, ica = FakeMongo(), ScriptedIca()
    client = client_for(mongo, ica)
    resp = upload(client, **params)
    assert resp.status_code == 200 and resp.json()["inserted"] == 1
    if calls:
        job = client.get(f"/examination/jobs/{resp.json()['job_id']}").json()
        assert (job["mode"], job["status"], job["succeeded"]) == ("full", "completed", 1)
    else:
        assert resp.json()["job_id"] is None
    assert ica.calls == calls


def test_icons_backfill_and_single_question_route():
    mongo = FakeMongo()
    ica = ScriptedIca(icons=[HTTPException(status_code=400, detail="x")])
    client = client_for(mongo, ica)
    upload(client)  # the icon step of the import job fails
    records = mongo.get_default_database()["question_decomposition"].docs
    assert records[0]["icons_ica"] is None

    resp = client.post("/examination/icons/backfill")
    assert resp.status_code == 202 and resp.json()["total"] == 1
    job = client.get(f"/examination/jobs/{resp.json()['job_id']}").json()
    assert (job["mode"], job["succeeded"]) == ("icons", 1)
    assert records[0]["entities"]["answers"][0]["icon"] == "write"
    assert client.post("/examination/icons/backfill").json() == {"job_id": None, "total": 0}

    one = client.post(f"/examination/{records[0]['question_id']}/icons", params={"max_tokens": 900})
    assert one.status_code == 200 and one.json()["entities"]["question_icon"] == "mode"
    assert ica.max_tokens[-1] == 900
    assert client.post(f"/examination/{ObjectId()}/icons").status_code == 404
    # Forcing picks them all again.
    assert client.post("/examination/icons/backfill", params={"only_missing": "false"}).json()["total"] == 1


def test_rephrase_route_returns_a_stored_rephrase_unless_forced():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client = client_for(mongo, ica)
    first = client.post(f"/questions/{q['_id']}/rephrase")
    assert first.status_code == 200 and first.json()["prompt"] == SIMPLE and ica.calls == ["rephrase"]
    assert client.post(f"/questions/{q['_id']}/rephrase").status_code == 200
    assert ica.calls == ["rephrase"]
    client.post(f"/questions/{q['_id']}/rephrase", params={"force": "true"})
    assert ica.calls == ["rephrase", "rephrase"]
    assert client.post(f"/questions/{ObjectId()}/rephrase").status_code == 404


async def test_the_icons_route_refuses_while_a_job_holds_the_lock():
    q = question()
    mongo, ica, lock = FakeMongo([q]), ScriptedIca(), asyncio.Semaphore(1)
    await lock.acquire()
    resp = client_for(mongo, ica, lock).post(f"/examination/{q['_id']}/icons")
    assert resp.status_code == 409 and ica.calls == []


# --- stored shapes ----------------------------------------------------------------------


def test_malformed_stored_entities_read_as_none():
    from claude_exam.api.game.domain import entities_of

    for bad in ({"entities": [{"name": "no id"}]}, {"entities": [], "answers": "x"}, {"question": 5}):
        assert entities_of({"entities": bad}, 2) == PromptEntities()
    with pytest.raises(ValueError):
        PromptEntities.model_validate({"entities": [{"name": "no id"}]})


# --- real MongoDB --------------------------------------------------------------------


@pytest.mark.skipif(not MONGO_URL, reason="set TEST_MONGO_URL to run against MongoDB")
async def test_pipeline_against_mongodb():
    """The decomposition, then the icon job updates its record in place; the conditional
    write matches the `updated_at` Mongo read back (stored to the millisecond)."""
    name = f"claude_exam_test_{secrets.token_hex(6)}"
    mongo = AsyncMongoClient(f"{MONGO_URL.rstrip('/')}/{name}", serverSelectionTimeoutMS=3000)
    try:
        db = mongo.get_default_database()
        q = question()
        await db["examination"].insert_one(q)
        ica = ScriptedIca(
            decompose=[reply(decomposition_with_plan_mode())],
            icons=[HTTPException(status_code=400, detail="x")],
        )
        jobs = db["simplification_jobs"]

        async def job(mode, only_missing=False):
            j = await jobs.insert_one(new_job("t", MODEL, 1, mode, only_missing))
            await run_simplification_job(
                mongo, ica, asyncio.Semaphore(1), j.inserted_id, [q["_id"]], MODEL, mode, only_missing
            )
            return await jobs.find_one({"_id": j.inserted_id})

        assert (await job("decompose"))["icons_failed"] == 1
        before = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
        assert (await job("icons", True))["succeeded"] == 1
        after = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
        assert after["entities"]["answers"][1]["entities"][0]["icon"] == "plan"
        assert (after["semantic"], after["flow"], after["created_at"]) == (
            before["semantic"],
            before["flow"],
            before["created_at"],
        )
        assert (await job("icons", True))["skipped"] == 1
        assert DecompositionRecord(**after).icons_ica is not None
    finally:
        await mongo.drop_database(name)
        await mongo.close()
