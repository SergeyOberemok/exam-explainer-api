"""The import -> simplify -> extract entities pipeline: job modes, the routes that start
them, and the standalone entity step updating an existing simplified record.

The last test runs the same pipeline against a real MongoDB (opt-in, like
`tests/test_game_mongo.py`): TEST_MONGO_URL=mongodb://127.0.0.1:27017.
"""

import asyncio
import copy
import json
import os
import secrets
from types import SimpleNamespace

import pytest
from bson import ObjectId
from fakes import matches, set_path
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient

from claude_exam.api import batch
from claude_exam.api.batch import new_job, run_simplification_job
from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_jobs_collection,
    get_async_mongo_client,
    get_async_simplified_collection,
    get_ica_client,
    get_simplification_lock,
)
from claude_exam.api.entities import MAX_ENTITIES, MAX_PER_ANSWER, _build_system_message
from claude_exam.api.main import app
from claude_exam.api.schemas import PromptEntities, SimplificationJob, SimplifiedQuestion

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
            doc[key] = doc.get(key, 0) + value
        for key, value in update.get("$push", {}).items():
            doc.setdefault(key, []).append(copy.deepcopy(value))

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
            "simplification_jobs": Coll(),
        }

    def get_default_database(self):
        return self.db


def reply(payload):
    return {
        "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 7},
    }


SIMPLE = {
    "question": "Which mode reads but never edits?",
    "answers": ["acceptEdits mode", "plan mode"],
    "feedbacks": ["no", "yes"],
    "explanation": "Plan mode blocks edits.",
}


def entities_reply(name="plan mode"):
    return {
        "entities": [{"id": "e1", "name": name, "kind": "object", "aliases": []}],
        "question_entities": ["e1"],
        "answers_entities": [[], ["e1"]],
    }


class ScriptedIca:
    """Answers the simplify and entity calls from separate queues (an Exception item is
    raised); defaults to good replies once a queue is empty. Records the call kinds."""

    def __init__(self, simplify=(), entities=()):
        self.queues = {"simplify": list(simplify), "entities": list(entities)}
        self.calls = []
        self.max_tokens = []

    async def chat_model(self, model_id, prompt, system=None, max_tokens=None):
        kind = "entities" if "visual entities" in (system or "") else "simplify"
        self.calls.append(kind)
        self.max_tokens.append(max_tokens)
        item = self.queues[kind].pop(0) if self.queues[kind] else None
        if isinstance(item, Exception):
            raise item
        return item or reply(SIMPLE if kind == "simplify" else entities_reply())


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
    db = mongo.get_default_database()
    return await db["examination_simplified"].find_one({"question_id": str(doc["_id"])})


# --- job modes ---------------------------------------------------------------------


async def test_simplify_mode_simplifies_then_adds_entities():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "simplify")
    assert ica.calls == ["simplify", "entities"]
    assert (job.mode, job.status, job.succeeded, job.failed) == ("simplify", "completed", 1, 0)
    record = await stored(mongo, q)
    assert record["entities"]["answers"][1]["entities"][0]["name"] == "plan mode"
    assert record["entities_ica"] is not None


async def test_simplify_only_mode_leaves_entities_for_later():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "simplify_only")
    assert ica.calls == ["simplify"]
    assert (job.status, job.succeeded) == ("completed", 1)
    record = await stored(mongo, q)
    assert record["prompt"]["question"] == SIMPLE["question"]
    assert record["entities_ica"] is None and record["entities"]["question"] == []


async def test_entities_failure_after_simplify_is_partial_not_a_failed_job():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(entities=[HTTPException(status_code=400, detail="boom")])
    job = await run_job(mongo, ica, [q["_id"]], "simplify")
    assert (job.status, job.succeeded, job.failed, job.entities_failed) == ("completed", 0, 1, 1)
    assert job.errors[0].stage == "entities" and job.errors[0].detail == "boom"
    assert (await stored(mongo, q))["prompt"]["question"] == SIMPLE["question"]


async def test_simplify_failure_is_reported_with_its_stage_and_fails_the_job():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(simplify=[HTTPException(status_code=400, detail="bad key")])
    job = await run_job(mongo, ica, [q["_id"]], "simplify")
    assert ica.calls == ["simplify"]  # no entity call without a simplified record
    assert (job.status, job.failed, job.entities_failed) == ("failed", 1, 0)
    assert job.errors[0].stage == "simplify"
    assert await stored(mongo, q) is None


async def test_entity_step_retries_transient_errors():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(entities=[HTTPException(status_code=429, detail="slow down")])
    job = await run_job(mongo, ica, [q["_id"]], "simplify")
    assert ica.calls == ["simplify", "entities", "entities"]
    assert job.succeeded == 1


async def test_entities_job_updates_existing_record_only():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "simplify_only")
    before = await stored(mongo, q)

    job = await run_job(mongo, ica, [q["_id"]], "entities", only_missing=True)
    after = await stored(mongo, q)
    assert (job.mode, job.status, job.succeeded, job.skipped) == ("entities", "completed", 1, 0)
    assert ica.calls == ["simplify", "entities"]
    assert after["entities"]["question"][0]["name"] == "plan mode"
    for unchanged in ("_id", "prompt", "ica", "created_at", "correct_response"):
        assert after[unchanged] == before[unchanged]
    assert after["updated_at"] >= before["updated_at"]
    assert len(mongo.get_default_database()["examination_simplified"].docs) == 1


async def test_entities_job_skips_records_that_got_entities_meanwhile():
    """only_missing is checked when the job reaches a question, not when it was queued."""
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "simplify")  # e.g. the import job it waited for
    ica.calls.clear()

    job = await run_job(mongo, ica, [q["_id"]], "entities", only_missing=True)
    assert ica.calls == []
    assert (job.status, job.succeeded, job.skipped, job.failed) == ("completed", 0, 1, 0)


async def test_entities_job_without_only_missing_re_extracts():
    q = question()
    mongo = FakeMongo([q])
    ica = ScriptedIca(entities=[reply(entities_reply("v1")), reply(entities_reply("v2"))])
    await run_job(mongo, ica, [q["_id"]], "simplify")
    job = await run_job(mongo, ica, [q["_id"]], "entities", only_missing=False)
    assert job.succeeded == 1
    assert (await stored(mongo, q))["entities"]["question"][0]["name"] == "v2"


async def test_entities_job_reports_a_question_that_is_not_simplified():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    job = await run_job(mongo, ica, [q["_id"]], "entities", only_missing=True)
    assert ica.calls == []
    assert (job.status, job.errors[0].stage, job.errors[0].status_code) == ("failed", "entities", 404)


# --- routes ------------------------------------------------------------------------


def client_for(mongo, ica, lock=None):
    db = mongo.get_default_database()
    lock = lock or asyncio.Semaphore(1)
    app.dependency_overrides[get_async_collection] = lambda: db["examination"]
    app.dependency_overrides[get_async_simplified_collection] = lambda: db["examination_simplified"]
    app.dependency_overrides[get_async_jobs_collection] = lambda: db["simplification_jobs"]
    app.dependency_overrides[get_async_mongo_client] = lambda: mongo
    app.dependency_overrides[get_ica_client] = lambda: ica
    app.dependency_overrides[get_simplification_lock] = lambda: lock
    return TestClient(app)  # background tasks run before the call returns


def upload(client, **params):
    item = {k: v for k, v in question().items() if k != "_id"}
    body = json.dumps({"results": [item]}).encode()
    return client.post(
        "/examination/import", params=params, files={"file": ("x.json", body, "application/json")}
    )


@pytest.mark.parametrize(
    ("params", "mode", "calls"),
    [({}, "simplify", ["simplify", "entities"]), ({"entities": "false"}, "simplify_only", ["simplify"])],
)
def test_import_runs_entities_unless_disabled(params, mode, calls):
    mongo, ica = FakeMongo(), ScriptedIca()
    client = client_for(mongo, ica)
    resp = upload(client, **params)
    assert resp.status_code == 200
    job = client.get(f"/examination/jobs/{resp.json()['job_id']}").json()
    assert (job["mode"], job["status"], job["succeeded"]) == (mode, "completed", 1)
    assert ica.calls == calls


def test_import_without_entities_then_backfill_adds_them():
    mongo, ica = FakeMongo(), ScriptedIca()
    client = client_for(mongo, ica)
    upload(client, entities="false")
    resp = client.post("/examination/entities/backfill")
    assert resp.status_code == 202 and resp.json()["total"] == 1
    job = client.get(f"/examination/jobs/{resp.json()['job_id']}").json()
    assert (job["mode"], job["only_missing"], job["succeeded"]) == ("entities", True, 1)
    record = mongo.get_default_database()["examination_simplified"].docs[0]
    assert record["entities_ica"] is not None

    # Nothing left: a second backfill starts no job.
    assert client.post("/examination/entities/backfill").json() == {"job_id": None, "total": 0}
    # Forcing re-extraction picks it up again.
    assert client.post("/examination/entities/backfill", params={"only_missing": "false"}).json()["total"] == 1


def test_simplify_route_can_skip_entities():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    resp = client_for(mongo, ica).post(f"/questions/{q['_id']}/simplify", params={"entities": "false"})
    assert resp.status_code == 200 and ica.calls == ["simplify"]
    assert resp.json()["entities_ica"] is None


def test_entities_route_updates_existing_record_and_passes_max_tokens():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client = client_for(mongo, ica)
    client.post(f"/questions/{q['_id']}/simplify", params={"entities": "false"})
    resp = client.post(f"/examination/{q['_id']}/entities", params={"max_tokens": 900})
    assert resp.status_code == 200
    assert resp.json()["entities"]["question"][0]["name"] == "plan mode"
    assert ica.max_tokens[-1] == 900


def test_entities_route_needs_a_simplified_record():
    q = question()
    resp = client_for(FakeMongo([q]), ScriptedIca()).post(f"/examination/{q['_id']}/entities")
    assert resp.status_code == 404


async def test_entities_route_refuses_while_a_job_holds_the_lock():
    q = question()
    mongo, ica, lock = FakeMongo([q]), ScriptedIca(), asyncio.Semaphore(1)
    await lock.acquire()
    resp = client_for(mongo, ica, lock).post(f"/examination/{q['_id']}/entities")
    assert resp.status_code == 409 and ica.calls == []


# --- stored shapes and prompt --------------------------------------------------------


def test_malformed_stored_entities_read_as_none():
    record = {
        "question_id": "x",
        "section": "",
        "assessment_type": "",
        "source": "",
        "correct_response": ["a"],
        "related_lectures": [],
        "prompt": SIMPLE,
        "ica": {"model_id": MODEL, "finish_reason": "stop", "total_tokens": 1},
        "created_at": "2026-11-06T00:00:00Z",
        "updated_at": "2026-11-06T00:00:00Z",
    }
    for bad in ({"entities": [{"name": "no id"}]}, {"entities": [], "answers": "x"}, {"question": 5}):
        assert SimplifiedQuestion(**record, entities=bad).entities == PromptEntities()
    with pytest.raises(ValueError):
        PromptEntities.model_validate({"entities": [{"name": "no id"}]})


def test_prompt_limits_come_from_the_constants():
    message = _build_system_message(3)
    assert f"at most {MAX_ENTITIES} entities, at most {MAX_PER_ANSWER} per answer" in message
    assert "{{" not in message


# --- real MongoDB --------------------------------------------------------------------


@pytest.mark.skipif(not MONGO_URL, reason="set TEST_MONGO_URL to run against MongoDB")
async def test_pipeline_against_mongodb():
    """Simplify only, then the entity job updates the record in place; the conditional
    write matches the `updated_at` Mongo read back (stored to the millisecond)."""
    name = f"claude_exam_test_{secrets.token_hex(6)}"
    mongo = AsyncMongoClient(f"{MONGO_URL.rstrip('/')}/{name}", serverSelectionTimeoutMS=3000)
    try:
        db = mongo.get_default_database()
        q = question()
        await db["examination"].insert_one(q)
        ica = ScriptedIca()
        jobs = db["simplification_jobs"]

        async def job(mode, only_missing=False):
            j = await jobs.insert_one(new_job("t", MODEL, 1, mode, only_missing))
            await run_simplification_job(
                mongo, ica, asyncio.Semaphore(1), j.inserted_id, [q["_id"]], MODEL, mode, only_missing
            )
            return await jobs.find_one({"_id": j.inserted_id})

        assert (await job("simplify_only"))["succeeded"] == 1
        before = await db["examination_simplified"].find_one({"question_id": str(q["_id"])})
        assert (await job("entities", True))["succeeded"] == 1
        after = await db["examination_simplified"].find_one({"question_id": str(q["_id"])})
        assert after["entities"]["question"][0]["name"] == "plan mode"
        assert (after["prompt"], after["ica"], after["created_at"]) == (
            before["prompt"],
            before["ica"],
            before["created_at"],
        )
        assert (await job("entities", True))["skipped"] == 1
        assert await db["examination_simplified"].count_documents({}) == 1
        assert ica.calls == ["simplify", "entities"]
    finally:
        await mongo.drop_database(name)
        await mongo.close()
