import asyncio
import logging
from datetime import datetime, timezone
from typing import NamedTuple

from bson import ObjectId
from fastapi import HTTPException
from pymongo import AsyncMongoClient

from claude_exam.api import pipeline_keys
from claude_exam.api.beat import beat_and_store
from claude_exam.api.check import check_and_store
from claude_exam.api.decompose import decompose_and_store, read_decomposition
from claude_exam.api.flow import flow_and_store, resolve_and_store
from claude_exam.api.game.domain import difficulty_of, story_of
from claude_exam.api.ica import IcaClient
from claude_exam.api.icons import assign_and_store
from claude_exam.api.objectives import tag_and_store
from claude_exam.api.pipeline_keys import Records
from claude_exam.api.schemas import IcaModelId, JobMode, JobStage
from claude_exam.api.rephrase import rephrase_and_store
from claude_exam.api.story import story_and_store

logger = logging.getLogger(__name__)

# Transient ICA failures worth retrying: rate limit, bad gateway / unparsable
# model output, timeout. Value = max retries.
RETRIES_BY_STATUS = {429: 3, 502: 2, 504: 2}
BACKOFF_SECONDS = 5.0
# A running job refreshes `heartbeat_at` before every ICA attempt. Must exceed
# the ICA call timeout plus the longest backoff sleep.
STALE_JOB_SECONDS = 15 * 60
# Questions a job works on at the same time, unless the caller says otherwise
# (`PIPELINE_CONCURRENCY` in the settings). The steps of one question stay in order.
DEFAULT_CONCURRENCY = 3
# The steps each job mode runs for a question, in order. In "full" the objectives come
# first because the story step orders a question by its difficulty. The decomposition
# (with its parts and entities, built in code) comes before what is built on it: the
# icons of its entities, the pre-answer flow and the resolution that continues it. The
# beat (key-free, the line before the question) needs the story's stage and comes after
# the decomposition, whose answer-only entities its leak check uses. The rephrase is
# optional and comes last, set in the question's chapter, stage and beat; the check
# tests it.
STAGES: dict[JobMode, tuple[JobStage, ...]] = {
    "full": (
        "objectives",
        "story",
        "decompose",
        "beat",
        "icons",
        "flow",
        "resolve",
        "rephrase",
        "check",
    ),
    "decompose": ("decompose", "icons", "flow", "resolve"),
    "icons": ("icons",),
    "flow": ("flow", "resolve"),
    "resolve": ("resolve",),
    "rephrase": ("rephrase", "check"),
    "check": ("check",),
    "objectives": ("objectives",),
    "story": ("story", "beat"),
    "beat": ("beat",),
}
# A step that needs what another step of the same job stores: when that one fails (or
# could not run), this one is not tried. Steps not listed here run whatever happened
# before them, so a question without objectives still gets its decomposition.
_NEEDS: dict[JobStage, JobStage] = {
    "beat": "story",
    "icons": "decompose",
    "flow": "decompose",
    "resolve": "flow",
    "check": "rephrase",
}
# A counter kept from before "full" existed: questions whose earlier steps were stored
# but whose icon step failed.
_LEGACY_COUNTER = {"icons": "icons_failed"}


class Stores(NamedTuple):
    examination: object
    simplified: object
    objectives: object
    story: object
    decomposition: object = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_job(
    source: str,
    model_id: IcaModelId,
    total: int,
    mode: JobMode,
    only_missing: bool = False,
) -> dict:
    """A queued job document for `simplification_jobs` (the layout `SimplificationJob` reads)."""
    now = _now()
    return {
        "source": source,
        "model_id": model_id,
        "mode": mode,
        "only_missing": only_missing,
        "status": "queued",
        "total": total,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "icons_failed": 0,
        "skipped": 0,
        "stages_failed": {},
        "errors": [],
        "created_at": now,
        "heartbeat_at": now,
        "started_at": None,
        "finished_at": None,
    }


async def _heartbeat(jobs, job_id: ObjectId) -> None:
    await jobs.update_one({"_id": job_id}, {"$set": {"heartbeat_at": _now()}})


async def _with_retry(operation, jobs, job_id: ObjectId) -> None:
    """Run `operation()`, retrying transient ICA failures with backoff."""
    attempt = 0
    while True:
        await _heartbeat(jobs, job_id)
        try:
            await operation()
            return
        except HTTPException as exc:
            if attempt >= RETRIES_BY_STATUS.get(exc.status_code, 0):
                raise
            await asyncio.sleep(BACKOFF_SECONDS * 2**attempt)
            attempt += 1


async def _story(stores: Stores, doc: dict) -> dict:
    """Where the question stands in the story today (chapter, stage, beat)."""
    record = await stores.story.find_one({"question_id": str(doc["_id"])})
    return story_of(doc, record)


async def _decomposition(stores: Stores, doc: dict):
    """The question's current decomposition (None without one), for the leak checks."""
    if stores.decomposition is None:
        return None
    record = await stores.decomposition.find_one({"question_id": str(doc["_id"])})
    return read_decomposition(record, doc)


async def load_records(stores: Stores, doc: dict) -> Records:
    """The question's stored records, as `pipeline_keys` reads them."""
    query = {"question_id": str(doc["_id"])}

    async def find(collection) -> dict | None:
        return await collection.find_one(query) if collection is not None else None

    return Records(
        objectives=await find(stores.objectives),
        story=await find(stores.story),
        decomposition=await find(stores.decomposition),
        simplified=await find(stores.simplified),
    )


async def is_current(stage: JobStage, stores: Stores, doc: dict) -> bool:
    """Whether the question already has what `stage` would store, written from today's
    inputs (`pipeline_keys.is_current`; a seed story record counts as current). A job
    with `only_missing` skips such a step, which is what makes running it again cheap."""
    return pipeline_keys.is_current(stage, doc, await load_records(stores, doc))


async def missing_stages(mode: JobMode, stores: Stores, doc: dict) -> list[JobStage]:
    """The steps of `mode` a question still needs (its records are read once)."""
    records = await load_records(stores, doc)
    return [s for s in STAGES[mode] if not pipeline_keys.is_current(s, doc, records)]


def _step(stage: JobStage, ica: IcaClient, stores: Stores, doc: dict, model_id: IcaModelId):
    """The operation of one pipeline step for one question."""

    async def objectives():
        await tag_and_store(ica, stores.objectives, doc, model_id)

    async def story():
        tags = await stores.objectives.find_one({"question_id": str(doc["_id"])})
        await story_and_store(
            ica, stores.story, doc, model_id, difficulty=difficulty_of(tags)
        )

    async def decomposed():
        await decompose_and_store(ica, stores.decomposition, doc, model_id)

    async def drawn():
        await flow_and_store(ica, stores.decomposition, doc, model_id)

    async def resolved():
        await resolve_and_store(ica, stores.decomposition, doc, model_id)

    async def icons():
        await assign_and_store(ica, stores.decomposition, doc, model_id)

    async def written():
        await beat_and_store(
            ica, stores.story, doc, model_id, decomposition=await _decomposition(stores, doc)
        )

    async def rephrased():
        await rephrase_and_store(
            ica,
            stores.simplified,
            doc,
            model_id,
            place=await _story(stores, doc),
            decomposition=await _decomposition(stores, doc),
        )

    async def checked():
        await check_and_store(
            ica, stores.simplified, doc, model_id, story=await _story(stores, doc)
        )

    return {
        "objectives": objectives,
        "story": story,
        "beat": written,
        "decompose": decomposed,
        "icons": icons,
        "rephrase": rephrased,
        "check": checked,
        "flow": drawn,
        "resolve": resolved,
    }[stage]


async def _process(
    object_id: ObjectId,
    ica: IcaClient,
    stores: Stores,
    jobs,
    job_id: ObjectId,
    model_id: IcaModelId,
    mode: JobMode,
    only_missing: bool,
    stages: tuple[JobStage, ...] | None = None,
) -> bool:
    """Run the steps of `mode` for one question and record the outcome on the job.
    Never raises. Returns whether anything was stored or was already there."""
    question_id = str(object_id)
    stages = stages or STAGES[mode]
    stored = skipped = 0
    blocked: set[JobStage] = set()
    errors: list[dict] = []

    def failed(stage: JobStage, status_code: int, detail: str) -> None:
        blocked.add(stage)
        errors.append(
            {
                "question_id": question_id,
                "stage": stage,
                "status_code": status_code,
                "detail": detail,
            }
        )

    stage: JobStage = stages[0]
    try:
        doc = await stores.examination.find_one({"_id": object_id})
        if doc is None:
            raise HTTPException(status_code=404, detail="question not found")
        for stage in stages:
            if _NEEDS.get(stage) in blocked:
                blocked.add(stage)
                continue
            try:
                if only_missing and await is_current(stage, stores, doc):
                    skipped += 1
                    continue
                await _with_retry(_step(stage, ica, stores, doc, model_id), jobs, job_id)
                stored += 1
            except HTTPException as exc:
                failed(stage, exc.status_code, str(exc.detail))
    except HTTPException as exc:
        failed(stage, exc.status_code, str(exc.detail))
    except Exception as exc:  # keep the batch going on any per-question failure
        logger.exception("%s failed for %s", stage, question_id)
        failed(stage, 500, f"{type(exc).__name__}: {exc}")

    update: dict = {"$inc": {"processed": 1}, "$set": {"heartbeat_at": _now()}}
    if errors:
        update["$inc"]["failed"] = 1
        update["$push"] = {"errors": {"$each": errors}}
        for error in errors:
            key = f"stages_failed.{error['stage']}"
            update["$inc"][key] = update["$inc"].get(key, 0) + 1
            legacy = _LEGACY_COUNTER.get(error["stage"])
            if legacy and stored:
                update["$inc"][legacy] = 1
    elif stored:
        update["$inc"]["succeeded"] = 1
    else:
        update["$inc"]["skipped"] = 1
    await jobs.update_one({"_id": job_id}, update)
    # Nothing left to do is not a failure.
    return bool(stored or skipped)


async def run_simplification_job(
    mongo: AsyncMongoClient,
    ica: IcaClient,
    lock: asyncio.Semaphore,
    job_id: ObjectId,
    question_ids: list[ObjectId],
    model_id: IcaModelId,
    mode: JobMode = "full",
    only_missing: bool = False,
    concurrency: int = 1,
    stages: tuple[JobStage, ...] | None = None,
) -> None:
    """
    Process the given examination questions (see `JobMode` and `STAGES`), `concurrency`
    of them at a time; the steps of one question run in order. Each step updates only
    its own fields or its own collection.

    With `only_missing`, a step is skipped for a question that has its result by the
    time the job reaches it (`is_current`), and a question whose every step was skipped
    counts as `skipped`. So a repeated backfill, one queued behind an import job, or a
    "full" job run again after a crash costs ICA calls only for what is left.

    The steps retry separately. A failure is recorded on the job with the step it comes
    from (`stage`) and the question goes on with the steps that do not need the failed
    one (`_NEEDS`); what was stored is kept. Progress is written after every question.
    Jobs in this process run one at a time (`lock`) so concurrent imports do not
    multiply ICA load. The job fails only when no question got anything stored.

    `stages` replaces the steps of `mode` (in the order given): how
    `scripts/run_pipeline.py` runs a chosen few steps for one question.
    """
    db = mongo.get_default_database()
    stores = Stores(
        db["examination"],
        db["examination_simplified"],
        db["question_objectives"],
        db["question_story"],
        db["question_decomposition"],
    )
    jobs = db["simplification_jobs"]

    status = "completed"
    try:
        async with lock:
            await jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "running",
                        "started_at": _now(),
                        "heartbeat_at": _now(),
                    }
                },
            )
            slots = asyncio.Semaphore(max(1, concurrency))

            async def one(object_id: ObjectId) -> bool:
                async with slots:
                    return await _process(
                        object_id,
                        ica,
                        stores,
                        jobs,
                        job_id,
                        model_id,
                        mode,
                        only_missing,
                        stages,
                    )

            results = await asyncio.gather(*(one(i) for i in question_ids))
        # Every question failing (bad key, ICA down) is a failed job, not a completed one.
        if question_ids and not any(results):
            status = "failed"
    except Exception:
        logger.exception("simplification job %s crashed", job_id)
        status = "failed"

    try:
        await jobs.update_one(
            {"_id": job_id}, {"$set": {"status": status, "finished_at": _now()}}
        )
    except Exception:
        # Mongo itself may be the reason we crashed; the stale-job sweep will catch it.
        logger.exception("could not record final status for job %s", job_id)
