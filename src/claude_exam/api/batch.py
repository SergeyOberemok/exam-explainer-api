import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import HTTPException
from pymongo import AsyncMongoClient

from claude_exam.api.entities import enrich_and_store
from claude_exam.api.ica import IcaClient
from claude_exam.api.icons import assign_and_store
from claude_exam.api.objectives import current_query, tag_and_store
from claude_exam.api.schemas import IcaModelId, JobMode, JobStage
from claude_exam.api.simplify import simplify_and_store

logger = logging.getLogger(__name__)

# Transient ICA failures worth retrying: rate limit, bad gateway / unparsable
# model output, timeout. Value = max retries.
RETRIES_BY_STATUS = {429: 3, 502: 2, 504: 2}
BACKOFF_SECONDS = 5.0
# A running job refreshes `heartbeat_at` before every ICA attempt. Must exceed
# the ICA call timeout plus the longest backoff sleep.
STALE_JOB_SECONDS = 15 * 60
# The steps each job mode runs for a question, in order.
STAGES: dict[JobMode, tuple[JobStage, ...]] = {
    "simplify": ("simplify", "entities", "icons"),
    "simplify_only": ("simplify",),
    "entities": ("entities", "icons"),
    "icons": ("icons",),
    "objectives": ("objectives",),
}
# Set on the simplified record only by a successful run of that step.
_DONE_FIELD = {"entities": "entities_ica", "icons": "icons_ica"}


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
        "entities_failed": 0,
        "icons_failed": 0,
        "skipped": 0,
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


async def _already_done(simplified, objectives, question_id: str, mode: JobMode) -> bool:
    """Whether the question already has what an "entities", "icons" or "objectives" job
    would add (objectives: tagged with the current catalog and prompt)."""
    if mode == "objectives":
        return await objectives.find_one(current_query(question_id), {"_id": 1}) is not None
    found = await simplified.find_one(
        {"question_id": question_id, _DONE_FIELD[mode]: {"$ne": None}}, {"_id": 1}
    )
    return found is not None


def _step(
    stage: JobStage, ica: IcaClient, simplified, objectives, doc: dict, model_id: IcaModelId
):
    """The operation of one pipeline step for one question."""
    if stage == "objectives":
        return lambda: tag_and_store(ica, objectives, doc, model_id)
    if stage == "simplify":
        return lambda: simplify_and_store(ica, simplified, doc, model_id, with_entities=False)
    if stage == "entities":
        return lambda: enrich_and_store(ica, simplified, doc, model_id)
    return lambda: assign_and_store(ica, simplified, str(doc["_id"]), model_id)


async def run_simplification_job(
    mongo: AsyncMongoClient,
    ica: IcaClient,
    lock: asyncio.Semaphore,
    job_id: ObjectId,
    question_ids: list[ObjectId],
    model_id: IcaModelId,
    mode: JobMode = "simplify",
    only_missing: bool = False,
) -> None:
    """
    Process the given examination questions one by one (see `JobMode` and `STAGES`).
    Simplifying replaces any stored simplified record; extracting entities and picking
    icons each update only their own fields of the existing record. With
    `mode="entities"`, `"icons"` or `"objectives"` and `only_missing`, a question that
    has them by the time the job reaches it is skipped (`skipped`), so a repeated
    backfill, or one queued behind an import job, costs no ICA calls.

    The steps retry separately. Per-question failures are recorded on the job with the
    step they come from (`stage`) and the batch continues; a step failing after an
    earlier one was stored also counts in `entities_failed` / `icons_failed` (what was
    stored is kept; backfill the rest later). Progress is written after every
    question. Jobs in this process run one at a time (`lock`) so concurrent imports
    do not multiply ICA load. The job fails only when no question got anything stored.
    """
    db = mongo.get_default_database()
    examination = db["examination"]
    simplified = db["examination_simplified"]
    objectives = db["question_objectives"]
    jobs = db["simplification_jobs"]

    stored_any = False
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
            for object_id in question_ids:
                question_id = str(object_id)
                stage: JobStage = STAGES[mode][0]
                steps_stored = 0
                skipped = False
                error: dict | None = None
                try:
                    if (
                        (mode in _DONE_FIELD or mode == "objectives")
                        and only_missing
                        and await _already_done(simplified, objectives, question_id, mode)
                    ):
                        skipped = True
                    else:
                        doc = await examination.find_one({"_id": object_id})
                        if doc is None:
                            raise HTTPException(status_code=404, detail="question not found")
                        for stage in STAGES[mode]:
                            await _with_retry(
                                _step(stage, ica, simplified, objectives, doc, model_id),
                                jobs,
                                job_id,
                            )
                            steps_stored += 1
                except HTTPException as exc:
                    error = {
                        "question_id": question_id,
                        "stage": stage,
                        "status_code": exc.status_code,
                        "detail": str(exc.detail),
                    }
                except Exception as exc:  # keep the batch going on any per-question failure
                    logger.exception("%s failed for %s", stage, question_id)
                    error = {
                        "question_id": question_id,
                        "stage": stage,
                        "status_code": 500,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }

                update: dict = {
                    "$inc": {"processed": 1},
                    "$set": {"heartbeat_at": _now()},
                }
                if skipped:
                    update["$inc"]["skipped"] = 1
                    stored_any = True  # nothing left to do is not a failure
                elif error is None:
                    update["$inc"]["succeeded"] = 1
                    stored_any = True
                else:
                    update["$inc"]["failed"] = 1
                    update["$push"] = {"errors": error}
                    if steps_stored:
                        update["$inc"][f"{stage}_failed"] = 1
                        stored_any = True
                await jobs.update_one({"_id": job_id}, update)
        # Every question failing (bad key, ICA down) is a failed job, not a completed one.
        if question_ids and not stored_any:
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
