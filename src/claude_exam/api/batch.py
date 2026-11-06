import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import HTTPException
from pymongo import AsyncMongoClient

from claude_exam.api.entities import enrich_and_store
from claude_exam.api.ica import IcaClient
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


async def _has_entities(simplified, question_id: str) -> bool:
    """Whether the record already has entities from a successful extraction."""
    found = await simplified.find_one(
        {"question_id": question_id, "entities_ica": {"$ne": None}}, {"_id": 1}
    )
    return found is not None


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
    Process the given examination questions one by one (see `JobMode`). Simplifying
    replaces any stored simplified record; extracting entities updates only the entity
    fields of the existing record. With `mode="entities"` and `only_missing`, a question
    that has entities by the time the job reaches it is skipped (`skipped`), so a
    repeated backfill, or one queued behind an import job, costs no ICA calls.

    Simplification and entity extraction retry separately. Per-question failures are
    recorded on the job with the step they come from (`stage`) and the batch continues;
    a failed extraction after a stored simplification also counts in `entities_failed`
    (the text is kept; backfill the entities later). Progress is written after every
    question. Jobs in this process run one at a time (`lock`) so concurrent imports
    do not multiply ICA load. The job fails only when no question got anything stored.
    """
    db = mongo.get_default_database()
    examination = db["examination"]
    simplified = db["examination_simplified"]
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
                stage: JobStage = "entities" if mode == "entities" else "simplify"
                text_stored = skipped = False
                error: dict | None = None
                try:
                    if (
                        mode == "entities"
                        and only_missing
                        and await _has_entities(simplified, question_id)
                    ):
                        skipped = True
                    else:
                        doc = await examination.find_one({"_id": object_id})
                        if doc is None:
                            raise HTTPException(status_code=404, detail="question not found")
                        if mode != "entities":
                            await _with_retry(
                                lambda: simplify_and_store(
                                    ica, simplified, doc, model_id, with_entities=False
                                ),
                                jobs,
                                job_id,
                            )
                            text_stored = True
                        if mode != "simplify_only":
                            stage = "entities"
                            await _with_retry(
                                lambda: enrich_and_store(ica, simplified, doc, model_id),
                                jobs,
                                job_id,
                            )
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
                    if text_stored:
                        update["$inc"]["entities_failed"] = 1
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
