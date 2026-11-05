import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

from bson import ObjectId
from fastapi import HTTPException
from pymongo import AsyncMongoClient

from claude_exam.api.entities import enrich_and_store
from claude_exam.api.ica import IcaClient
from claude_exam.api.schemas import IcaModelId
from claude_exam.api.simplify import simplify_and_store

logger = logging.getLogger(__name__)

# "simplify": simplify (and extract entities for) each question.
# "entities": only add entities to questions that are already simplified.
JobMode = Literal["simplify", "entities"]

# Transient ICA failures worth retrying: rate limit, bad gateway / unparsable
# model output, timeout. Value = max retries.
RETRIES_BY_STATUS = {429: 3, 502: 2, 504: 2}
BACKOFF_SECONDS = 5.0
# A running job refreshes `heartbeat_at` before every ICA attempt. Must exceed
# the ICA call timeout plus the longest backoff sleep.
STALE_JOB_SECONDS = 15 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


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


async def _simplify_with_retry(
    ica: IcaClient,
    simplified,
    doc: dict,
    model_id: IcaModelId,
    jobs,
    job_id: ObjectId,
    mode: JobMode = "simplify",
) -> None:
    if mode == "simplify":
        # Simplification and entity extraction retry separately: a rate limit on the
        # second call is retried, and if it persists it is reported on the job (the
        # simplified text is already stored by then).
        await _with_retry(
            lambda: simplify_and_store(
                ica, simplified, doc, model_id, with_entities=False
            ),
            jobs,
            job_id,
        )
    try:
        await _with_retry(
            lambda: enrich_and_store(ica, simplified, doc, model_id), jobs, job_id
        )
    except HTTPException as exc:
        if mode == "simplify":
            exc.detail = f"simplified, but entity extraction failed: {exc.detail}"
        raise


async def run_simplification_job(
    mongo: AsyncMongoClient,
    ica: IcaClient,
    lock: asyncio.Semaphore,
    job_id: ObjectId,
    question_ids: list[ObjectId],
    model_id: IcaModelId,
    mode: JobMode = "simplify",
) -> None:
    """
    Simplify the given examination questions one by one, replacing any stored
    simplified record (`mode="entities"` only backfills the entities of questions
    that are already simplified; its failures are reported, never swallowed).
    Per-question failures are recorded on the job and the
    batch continues. Progress is written after every question. Jobs in this
    process run one at a time (`lock`) so concurrent imports don't multiply
    ICA load.
    """
    db = mongo.get_default_database()
    examination = db["examination"]
    simplified = db["examination_simplified"]
    jobs = db["simplification_jobs"]

    succeeded = 0
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
                error: dict | None = None
                try:
                    doc = await examination.find_one({"_id": object_id})
                    if doc is None:
                        raise HTTPException(status_code=404, detail="question not found")
                    await _simplify_with_retry(
                        ica, simplified, doc, model_id, jobs, job_id, mode
                    )
                except HTTPException as exc:
                    error = {
                        "question_id": question_id,
                        "status_code": exc.status_code,
                        "detail": str(exc.detail),
                    }
                except Exception as exc:  # keep the batch going on any per-question failure
                    logger.exception("simplification failed for %s", question_id)
                    error = {
                        "question_id": question_id,
                        "status_code": 500,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }

                update: dict = {
                    "$inc": {"processed": 1},
                    "$set": {"heartbeat_at": _now()},
                }
                if error is None:
                    succeeded += 1
                    update["$inc"]["succeeded"] = 1
                else:
                    update["$inc"]["failed"] = 1
                    update["$push"] = {"errors": error}
                await jobs.update_one({"_id": job_id}, update)
        # Every question failing (bad key, ICA down) is a failed job, not a completed one.
        if question_ids and succeeded == 0:
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
