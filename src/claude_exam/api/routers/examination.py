import json
from datetime import datetime, timezone
from typing import Annotated

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile

from claude_exam.api.batch import run_simplification_job
from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncJobsCollectionDep,
    AsyncMongoClientDep,
    IcaClientDep,
    SimplificationLockDep,
)
from claude_exam.api.normalization import normalize_result
from claude_exam.api.schemas import IcaModelId, ImportResult, SimplificationJob

router = APIRouter(prefix="/examination", tags=["examination"])


@router.post("/import")
async def import_practice_exam(
    file: UploadFile,
    collection: AsyncCollectionDep,
    jobs: AsyncJobsCollectionDep,
    mongo: AsyncMongoClientDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    background_tasks: BackgroundTasks,
    simplify: Annotated[
        bool, Query(description="Start a background simplification job after import")
    ] = True,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for simplification")
    ] = "claude-haiku-5-5",
) -> ImportResult:
    """
    Import a practice-exam JSON file. Unless `simplify=false`, a background job
    then simplifies each newly imported question through ICA, replacing any
    stored simplified record; track it via `GET /examination/jobs/{job_id}`.
    """
    try:
        data = json.loads(await file.read())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="expected a top-level JSON object")

    results = data.get("results")
    if not isinstance(results, list):
        raise HTTPException(status_code=400, detail="expected a top-level 'results' array")
    if not all(isinstance(item, dict) for item in results):
        raise HTTPException(status_code=400, detail="each item in 'results' must be an object")

    source = file.filename or "unknown"
    documents = [normalize_result(item, source) for item in results]

    inserted_ids: list[ObjectId] = []
    if documents:
        inserted_ids = (await collection.insert_many(documents)).inserted_ids

    job_id: str | None = None
    if simplify and inserted_ids:
        job = await jobs.insert_one(
            {
                "source": source,
                "model_id": model_id,
                "status": "queued",
                "total": len(inserted_ids),
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "errors": [],
                "created_at": datetime.now(timezone.utc),
                "heartbeat_at": datetime.now(timezone.utc),
                "started_at": None,
                "finished_at": None,
            }
        )
        job_id = str(job.inserted_id)
        background_tasks.add_task(
            run_simplification_job,
            mongo,
            ica,
            lock,
            job.inserted_id,
            inserted_ids,
            model_id,
        )

    return ImportResult(source=source, inserted=len(inserted_ids), job_id=job_id)


@router.get("/jobs/{job_id}")
async def get_simplification_job(
    job_id: str, jobs: AsyncJobsCollectionDep
) -> SimplificationJob:
    try:
        object_id = ObjectId(job_id)
    except InvalidId as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    doc = await jobs.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="job not found")
    doc["id"] = str(doc.pop("_id"))
    return SimplificationJob(**doc)
