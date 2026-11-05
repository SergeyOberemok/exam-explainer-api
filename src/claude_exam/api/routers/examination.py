import json
from datetime import datetime, timezone
from typing import Annotated

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    UploadFile,
)

from claude_exam.api.batch import run_simplification_job
from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncJobsCollectionDep,
    AsyncMongoClientDep,
    AsyncSimplifiedCollectionDep,
    IcaClientDep,
    QuestionObjectIdDep,
    SimplificationLockDep,
)
from claude_exam.api.entities import enrich_and_store
from claude_exam.api.normalization import normalize_result
from claude_exam.api.schemas import (
    EntitiesBackfill,
    IcaModelId,
    ImportResult,
    SimplificationJob,
    SimplifiedQuestion,
)
from claude_exam.api.security import require_api_key

# Study/admin routes: they need the API key when one is configured.
router = APIRouter(
    prefix="/examination", tags=["examination"], dependencies=[Depends(require_api_key)]
)


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


@router.post("/{question_id}/entities")
async def extract_question_entities(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    ica: IcaClientDep,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for extraction")
    ] = "claude-haiku-5-5",
) -> SimplifiedQuestion:
    """
    (Re)extract the visual entities of one already simplified question through ICA
    and store them on its simplified record; the simplified text is not touched.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    return await enrich_and_store(ica, simplified, doc, model_id)


@router.post("/entities/backfill", status_code=202)
async def backfill_entities(
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    jobs: AsyncJobsCollectionDep,
    mongo: AsyncMongoClientDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    background_tasks: BackgroundTasks,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for extraction")
    ] = "claude-haiku-5-5",
    only_missing: Annotated[
        bool, Query(description="Skip records that already have entities")
    ] = True,
) -> EntitiesBackfill:
    """
    Start a background job that extracts entities for already simplified questions
    (by default only those without any); track it via `GET /examination/jobs/{job_id}`.
    """
    # `entities_ica` is set only by a successful extraction, so a question that really
    # has no entities is not picked again (None also matches a missing field).
    query: dict = {"entities_ica": None} if only_missing else {}
    question_ids = [
        ObjectId(doc["question_id"])
        async for doc in simplified.find(query, {"question_id": 1})
        if ObjectId.is_valid(doc.get("question_id", ""))
    ]
    if not question_ids:
        return EntitiesBackfill(job_id=None, total=0)

    now = datetime.now(timezone.utc)
    job = await jobs.insert_one(
        {
            "source": "entities-backfill",
            "model_id": model_id,
            "status": "queued",
            "total": len(question_ids),
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
            "created_at": now,
            "heartbeat_at": now,
            "started_at": None,
            "finished_at": None,
        }
    )
    background_tasks.add_task(
        run_simplification_job,
        mongo,
        ica,
        lock,
        job.inserted_id,
        question_ids,
        model_id,
        "entities",
    )
    return EntitiesBackfill(job_id=str(job.inserted_id), total=len(question_ids))


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
