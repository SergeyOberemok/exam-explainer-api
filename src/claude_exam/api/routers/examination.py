import json
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

from claude_exam.api.batch import new_job, run_simplification_job
from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncJobsCollectionDep,
    AsyncMongoClientDep,
    AsyncObjectivesCollectionDep,
    AsyncSimplifiedCollectionDep,
    IcaClientDep,
    QuestionObjectIdDep,
    SimplificationLockDep,
)
from claude_exam.api.entities import enrich_and_store
from claude_exam.api.icons import assign_and_store
from claude_exam.api.normalization import normalize_result
from claude_exam.api.objectives import current_query, tag_and_store
from claude_exam.api.schemas import (
    EntitiesBackfill,
    IcaModelId,
    ImportResult,
    QuestionObjectives,
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
    entities: Annotated[
        bool,
        Query(
            description="Also extract visual entities after simplifying each question "
            "(false: simplify only; add them later with POST /examination/entities/backfill)"
        ),
    ] = True,
) -> ImportResult:
    """
    Import a practice-exam JSON file. Unless `simplify=false`, a background job
    then simplifies each newly imported question through ICA, replacing any
    stored simplified record, and (unless `entities=false`) extracts its visual
    entities onto that record; track it via `GET /examination/jobs/{job_id}`.
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
        mode = "simplify" if entities else "simplify_only"
        job = await jobs.insert_one(new_job(source, model_id, len(inserted_ids), mode))
        job_id = str(job.inserted_id)
        background_tasks.add_task(
            run_simplification_job,
            mongo,
            ica,
            lock,
            job.inserted_id,
            inserted_ids,
            model_id,
            mode,
        )

    return ImportResult(source=source, inserted=len(inserted_ids), job_id=job_id)


@router.post("/{question_id}/entities")
async def extract_question_entities(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for extraction")
    ] = "claude-haiku-5-5",
    max_tokens: Annotated[
        int | None, Query(ge=1, description="Token limit for the extraction reply")
    ] = None,
) -> SimplifiedQuestion:
    """
    (Re)extract the visual entities of one already simplified question through ICA
    and store them on its simplified record; the simplified text is not touched.
    The new entities have no icons: call `POST /examination/{id}/icons` afterwards.
    Shares the job lock so it never adds ICA load on top of a running job: while a
    job runs it answers 409 (retry once the job has finished) instead of waiting.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a simplification job is running; retry when it has finished",
        )
    async with lock:
        return await enrich_and_store(ica, simplified, doc, model_id, max_tokens)


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
    (by default only those without any; the job checks again when it reaches each
    question, so repeated or overlapping backfills skip what is already done);
    track it via `GET /examination/jobs/{job_id}`.
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

    job = await jobs.insert_one(
        new_job("entities-backfill", model_id, len(question_ids), "entities", only_missing)
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
        only_missing,
    )
    return EntitiesBackfill(job_id=str(job.inserted_id), total=len(question_ids))


@router.post("/{question_id}/icons")
async def pick_question_icons(
    question_id: QuestionObjectIdDep,
    simplified: AsyncSimplifiedCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for picking")
    ] = "claude-haiku-5-5",
    max_tokens: Annotated[
        int | None, Query(ge=1, description="Token limit for the reply")
    ] = None,
) -> SimplifiedQuestion:
    """
    (Re)pick the icons of one already simplified question through ICA: one key from
    the icon catalog for each of its entities, for the question and for each answer.
    Only the icon fields change. Answers 409 while a job runs, like the entity route.
    """
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a simplification job is running; retry when it has finished",
        )
    async with lock:
        return await assign_and_store(ica, simplified, str(question_id), model_id, max_tokens)


@router.post("/icons/backfill", status_code=202)
async def backfill_icons(
    simplified: AsyncSimplifiedCollectionDep,
    jobs: AsyncJobsCollectionDep,
    mongo: AsyncMongoClientDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    background_tasks: BackgroundTasks,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for picking")
    ] = "claude-haiku-5-5",
    only_missing: Annotated[
        bool, Query(description="Skip records that already have icons")
    ] = True,
) -> EntitiesBackfill:
    """
    Start a background job that picks icons for already simplified questions (by
    default only those without any; pass `only_missing=false` after changing the icon
    catalog to pick all again); track it via `GET /examination/jobs/{job_id}`.
    """
    query: dict = {"icons_ica": None} if only_missing else {}
    question_ids = [
        ObjectId(doc["question_id"])
        async for doc in simplified.find(query, {"question_id": 1})
        if ObjectId.is_valid(doc.get("question_id", ""))
    ]
    if not question_ids:
        return EntitiesBackfill(job_id=None, total=0)

    job = await jobs.insert_one(
        new_job("icons-backfill", model_id, len(question_ids), "icons", only_missing)
    )
    background_tasks.add_task(
        run_simplification_job,
        mongo,
        ica,
        lock,
        job.inserted_id,
        question_ids,
        model_id,
        "icons",
        only_missing,
    )
    return EntitiesBackfill(job_id=str(job.inserted_id), total=len(question_ids))


@router.post("/{question_id}/objectives")
async def tag_question_objectives(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for tagging")
    ] = "claude-haiku-5-5",
    max_tokens: Annotated[
        int | None, Query(ge=1, description="Token limit for the reply")
    ] = None,
) -> QuestionObjectives:
    """
    (Re)tag one question through ICA with its learning objectives, level, difficulty
    and what each wrong answer reveals, replacing its `question_objectives` record.
    422 when the catalog has no objectives for the question's section. Answers 409
    while a job runs, like the entity route.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a simplification job is running; retry when it has finished",
        )
    async with lock:
        return await tag_and_store(ica, objectives, doc, model_id, max_tokens)


@router.post("/objectives/backfill", status_code=202)
async def backfill_objectives(
    collection: AsyncCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    jobs: AsyncJobsCollectionDep,
    mongo: AsyncMongoClientDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    background_tasks: BackgroundTasks,
    model_id: Annotated[
        IcaModelId, Query(description="ICA model used for tagging")
    ] = "claude-haiku-5-5",
    only_missing: Annotated[
        bool,
        Query(description="Skip questions already tagged with the current catalog and prompt"),
    ] = True,
    section: Annotated[
        str | None, Query(max_length=200, description="Only questions of this section")
    ] = None,
    limit: Annotated[
        int | None, Query(ge=1, description="Tag at most this many questions")
    ] = None,
) -> EntitiesBackfill:
    """
    Start a background job that tags questions with learning objectives (by default
    only those not tagged yet, or tagged before the objective catalog or the prompt
    last changed, so running it again after editing the catalog redoes everything);
    track it via `GET /examination/jobs/{job_id}`.
    """
    question_ids: list[ObjectId] = []
    async for doc in collection.find({"section": section} if section else {}, {"_id": 1}):
        if only_missing and await objectives.find_one(current_query(str(doc["_id"])), {"_id": 1}):
            continue
        question_ids.append(doc["_id"])
        if limit is not None and len(question_ids) >= limit:
            break
    if not question_ids:
        return EntitiesBackfill(job_id=None, total=0)

    job = await jobs.insert_one(
        new_job("objectives-backfill", model_id, len(question_ids), "objectives", only_missing)
    )
    background_tasks.add_task(
        run_simplification_job,
        mongo,
        ica,
        lock,
        job.inserted_id,
        question_ids,
        model_id,
        "objectives",
        only_missing,
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
