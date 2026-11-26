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
from pydantic import ValidationError

from claude_exam.api.batch import (
    Stores,
    is_current,
    missing_stages,
    new_job,
    run_simplification_job,
)
from claude_exam.api.check import check_and_store
from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncDecompositionCollectionDep,
    ExaminationDocDep,
    AsyncJobsCollectionDep,
    AsyncMongoClientDep,
    AsyncObjectivesCollectionDep,
    AsyncSimplifiedCollectionDep,
    AsyncStoryCollectionDep,
    IcaClientDep,
    PipelineConcurrencyDep,
    QuestionObjectIdDep,
    SimplificationLockDep,
)
from claude_exam.api.decompose import decompose_and_store, read_decomposition
from claude_exam.api.flow import flow_and_store, resolve_and_store
from claude_exam.api import hold
from claude_exam.api.game.domain import story_of
from claude_exam.api.icons import assign_and_store
from claude_exam.api.ingest import plan_import
from claude_exam.api.normalization import normalize_result
from claude_exam.api.objectives import tag_and_store
from claude_exam.api.schemas import (
    DecompositionRecord,
    EntitiesBackfill,
    IcaModelId,
    ImportResult,
    JobMode,
    QuestionObjectives,
    QuestionStory,
    SimplificationJob,
    SimplifiedQuestion,
)
from claude_exam.api.security import require_api_key
from claude_exam.api.beat import beat_and_store
from claude_exam.api.story import story_and_store

# Study/admin routes: they need the API key when one is configured.
router = APIRouter(
    prefix="/examination", tags=["examination"], dependencies=[Depends(require_api_key)]
)

ModelIdQuery = Annotated[
    IcaModelId, Query(description="ICA model the job or call uses")
]
MaxTokensQuery = Annotated[
    int | None, Query(ge=1, description="Token limit for the reply")
]


class _JobStarter:
    """What every route that starts a background job needs, as one dependency."""

    def __init__(
        self,
        jobs: AsyncJobsCollectionDep,
        mongo: AsyncMongoClientDep,
        ica: IcaClientDep,
        lock: SimplificationLockDep,
        concurrency: PipelineConcurrencyDep,
        background_tasks: BackgroundTasks,
    ) -> None:
        self.jobs, self.mongo, self.ica, self.lock = jobs, mongo, ica, lock
        self.concurrency, self.background_tasks = concurrency, background_tasks

    async def start(
        self,
        source: str,
        question_ids: list[ObjectId],
        model_id: IcaModelId,
        mode: JobMode,
        only_missing: bool = False,
    ) -> str | None:
        """Queue a job for `question_ids` and return its id; None when there are none."""
        if not question_ids:
            return None
        job = await self.jobs.insert_one(
            new_job(source, model_id, len(question_ids), mode, only_missing)
        )
        self.background_tasks.add_task(
            run_simplification_job,
            self.mongo,
            self.ica,
            self.lock,
            job.inserted_id,
            question_ids,
            model_id,
            mode,
            only_missing,
            self.concurrency,
        )
        return str(job.inserted_id)


JobStarterDep = Annotated[_JobStarter, Depends(_JobStarter)]


def _refuse_while_a_job_runs(lock) -> None:
    """The single-question routes share the job lock so they never add ICA load on top
    of a running job: while one runs they answer 409 instead of waiting."""
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a simplification job is running; retry when it has finished",
        )


@router.post("/import")
async def import_practice_exam(
    file: UploadFile,
    collection: AsyncCollectionDep,
    starter: JobStarterDep,
    simplify: Annotated[
        bool, Query(description="Start the background pipeline job after import")
    ] = True,
    manual: Annotated[
        bool,
        Query(
            description="Store the questions held for manual processing: no job starts, "
            "the game does not play them and the backfills skip them until each is "
            "processed from the evaluation page (or released by hand)"
        ),
    ] = False,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    dry_run: Annotated[
        bool,
        Query(description="Store nothing and start no job: only report what an import would do"),
    ] = False,
) -> ImportResult:
    """
    Import a practice-exam JSON file. A question the bank already holds (same question
    text and answers) is left out, so the same file can be uploaded again. Unless
    `simplify=false`, a background job then runs the whole pipeline for each new
    question (job mode "full": objectives, story, decomposition with its parts and
    entities, icons, flow, resolution, rephrase, check);
    track it via `GET /examination/jobs/{job_id}`.

    The result also says what the pipeline cannot do for the new questions until a
    catalog is extended (`unknown_scenarios`, `unknown_sections`). Call it with
    `dry_run=true` first to see that before anything is stored.

    With `manual=true` no job starts whatever `simplify` says: the questions are stored
    held (`pipeline_mode: "manual"`, see `hold.py`) and processed one by one from the
    evaluation page; a question is released into the game once its steps are done.
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
    hashes = [doc["text_hash"] for doc in documents]
    stored = {
        doc["text_hash"]
        async for doc in collection.find({"text_hash": {"$in": hashes}}, {"text_hash": 1})
    }
    new, report = plan_import(documents, stored)
    if dry_run:
        return ImportResult(source=source, inserted=0, dry_run=True, **report)

    if manual:
        new = [{**doc, hold.FIELD: hold.MANUAL} for doc in new]
    inserted_ids: list[ObjectId] = []
    if new:
        inserted_ids = (await collection.insert_many(new)).inserted_ids

    job_id: str | None = None
    if simplify and not manual:
        job_id = await starter.start(source, inserted_ids, model_id, "full")

    return ImportResult(
        source=source, inserted=len(inserted_ids), job_id=job_id, manual=manual, **report
    )


@router.post("/pipeline/backfill", status_code=202)
async def backfill_pipeline(
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    story: AsyncStoryCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    starter: JobStarterDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    only_missing: Annotated[
        bool,
        Query(
            description="Only questions that lack a step or have one written with an "
            "older prompt or catalog, and only those steps"
        ),
    ] = True,
    source: Annotated[
        str | None, Query(max_length=300, description="Only questions imported from this file")
    ] = None,
    limit: Annotated[
        int | None, Query(ge=1, description="Process at most this many questions")
    ] = None,
    include_manual: Annotated[
        bool,
        Query(description="Also questions held for manual processing (imported with manual=true)"),
    ] = False,
) -> EntitiesBackfill:
    """
    Start a background job that runs the whole pipeline (job mode "full") for stored
    questions. By default it does only what is missing or outdated, step by step, so
    it is how an interrupted import is resumed and how a changed prompt or catalog is
    applied; `only_missing=false` redoes everything. Track it via
    `GET /examination/jobs/{job_id}`.
    """
    stores = Stores(collection, simplified, objectives, story, decomposition)
    question_ids: list[ObjectId] = []
    query = hold.with_hold({"source": source} if source else {}, include_manual)
    async for doc in collection.find(query):
        if only_missing and not await missing_stages("full", stores, doc):
            continue
        question_ids.append(doc["_id"])
        if limit is not None and len(question_ids) >= limit:
            break
    job_id = await starter.start("pipeline-backfill", question_ids, model_id, "full", only_missing)
    return EntitiesBackfill(job_id=job_id, total=len(question_ids))


@router.post("/{question_id}/icons")
async def pick_question_icons(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> DecompositionRecord:
    """
    (Re)pick the icons of one decomposed question through ICA: one key from the icon
    catalog for each of its entities, for the question and for each answer. Only the
    icon fields change. 404 without a decomposition; answers 409 while a job runs.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await assign_and_store(ica, decomposition, doc, model_id, max_tokens)


@router.post("/icons/backfill", status_code=202)
async def backfill_icons(
    collection: AsyncCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    starter: JobStarterDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    only_missing: Annotated[
        bool, Query(description="Skip questions whose icons are current")
    ] = True,
    include_manual: Annotated[
        bool,
        Query(description="Also questions held for manual processing (imported with manual=true)"),
    ] = False,
) -> EntitiesBackfill:
    """
    Start a background job that picks icons for decomposed questions (by default only
    those without current icons: none yet, or picked from another catalog, prompt or
    set of entities); track it via `GET /examination/jobs/{job_id}`.
    """
    stores = Stores(collection, None, None, None, decomposition)
    question_ids: list[ObjectId] = []
    async for record in decomposition.find({}, {"question_id": 1}):
        if not ObjectId.is_valid(record.get("question_id", "")):
            continue
        doc = await collection.find_one({"_id": ObjectId(record["question_id"])})
        if doc is None or (hold.is_held(doc) and not include_manual):
            continue
        if only_missing and await is_current("icons", stores, doc):
            continue
        question_ids.append(doc["_id"])
    job_id = await starter.start("icons-backfill", question_ids, model_id, "icons", only_missing)
    return EntitiesBackfill(job_id=job_id, total=len(question_ids))


@router.post("/{question_id}/check")
async def check_question(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    story: AsyncStoryCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> SimplifiedQuestion:
    """
    (Re)check the rephrase of one question: a model answers it without the key, and
    the verdict is stored on the rephrase record (`check`). A rephrase whose verdict is
    "changed" is not offered by the game. 404 without a rephrase; 409 while a job runs.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    _refuse_while_a_job_runs(lock)
    async with lock:
        record = await story.find_one({"question_id": str(question_id)})
        return await check_and_store(
            ica, simplified, doc, model_id, max_tokens, story=story_of(doc, record)
        )


@router.get("/{question_id}/decomposition")
async def get_question_decomposition(
    question_id: QuestionObjectIdDep,
    decomposition: AsyncDecompositionCollectionDep,
) -> DecompositionRecord:
    """The stored decomposition, pre-answer flow and resolution of one question (404
    when it has none). Includes the answer key (the resolution)."""
    record = await decomposition.find_one({"question_id": str(question_id)})
    if record is None:
        raise HTTPException(status_code=404, detail="question has no decomposition")
    try:
        return DecompositionRecord(**record)
    except ValidationError as exc:
        # A decomposition of an older shape: what it holds is outdated anyway.
        raise HTTPException(
            status_code=404,
            detail="the stored decomposition is outdated or malformed; decompose it again",
        ) from exc


@router.post("/{question_id}/decompose")
async def decompose_question(
    doc: ExaminationDocDep,
    decomposition: AsyncDecompositionCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> DecompositionRecord:
    """(Re)write the decomposition of one question (key-free, `decompose.py`), with its
    parts and entities. Its old icons, flow and resolution are cleared: run the pipeline
    backfill (or `/icons`, `/flow`, `/resolve`) after it. 409 while a job runs."""
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await decompose_and_store(ica, decomposition, doc, model_id, max_tokens)


@router.post("/{question_id}/flow")
async def draw_question_flow(
    doc: ExaminationDocDep,
    decomposition: AsyncDecompositionCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> DecompositionRecord:
    """(Re)write the pre-answer flow of a decomposed question (key-free, `flow.py`);
    its resolution is cleared. 404 without a current decomposition, 409 while a job
    runs."""
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await flow_and_store(ica, decomposition, doc, model_id, max_tokens)


@router.post("/{question_id}/resolve")
async def resolve_question(
    doc: ExaminationDocDep,
    decomposition: AsyncDecompositionCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> DecompositionRecord:
    """(Re)write the resolution of a question with a current flow (sees the key,
    `flow.py`). 404 without a flow, 409 while a job runs."""
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await resolve_and_store(ica, decomposition, doc, model_id, max_tokens)


@router.post("/{question_id}/objectives")
async def tag_question_objectives(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
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
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await tag_and_store(ica, objectives, doc, model_id, max_tokens)


@router.post("/objectives/backfill", status_code=202)
async def backfill_objectives(
    collection: AsyncCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    starter: JobStarterDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
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
    include_manual: Annotated[
        bool,
        Query(description="Also questions held for manual processing (imported with manual=true)"),
    ] = False,
) -> EntitiesBackfill:
    """
    Start a background job that tags questions with learning objectives (by default
    only those not tagged yet, or tagged before the objective catalog or the prompt
    last changed, so running it again after editing the catalog redoes everything);
    track it via `GET /examination/jobs/{job_id}`.
    """
    question_ids: list[ObjectId] = []
    stores = Stores(collection, None, objectives, None)
    query = hold.with_hold({"section": section} if section else {}, include_manual)
    async for doc in collection.find(query):
        if only_missing and await is_current("objectives", stores, doc):
            continue
        question_ids.append(doc["_id"])
        if limit is not None and len(question_ids) >= limit:
            break
    job_id = await starter.start(
        "objectives-backfill", question_ids, model_id, "objectives", only_missing
    )
    return EntitiesBackfill(job_id=job_id, total=len(question_ids))


@router.post("/{question_id}/story")
async def write_question_story(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    story: AsyncStoryCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> QuestionStory:
    """
    (Re)write the story of one question through ICA: the stage of its chapter and the
    lines shown after it (and the chapter itself for a question without a
    scenario paragraph), replacing its `question_story` record (also a hand-written
    seed record; load the seed again to restore that). It clears the beat, written for
    the old stage: write it again with `POST /examination/{id}/beat` (a story job does
    both). 422 when the question's scenario
    paragraph is not in the scenario catalog. Answers 409 while a job runs, like the
    entity route.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    _refuse_while_a_job_runs(lock)
    async with lock:
        return await story_and_store(ica, story, doc, model_id, max_tokens)


@router.post("/{question_id}/beat")
async def write_question_beat(
    question_id: QuestionObjectIdDep,
    collection: AsyncCollectionDep,
    story: AsyncStoryCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    ica: IcaClientDep,
    lock: SimplificationLockDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    max_tokens: MaxTokensQuery = None,
) -> QuestionStory:
    """
    (Re)write the beat of one question through ICA, key-free: the line shown before it,
    stored on its `question_story` record (a seed's hand-written beat too). 404 without
    a current story record (the stage comes from it); 409 while a job runs.
    """
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    _refuse_while_a_job_runs(lock)
    async with lock:
        record = await decomposition.find_one({"question_id": str(question_id)})
        return await beat_and_store(
            ica, story, doc, model_id, max_tokens, read_decomposition(record, doc)
        )


@router.post("/story/backfill", status_code=202)
async def backfill_story(
    collection: AsyncCollectionDep,
    story: AsyncStoryCollectionDep,
    starter: JobStarterDep,
    model_id: ModelIdQuery = "claude-haiku-5-5",
    only_missing: Annotated[
        bool,
        Query(
            description="Skip questions with a seed record or one written with the "
            "current scenario catalog and prompt"
        ),
    ] = True,
    limit: Annotated[
        int | None, Query(ge=1, description="Write at most this many stories")
    ] = None,
    include_manual: Annotated[
        bool,
        Query(description="Also questions held for manual processing (imported with manual=true)"),
    ] = False,
) -> EntitiesBackfill:
    """
    Start a background job that writes the story of questions (by default only those
    without a current record, so questions covered by the seed file cost nothing and
    a new import is the usual reason to run it); track it via
    `GET /examination/jobs/{job_id}`.
    """
    question_ids: list[ObjectId] = []
    stores = Stores(collection, None, None, story)
    async for doc in collection.find(hold.with_hold({}, include_manual)):
        if only_missing and not await missing_stages("story", stores, doc):
            continue
        question_ids.append(doc["_id"])
        if limit is not None and len(question_ids) >= limit:
            break
    job_id = await starter.start("story-backfill", question_ids, model_id, "story", only_missing)
    return EntitiesBackfill(job_id=job_id, total=len(question_ids))


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
