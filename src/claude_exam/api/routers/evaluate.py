"""The prompt evaluation page's API: read one question with its stored records, and dry-run
pipeline steps on it (`pipeline_preview.run_dry`: computed in memory, nothing written), or,
with `persist`, really run them (`batch.run_simplification_job`: the records are written).

The page itself is static (`api/static/evaluate/index.html`, served at `/evaluate/ui/`).
It also lists the questions held for manual processing (an import with `manual=true`,
see `hold.py`) and releases or holds one by hand.
"""

import time
from collections.abc import AsyncIterable
from typing import Annotated, Any, get_args

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from claude_exam.api import hold
from claude_exam.api.batch import STAGES, Stores, missing_stages, new_job, run_simplification_job
from claude_exam.api.compose import set_held
from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncMongoClientDep,
    AsyncDecompositionCollectionDep,
    AsyncGameQuestionCollectionDep,
    AsyncObjectivesCollectionDep,
    AsyncSimplifiedCollectionDep,
    AsyncStoryCollectionDep,
    ExaminationDocDep,
    IcaClientDep,
    SimplificationLockDep,
)
from claude_exam.api.pipeline_preview import DRY_STAGES, Result, original_view, run_dry
from claude_exam.api.schemas import IcaModelId, JobStage
from claude_exam.api.security import require_api_key
from claude_exam.html_clean import clean_question

router = APIRouter(
    prefix="/evaluate", tags=["evaluate"], dependencies=[Depends(require_api_key)]
)


class EvaluationView(BaseModel):
    original: dict[str, Any]
    stored: dict[str, Any]


class EvaluationRunRequest(BaseModel):
    stages: list[JobStage] = Field(default_factory=lambda: list(DRY_STAGES), min_length=1)
    model: IcaModelId = "claude-haiku-5-5"
    persist: bool = Field(
        default=False,
        description="a real run: the steps write their records (any step of the full job)",
    )
    force: bool = Field(
        default=False, description="with persist: redo steps whose result is current"
    )


class EvaluationEvent(BaseModel):
    """One line of the run's stream: a finished step and the result so far, then a last
    line with `stage: "done"`."""

    stage: str
    status: str
    tokens: int = 0
    seconds: float = 0
    detail: str = ""
    warnings: list[str] = []
    result: dict[str, Any]


class HeldQuestion(BaseModel):
    id: str
    source: str
    section: str
    question: str
    missing: list[str]


class HeldQuestionList(BaseModel):
    items: list[HeldQuestion]
    total: int


class HoldState(BaseModel):
    id: str
    held: bool
    missing: list[str]


# Characters of the question text a list item shows.
_SNIPPET = 160


class EvaluationOptions(BaseModel):
    stages: list[str]
    dry_stages: list[str]
    models: list[str]
    default_model: str


@router.get("/options")
async def get_options() -> EvaluationOptions:
    """The steps (in pipeline order) and the models the page offers."""
    return EvaluationOptions(
        stages=[*STAGES["full"], *STAGES["rephrase"]],
        dry_stages=list(DRY_STAGES),
        models=list(get_args(IcaModelId)),
        default_model="claude-haiku-5-5",
    )


@router.get("/questions")
async def list_held_questions(
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    game_question: AsyncGameQuestionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    source: Annotated[
        str | None, Query(max_length=300, description="Only questions imported from this file")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> HeldQuestionList:
    """The questions held for manual processing (imported with `manual=true`), oldest
    first, each with the steps of the full pipeline it still needs."""
    stores = Stores(collection, simplified, objectives, story, decomposition, game_question)
    query = {**hold.HELD, **({"source": source} if source else {})}
    items = []
    async for doc in collection.find(query).sort("_id", 1).limit(limit):
        text = " ".join(clean_question((doc.get("prompt") or {}).get("question", "")).split())
        items.append(
            HeldQuestion(
                id=str(doc["_id"]),
                source=doc.get("source", ""),
                section=doc.get("section", ""),
                question=text[:_SNIPPET],
                missing=await missing_stages("full", stores, doc),
            )
        )
    return HeldQuestionList(items=items, total=await collection.count_documents(query))


@router.get("/questions/{question_id}")
async def get_evaluation(
    doc: ExaminationDocDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    game_question: AsyncGameQuestionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
) -> EvaluationView:
    """The original question (with its key, whether it is held for manual processing and
    the steps it still needs) and what the pipeline stored for it."""
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc,
        await simplified.find_one(query),
        await decomposition.find_one(query),
        await game_question.find_one(query),
    )
    stores = Stores(collection, simplified, objectives, story, decomposition, game_question)
    return EvaluationView(
        original={
            **original_view(doc, await story.find_one(query), await objectives.find_one(query)),
            "held": hold.is_held(doc),
            "missing": await missing_stages("full", stores, doc),
        },
        stored=result.as_json(doc),
    )


async def _set_hold(doc: dict, held: bool, stores: Stores) -> HoldState:
    update = {"$set": {hold.FIELD: hold.MANUAL}} if held else {"$unset": {hold.FIELD: ""}}
    await stores.examination.update_one({"_id": doc["_id"]}, update)
    await set_held(stores.game_question, str(doc["_id"]), held)
    return HoldState(
        id=str(doc["_id"]), held=held, missing=await missing_stages("full", stores, doc)
    )


@router.post("/questions/{question_id}/release")
async def release_question(
    doc: ExaminationDocDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    game_question: AsyncGameQuestionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
) -> HoldState:
    """Releases a held question into the game (and the backfills) as it is, whatever
    steps it still lacks. A real run releases it by itself once its steps are done."""
    return await _set_hold(
        doc, False, Stores(collection, simplified, objectives, story, decomposition, game_question)
    )


@router.post("/questions/{question_id}/hold")
async def hold_question(
    doc: ExaminationDocDep,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    game_question: AsyncGameQuestionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
) -> HoldState:
    """Holds a question for manual processing: the game no longer plays it and the
    backfills skip it. A real run that leaves none of its required steps missing
    releases it again."""
    return await _set_hold(
        doc, True, Stores(collection, simplified, objectives, story, decomposition, game_question)
    )


def _refuse_persist_while_a_job_runs(
    body: EvaluationRunRequest, lock: SimplificationLockDep
) -> None:
    """A real run shares the job lock, so it answers 409 instead of waiting (checked before
    the stream starts: an error inside it could no longer change the status)."""
    if body.persist and lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a simplification job is running; retry when it has finished",
        )


@router.post(
    "/questions/{question_id}/run",
    dependencies=[Depends(_refuse_persist_while_a_job_runs)],
)
async def run_evaluation(
    body: EvaluationRunRequest,
    doc: ExaminationDocDep,
    ica: IcaClientDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
    mongo: AsyncMongoClientDep,
    lock: SimplificationLockDep,
) -> AsyncIterable[EvaluationEvent]:
    """Runs `stages` in order; streams JSON Lines, one per step as it finishes.

    By default a dry run from the stored records: writes nothing. With `persist` it is a
    real run that writes the records (any step of the `full` job; 409 while a job runs)."""
    if body.persist:
        async for event in _persist_evaluation(
            body, doc, ica, mongo, lock, simplified, decomposition, story
        ):
            yield event
        return
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc, await simplified.find_one(query), await decomposition.find_one(query)
    )
    story_record = await story.find_one(query)
    tags = await objectives.find_one(query)
    async for step in run_dry(
        ica, doc, result, story_record, body.stages, body.model, objectives_record=tags
    ):
        yield EvaluationEvent(**step, result=result.as_json(doc))
    yield EvaluationEvent(stage="done", status="done", result=result.as_json(doc))


async def _stored_result(doc: dict, simplified, decomposition, story, game=None) -> Result:
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc,
        await simplified.find_one(query),
        await decomposition.find_one(query),
        await game.find_one(query) if game is not None else None,
    )
    result.beat = ((await story.find_one(query)) or {}).get("beat") or ""
    return result


async def _persist_evaluation(
    body: EvaluationRunRequest,
    doc: dict,
    ica,
    mongo,
    lock,
    simplified,
    decomposition,
    story,
) -> AsyncIterable[EvaluationEvent]:
    """Real run: each step is a one-question job (`batch.run_simplification_job`, the path
    of the import and the backfill), so its result is WRITTEN; after each step the stored
    records are read back."""
    jobs = mongo.get_default_database()["simplification_jobs"]
    for stage in body.stages:
        started = time.monotonic()
        inserted = await jobs.insert_one(
            new_job("evaluate", body.model, 1, "full", only_missing=not body.force)
        )
        await run_simplification_job(
            mongo,
            ica,
            lock,
            inserted.inserted_id,
            [doc["_id"]],
            body.model,
            "full",
            only_missing=not body.force,
            concurrency=1,
            stages=(stage,),
        )
        job = await jobs.find_one({"_id": inserted.inserted_id}) or {}
        error = next((e for e in job.get("errors", []) if e.get("stage") == stage), None)
        step = {
            "stage": stage,
            "status": f"failed {error['status_code']}" if error else "ok or current",
            "seconds": round(time.monotonic() - started, 1),
            "detail": error["detail"] if error else "",
        }
        result = await _stored_result(
            doc, simplified, decomposition, story, mongo.get_default_database()["game_question"]
        )
        yield EvaluationEvent(**step, result=result.as_json(doc))
    result = await _stored_result(
            doc, simplified, decomposition, story, mongo.get_default_database()["game_question"]
        )
    yield EvaluationEvent(stage="done", status="done", result=result.as_json(doc))
