"""The prompt evaluation page's API: read one question with its stored records, and dry-run
pipeline steps on it (`pipeline_preview.run_dry`: computed in memory, nothing written), or,
with `persist`, really run them (`batch.run_simplification_job`: the records are written).

The page itself is static (`api/static/evaluate/index.html`, served at `/evaluate/ui/`).
"""

import time
from collections.abc import AsyncIterable
from typing import Any, get_args

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from claude_exam.api.batch import STAGES, new_job, run_simplification_job
from claude_exam.api.dependencies import (
    AsyncMongoClientDep,
    AsyncDecompositionCollectionDep,
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


class EvaluationOptions(BaseModel):
    stages: list[str]
    dry_stages: list[str]
    models: list[str]
    default_model: str


@router.get("/options")
async def get_options() -> EvaluationOptions:
    """The steps (in pipeline order) and the models the page offers."""
    return EvaluationOptions(
        stages=list(STAGES["full"]),
        dry_stages=list(DRY_STAGES),
        models=list(get_args(IcaModelId)),
        default_model="claude-haiku-5-5",
    )


@router.get("/questions/{question_id}")
async def get_evaluation(
    doc: ExaminationDocDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    story: AsyncStoryCollectionDep,
    objectives: AsyncObjectivesCollectionDep,
) -> EvaluationView:
    """The original question (with its key) and what the pipeline stored for it."""
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc, await simplified.find_one(query), await decomposition.find_one(query)
    )
    return EvaluationView(
        original=original_view(
            doc, await story.find_one(query), await objectives.find_one(query)
        ),
        stored=result.as_json(doc),
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
    async for step in run_dry(ica, doc, result, story_record, body.stages, body.model):
        yield EvaluationEvent(**step, result=result.as_json(doc))
    yield EvaluationEvent(stage="done", status="done", result=result.as_json(doc))


async def _stored_result(doc: dict, simplified, decomposition, story) -> Result:
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc, await simplified.find_one(query), await decomposition.find_one(query)
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
        result = await _stored_result(doc, simplified, decomposition, story)
        yield EvaluationEvent(**step, result=result.as_json(doc))
    result = await _stored_result(doc, simplified, decomposition, story)
    yield EvaluationEvent(stage="done", status="done", result=result.as_json(doc))
