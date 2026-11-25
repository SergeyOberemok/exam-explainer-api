"""The prompt evaluation page's API: read one question with its stored records, and dry-run
pipeline steps on it (`pipeline_preview.run_dry`: computed in memory, nothing written).

The page itself is static (`api/static/evaluate/index.html`, served at `/evaluate/ui/`).
"""

from collections.abc import AsyncIterable
from typing import Any, get_args

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from claude_exam.api.batch import STAGES
from claude_exam.api.dependencies import (
    AsyncDecompositionCollectionDep,
    AsyncObjectivesCollectionDep,
    AsyncSimplifiedCollectionDep,
    AsyncStoryCollectionDep,
    ExaminationDocDep,
    IcaClientDep,
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


@router.post("/questions/{question_id}/run")
async def run_evaluation(
    body: EvaluationRunRequest,
    doc: ExaminationDocDep,
    ica: IcaClientDep,
    simplified: AsyncSimplifiedCollectionDep,
    decomposition: AsyncDecompositionCollectionDep,
    story: AsyncStoryCollectionDep,
) -> AsyncIterable[EvaluationEvent]:
    """Dry-runs `stages` in order, starting from the stored records; streams JSON Lines,
    one per step as it finishes. Writes nothing."""
    query = {"question_id": str(doc["_id"])}
    result = Result.from_records(
        doc, await simplified.find_one(query), await decomposition.find_one(query)
    )
    story_record = await story.find_one(query)
    async for step in run_dry(ica, doc, result, story_record, body.stages, body.model):
        yield EvaluationEvent(**step, result=result.as_json(doc))
    yield EvaluationEvent(stage="done", status="done", result=result.as_json(doc))
