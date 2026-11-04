from typing import Annotated

from fastapi import APIRouter, Query

from claude_exam.api.dependencies import AsyncCollectionDep
from claude_exam.api.schemas import SearchResponse
from claude_exam.api.search import search_questions

# Study lookup: finds questions by text and returns the answer key and explanation.
# Kept apart from /game, whose responses never reveal the answer before answering.
router = APIRouter(prefix="/search", tags=["search"])


@router.get("/questions")
async def search(
    collection: AsyncCollectionDep,
    q: Annotated[str, Query(min_length=2, max_length=200)],
    section: Annotated[str, Query()] = "",
    assessment_type: Annotated[str, Query()] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
) -> SearchResponse:
    return await search_questions(
        collection, q, section=section, assessment_type=assessment_type, limit=limit
    )
