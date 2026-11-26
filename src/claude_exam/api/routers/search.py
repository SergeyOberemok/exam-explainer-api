from typing import Annotated

from fastapi import APIRouter, Depends, Query

from claude_exam.api.dependencies import AsyncCollectionDep, AsyncSimplifiedCollectionDep
from claude_exam.api.schemas import (
    SEARCH_TEXT_MAX,
    GameQuestionList,
    SearchRequest,
    SearchResponse,
)
from claude_exam.api.search import search_game_questions, search_questions
from claude_exam.api.security import require_api_key, search_rate_limit

# Study lookup: finds questions by text and returns the answer key and explanation.
# Kept apart from /game, whose responses never reveal the answer before answering.
# `/v1/search/game-questions` is the exception: the game's search, without the key.
#
# Each search has a GET (short texts, easy to call by hand) and a POST taking the same
# fields as a JSON body (`SearchRequest`), for a whole pasted question: percent-encoded,
# such a text can pass the request-line limit of a server or proxy, which then answers
# 400/414 without the API's error body.
router = APIRouter(
    prefix="/search", tags=["search"], dependencies=[Depends(search_rate_limit)]
)

SearchText = Annotated[str, Query(min_length=2, max_length=SEARCH_TEXT_MAX)]
Limit = Annotated[int, Query(ge=1, le=50)]
Section = Annotated[str, Query(max_length=200)]
AssessmentType = Annotated[str, Query(max_length=100)]


# Returns the answer key, so it needs the API key when one is configured.
@router.get("/questions", dependencies=[Depends(require_api_key)])
async def search(
    collection: AsyncCollectionDep,
    q: SearchText,
    section: Section = "",
    assessment_type: AssessmentType = "",
    limit: Limit = 5,
) -> SearchResponse:
    return await search_questions(
        collection, q, section=section, assessment_type=assessment_type, limit=limit
    )


@router.post("/questions", dependencies=[Depends(require_api_key)])
async def search_by_body(
    collection: AsyncCollectionDep, body: SearchRequest
) -> SearchResponse:
    """`GET /v1/search/questions` with the fields in a JSON body."""
    return await search_questions(
        collection,
        body.q,
        section=body.section,
        assessment_type=body.assessment_type,
        limit=body.limit,
    )


@router.get("/game-questions")
async def search_game(
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    q: SearchText,
    section: Section = "",
    assessment_type: AssessmentType = "",
    limit: Limit = 5,
) -> GameQuestionList:
    """Playable questions matching `q`, best first, without their answer key. To play one,
    add it to a round with `POST /v1/game/rounds/{round_id}/questions`."""
    return await search_game_questions(
        collection, simplified, q, section=section, assessment_type=assessment_type, limit=limit
    )


@router.post("/game-questions")
async def search_game_by_body(
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep, body: SearchRequest
) -> GameQuestionList:
    """`GET /v1/search/game-questions` with the fields in a JSON body: what the game
    sends, since a pasted question can be too long for a URL."""
    return await search_game_questions(
        collection,
        simplified,
        body.q,
        section=body.section,
        assessment_type=body.assessment_type,
        limit=body.limit,
    )
