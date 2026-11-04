from typing import Annotated

from fastapi import APIRouter, Query

from claude_exam.api.dependencies import AsyncCollectionDep
from claude_exam.api.schemas import GameQuestionList, SearchResponse
from claude_exam.api.search import search_game_questions, search_questions

# Study lookup: finds questions by text and returns the answer key and explanation.
# Kept apart from /game, whose responses never reveal the answer before answering.
# `/search/game-questions` is the exception: the game's search, without the key.
router = APIRouter(prefix="/search", tags=["search"])

SearchText = Annotated[str, Query(min_length=2, max_length=200)]
Limit = Annotated[int, Query(ge=1, le=50)]


@router.get("/questions")
async def search(
    collection: AsyncCollectionDep,
    q: SearchText,
    section: Annotated[str, Query()] = "",
    assessment_type: Annotated[str, Query()] = "",
    limit: Limit = 5,
) -> SearchResponse:
    return await search_questions(
        collection, q, section=section, assessment_type=assessment_type, limit=limit
    )


@router.get("/game-questions")
async def search_game(
    collection: AsyncCollectionDep,
    q: SearchText,
    section: Annotated[str, Query()] = "",
    assessment_type: Annotated[str, Query()] = "",
    limit: Limit = 5,
) -> GameQuestionList:
    """Playable questions matching `q`, best first, without their answer key. To play one,
    add it to a round with `POST /game/rounds/{round_id}/questions`."""
    return await search_game_questions(
        collection, q, section=section, assessment_type=assessment_type, limit=limit
    )
