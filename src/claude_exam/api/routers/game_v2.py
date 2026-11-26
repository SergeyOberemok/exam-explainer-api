"""`/v2/game` and `/v2/search/game-questions`: the game over composed questions.

Each question is one `game_question` document, put together by the pipeline's last step
(`compose.py`), so these routes only read it. A `GameCard` is what a player meets before
answering, in order: the beat, the briefing item by item, the ask, the tools (the
answers, each with what it proposes and the cast members it is made of) and the map
(the pre-answer flow). Answering returns the `GameOutcome` of the answer given: its
impact (`success`, `partial`, `failure`) and the story consequence, with the review and
the resolved flow.

Flat, null-free responses with lists in `items` (JsonUtility), the answer key only after
answering, once (`game.service.RoundRules`).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncGameQuestionCollectionDep,
    CardServiceDep,
    RoundObjectIdDep,
)
from claude_exam.api.schemas import (
    OBJECT_ID_PATTERN,
    SEARCH_TEXT_MAX,
    GameAnswerRequest,
    GameCard,
    GameCardAnswerResult,
    GameCardBatch,
    GameCardList,
    GameCardRound,
    GameCardRoundQuestion,
    GameFiftyFiftyRequest,
    GameFiftyFiftyResult,
    GameObjectiveList,
    GameRoundBatchRequest,
    GameRoundQuestionRequest,
    GameRoundRequest,
    GameSectionList,
    GameStory,
    SearchRequest,
)
from claude_exam.api.search import search_game_cards
from claude_exam.api.security import round_rate_limit, search_rate_limit

router = APIRouter(prefix="/game", tags=["game v2"])
search_router = APIRouter(
    prefix="/search", tags=["game v2"], dependencies=[Depends(search_rate_limit)]
)

QuestionIdPath = Annotated[str, Path(pattern=OBJECT_ID_PATTERN)]
Count = Annotated[int, Query(ge=1, le=50)]
Section = Annotated[str | None, Query(max_length=200)]
AssessmentType = Annotated[str | None, Query(max_length=100)]
SearchText = Annotated[str, Query(min_length=2, max_length=SEARCH_TEXT_MAX)]
Limit = Annotated[int, Query(ge=1, le=50)]


@router.post("/rounds", status_code=201, dependencies=[Depends(round_rate_limit)])
async def create_round(body: GameRoundRequest, service: CardServiceDep) -> GameCardRound:
    """Start a round of composed questions (the same filters as `/v1/game/rounds`). 404
    `no_questions` when nothing matches."""
    return await service.create_round(
        body.count,
        body.section,
        body.assessment_type,
        body.objective,
        body.scenario,
        body.stage,
        body.ordered,
    )


@router.post("/rounds/{round_id}/batches")
async def next_round_batch(
    body: GameRoundBatchRequest, round_oid: RoundObjectIdDep, service: CardServiceDep
) -> GameCardBatch:
    """The round's next questions, none already in it; asking for an existing batch
    returns it again."""
    return await service.next_batch(round_oid, body.batch_index, body.count)


@router.post("/rounds/{round_id}/answers")
async def answer_round_question(
    body: GameAnswerRequest, round_oid: RoundObjectIdDep, service: CardServiceDep
) -> GameCardAnswerResult:
    """Answer one question of a round, once: the outcome of that answer, the review and
    the resolved flow. The same answer again repeats the result; another one is 409
    `already_answered`."""
    return await service.answer(round_oid, body.question_id, body.answer_index)


@router.get("/rounds/{round_id}/answers/{question_id}")
async def get_round_answer(
    question_id: QuestionIdPath, round_oid: RoundObjectIdDep, service: CardServiceDep
) -> GameCardAnswerResult:
    """The answer the round recorded for a question (404 `question_not_answered`)."""
    return await service.recorded_answer(round_oid, question_id)


@router.post("/rounds/{round_id}/fifty-fifty")
async def use_round_fifty_fifty(
    body: GameFiftyFiftyRequest, round_oid: RoundObjectIdDep, service: CardServiceDep
) -> GameFiftyFiftyResult:
    """Remove two random wrong answers (tools) from an unanswered question."""
    return await service.fifty_fifty(round_oid, body.question_id)


@router.post("/rounds/{round_id}/questions")
async def add_round_question(
    body: GameRoundQuestionRequest, round_oid: RoundObjectIdDep, service: CardServiceDep
) -> GameCardRoundQuestion:
    """Add a served question (a search result) to the round. 404 `question_not_found`
    for an unknown id, 422 `question_not_playable` for one the game does not serve."""
    return await service.add_question(round_oid, body.question_id)


@router.get("/questions")
async def list_questions(
    service: CardServiceDep,
    count: Count = 10,
    section: Section = None,
    assessment_type: AssessmentType = None,
    shuffle: bool = True,
) -> GameCardList:
    """Served questions without their answer keys. To play, start a round."""
    return await service.list_questions(count, section, assessment_type, shuffle)


@router.get("/questions/{question_id}")
async def get_question(question_id: QuestionIdPath, service: CardServiceDep) -> GameCard:
    return await service.get_question(question_id)


@router.get("/sections")
async def list_sections(service: CardServiceDep) -> GameSectionList:
    return GameSectionList(items=await service.sections())


@router.get("/objectives")
async def list_objectives(service: CardServiceDep) -> GameObjectiveList:
    """The objective catalog with how many served questions each objective has."""
    return await service.objectives()


@router.get("/story")
async def get_story(service: CardServiceDep) -> GameStory:
    """The company, its chapters and stages with their served question counts."""
    return await service.story()


@search_router.get("/game-questions")
async def search_cards(
    collection: AsyncCollectionDep,
    cards: AsyncGameQuestionCollectionDep,
    q: SearchText,
    section: Annotated[str, Query(max_length=200)] = "",
    assessment_type: Annotated[str, Query(max_length=100)] = "",
    limit: Limit = 5,
) -> GameCardList:
    """Served questions matching `q`, best first, without their answer key. To play one,
    add it to a round with `POST /v2/game/rounds/{round_id}/questions`."""
    return await search_game_cards(
        collection, cards, q, section=section, assessment_type=assessment_type, limit=limit
    )


@search_router.post("/game-questions")
async def search_cards_by_body(
    collection: AsyncCollectionDep, cards: AsyncGameQuestionCollectionDep, body: SearchRequest
) -> GameCardList:
    """`GET /v2/search/game-questions` with the fields in a JSON body."""
    return await search_game_cards(
        collection,
        cards,
        body.q,
        section=body.section,
        assessment_type=body.assessment_type,
        limit=body.limit,
    )
