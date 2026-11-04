from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from claude_exam.api.dependencies import (
    GameServiceDep,
    QuestionObjectIdDep,
    RoundObjectIdDep,
    SimplifiedQuestionIdDep,
)
from claude_exam.api.schemas import (
    OBJECT_ID_PATTERN,
    GameAnswerRequest,
    GameAnswerResult,
    GameFiftyFiftyRequest,
    GameFiftyFiftyResult,
    GameQuestion,
    GameQuestionList,
    GameRound,
    GameRoundBatch,
    GameRoundBatchRequest,
    GameRoundQuestion,
    GameRoundQuestionRequest,
    GameRoundRequest,
    GameSectionList,
)
from claude_exam.api.security import round_rate_limit

# Backend for the Unity game; the rules live in `claude_exam.api.game.service`. Responses
# are flat, null-free objects (lists are wrapped in `items`) because UnityWebRequest +
# JsonUtility cannot parse a top-level JSON array or null values. Errors are
# `{"detail", "code"}` (see `claude_exam.api.errors` and `game.errors`).
#
# The game plays rounds (`POST /v1/game/rounds`), extended with `/batches` and search adds.
# Questions go out without their correct answer, feedback or explanation; the round
# document keeps those, and the game gets them for one question by answering it, once.
router = APIRouter(prefix="/game", tags=["game"])

QuestionIdPath = Annotated[str, Path(pattern=OBJECT_ID_PATTERN)]
Count = Annotated[int, Query(ge=1, le=50)]
Section = Annotated[str | None, Query(max_length=200)]
AssessmentType = Annotated[str | None, Query(max_length=100)]


@router.post("/rounds", status_code=201, dependencies=[Depends(round_rate_limit)])
async def create_game_round(body: GameRoundRequest, service: GameServiceDep) -> GameRound:
    """Start a round: random playable questions and their simplified versions, in one
    response. The answer key stays in the round document. 404 `no_questions` when nothing
    matches the filters."""
    return await service.create_round(body.count, body.section, body.assessment_type)


@router.post("/rounds/{round_id}/batches")
async def next_game_round_batch(
    body: GameRoundBatchRequest, round_oid: RoundObjectIdDep, service: GameServiceDep
) -> GameRoundBatch:
    """The round's next questions, none already in it (searched ones included), so a round
    goes on without repeats until the pool runs out (empty `items`). Request batches in
    order; asking for an existing batch again returns it (a retry after a lost response)."""
    return await service.next_batch(round_oid, body.batch_index, body.count)


@router.post("/rounds/{round_id}/answers")
async def answer_game_round_question(
    body: GameAnswerRequest, round_oid: RoundObjectIdDep, service: GameServiceDep
) -> GameAnswerResult:
    """Answer one question of a round, once; sending the same answer again repeats the
    result. A different answer is 409 `already_answered`: read the recorded one with
    `GET /rounds/{round_id}/answers/{question_id}`."""
    return await service.answer(round_oid, body.question_id, body.answer_index)


@router.get("/rounds/{round_id}/answers/{question_id}")
async def get_game_round_answer(
    question_id: QuestionIdPath, round_oid: RoundObjectIdDep, service: GameServiceDep
) -> GameAnswerResult:
    """The answer the round recorded for a question, e.g. when the answer's response was
    lost and the player then picked another card. 404 `question_not_answered` if none."""
    return await service.recorded_answer(round_oid, question_id)


@router.post("/rounds/{round_id}/fifty-fifty")
async def use_game_round_fifty_fifty(
    body: GameFiftyFiftyRequest, round_oid: RoundObjectIdDep, service: GameServiceDep
) -> GameFiftyFiftyResult:
    """Remove two random wrong answers from an unanswered question of the round. Once per
    question (asking again repeats the result), and the round has `fifty_fifty_charges` uses."""
    return await service.fifty_fifty(round_oid, body.question_id)


@router.post("/rounds/{round_id}/questions")
async def add_game_round_question(
    body: GameRoundQuestionRequest, round_oid: RoundObjectIdDep, service: GameServiceDep
) -> GameRoundQuestion:
    """Add one question (e.g. a search result) to a round, so it can be answered there.
    Adding a question the round already has changes nothing and returns it again.
    Like the round itself, the response carries no answer key."""
    return await service.add_question(round_oid, body.question_id)


@router.get("/questions")
async def list_game_questions(
    service: GameServiceDep,
    count: Count = 10,
    section: Section = None,
    assessment_type: AssessmentType = None,
    shuffle: bool = True,
) -> GameQuestionList:
    """A batch of playable questions (random by default), without answer keys. To play,
    start a round."""
    return await service.list_questions(count, section, assessment_type, shuffle)


@router.get("/simplified-questions")
async def list_simplified_game_questions(
    service: GameServiceDep,
    count: Count = 10,
    section: Section = None,
    assessment_type: AssessmentType = None,
    shuffle: bool = True,
) -> GameQuestionList:
    """A batch of plain-language questions (only those already simplified and playable)."""
    return await service.list_questions(
        count, section, assessment_type, shuffle, simplified=True
    )


@router.get("/simplified-questions/{question_id}")
async def get_simplified_game_question(
    question_id: SimplifiedQuestionIdDep, service: GameServiceDep
) -> GameQuestion:
    """Simplified version of one question; `question_id` is the id `/v1/game/questions` returned."""
    return await service.get_simplified(question_id)


@router.get("/questions/{question_id}")
async def get_game_question(
    object_id: QuestionObjectIdDep, service: GameServiceDep
) -> GameQuestion:
    return await service.get_question(object_id)


@router.get("/sections")
async def list_game_sections(
    service: GameServiceDep, simplified: bool = False
) -> GameSectionList:
    """Distinct sections, so the game can offer a category choice. With
    `simplified=true` only sections that have simplified questions."""
    return GameSectionList(items=await service.sections(simplified=simplified))
