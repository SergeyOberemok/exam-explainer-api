import asyncio
import re
from dataclasses import dataclass
from typing import Annotated

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, Query, Request
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.errors import (
    QuestionNotFound,
    RoundNotFound,
    SimplifiedNotFound,
)
from claude_exam.api.game.repository import QuestionRepository, RoundRepository
from claude_exam.api.game.service import GameService
from claude_exam.api.ica import IcaClient
from claude_exam.api.schemas import OBJECT_ID_PATTERN


def get_ica_client(request: Request) -> IcaClient:
    return request.app.state.ica_client


IcaClientDep = Annotated[IcaClient, Depends(get_ica_client)]


def get_async_collection(request: Request) -> AsyncCollection:
    """The `examination` collection (the question bank)."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["examination"]


AsyncCollectionDep = Annotated[AsyncCollection, Depends(get_async_collection)]


def get_async_simplified_collection(request: Request) -> AsyncCollection:
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["examination_simplified"]


AsyncSimplifiedCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_simplified_collection)
]


def get_async_rounds_collection(request: Request) -> AsyncCollection:
    """Game rounds: which questions a round holds and what has been answered."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["game_rounds"]


AsyncRoundsCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_rounds_collection)
]


def get_async_mongo_client(request: Request) -> AsyncMongoClient:
    return request.app.state.async_mongo_client


AsyncMongoClientDep = Annotated[AsyncMongoClient, Depends(get_async_mongo_client)]


def get_simplification_lock(request: Request) -> asyncio.Semaphore:
    return request.app.state.simplification_lock


SimplificationLockDep = Annotated[asyncio.Semaphore, Depends(get_simplification_lock)]


def get_async_jobs_collection(request: Request) -> AsyncCollection:
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["simplification_jobs"]


AsyncJobsCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_jobs_collection)
]


def get_question_object_id(question_id: str) -> ObjectId:
    """Path `question_id` as an ObjectId; a malformed id is a 404, not a 422."""
    try:
        return ObjectId(question_id)
    except InvalidId as exc:
        raise QuestionNotFound() from exc


QuestionObjectIdDep = Annotated[ObjectId, Depends(get_question_object_id)]


def get_round_object_id(round_id: str) -> ObjectId:
    """Path `round_id` as an ObjectId; a malformed id is a 404, like an expired round."""
    try:
        return ObjectId(round_id)
    except InvalidId as exc:
        raise RoundNotFound() from exc


RoundObjectIdDep = Annotated[ObjectId, Depends(get_round_object_id)]


def get_game_service(
    examination: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    rounds: AsyncRoundsCollectionDep,
) -> GameService:
    return GameService(QuestionRepository(examination, simplified), RoundRepository(rounds))


GameServiceDep = Annotated[GameService, Depends(get_game_service)]


_OBJECT_ID_HEX = re.compile(OBJECT_ID_PATTERN)


def get_simplified_question_id(question_id: str) -> str:
    """Path `question_id` of a simplified question (stored as a string); a malformed id is a
    404 like an unknown one."""
    if not _OBJECT_ID_HEX.match(question_id):
        raise SimplifiedNotFound()
    return question_id


SimplifiedQuestionIdDep = Annotated[str, Depends(get_simplified_question_id)]


@dataclass
class Pagination:
    offset: int
    limit: int


def get_pagination(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Pagination:
    return Pagination(offset=offset, limit=limit)


PaginationDep = Annotated[Pagination, Depends(get_pagination)]


@dataclass
class QuestionFilters:
    source: str | None = None
    section: str | None = None
    assessment_type: str | None = None
    q: str | None = None

    def to_mongo(self) -> dict:
        query: dict = {}
        if self.source:
            query["source"] = self.source
        if self.section:
            query["section"] = self.section
        if self.assessment_type:
            query["assessment_type"] = self.assessment_type
        if self.q:
            query["prompt.question"] = {"$regex": re.escape(self.q), "$options": "i"}
        return query


def get_question_filters(
    source: str | None = None,
    section: str | None = None,
    assessment_type: str | None = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
) -> QuestionFilters:
    return QuestionFilters(
        source=source, section=section, assessment_type=assessment_type, q=q
    )


QuestionFiltersDep = Annotated[QuestionFilters, Depends(get_question_filters)]
