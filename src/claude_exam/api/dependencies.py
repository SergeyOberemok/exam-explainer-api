import asyncio
import re
from dataclasses import dataclass
from typing import Annotated

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException, Query, Request
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.errors import (
    QuestionNotFound,
    RoundNotFound,
)
from claude_exam.api.game.card_repository import CardRepository
from claude_exam.api.game.card_service import CardService
from claude_exam.api.game.repository import RoundRepository
from claude_exam.api.ica import IcaClient


def get_ica_client(request: Request) -> IcaClient:
    return request.app.state.ica_client


IcaClientDep = Annotated[IcaClient, Depends(get_ica_client)]


def get_async_collection(request: Request) -> AsyncCollection:
    """The `examination` collection (the question bank)."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["examination"]


AsyncCollectionDep = Annotated[AsyncCollection, Depends(get_async_collection)]


def get_async_objectives_collection(request: Request) -> AsyncCollection:
    """Learning-objective tags, one record per examination question."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["question_objectives"]


AsyncObjectivesCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_objectives_collection)
]


def get_async_story_collection(request: Request) -> AsyncCollection:
    """Story records (chapter stage and story lines), one per examination question."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["question_story"]


AsyncStoryCollectionDep = Annotated[AsyncCollection, Depends(get_async_story_collection)]


def get_async_decomposition_collection(request: Request) -> AsyncCollection:
    """Question decompositions, flows and resolutions, one record per question."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["question_decomposition"]


AsyncDecompositionCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_decomposition_collection)
]


def get_async_game_question_collection(request: Request) -> AsyncCollection:
    """The composed game documents (`compose.py`), one per question the game serves."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["game_question"]


AsyncGameQuestionCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_game_question_collection)
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


def get_pipeline_concurrency(request: Request) -> int:
    """Questions a pipeline job works on at the same time (`PIPELINE_CONCURRENCY`)."""
    return getattr(request.app.state, "pipeline_concurrency", 1)


PipelineConcurrencyDep = Annotated[int, Depends(get_pipeline_concurrency)]


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


async def get_examination_doc(
    question_id: QuestionObjectIdDep, collection: AsyncCollectionDep
) -> dict:
    """The `examination` document of the path's `question_id` (404 when there is none)."""
    doc = await collection.find_one({"_id": question_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    return doc


ExaminationDocDep = Annotated[dict, Depends(get_examination_doc)]


def get_round_object_id(round_id: str) -> ObjectId:
    """Path `round_id` as an ObjectId; a malformed id is a 404, like an expired round."""
    try:
        return ObjectId(round_id)
    except InvalidId as exc:
        raise RoundNotFound() from exc


RoundObjectIdDep = Annotated[ObjectId, Depends(get_round_object_id)]


def get_card_service(
    cards: AsyncGameQuestionCollectionDep,
    examination: AsyncCollectionDep,
    rounds: AsyncRoundsCollectionDep,
) -> CardService:
    """The `/v2/game` service, over the composed `game_question` documents."""
    return CardService(CardRepository(cards, examination), RoundRepository(rounds))


CardServiceDep = Annotated[CardService, Depends(get_card_service)]


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
