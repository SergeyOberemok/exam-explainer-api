import asyncio
import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query, Request
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.collection import Collection

from claude_exam.api.ica import IcaClient


def get_examination_collection(request: Request) -> Collection:
    return request.app.state.mongo_client.get_default_database()["examination"]


ExaminationCollectionDep = Annotated[Collection, Depends(get_examination_collection)]


def get_ica_client(request: Request) -> IcaClient:
    return request.app.state.ica_client


IcaClientDep = Annotated[IcaClient, Depends(get_ica_client)]


def get_async_collection(request: Request) -> AsyncCollection:
    """Async PyMongo collection — use this in new async routes instead of ExaminationCollectionDep."""
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["examination"]


AsyncCollectionDep = Annotated[AsyncCollection, Depends(get_async_collection)]


def get_async_simplified_collection(request: Request) -> AsyncCollection:
    client: AsyncMongoClient = request.app.state.async_mongo_client
    return client.get_default_database()["examination_simplified"]


AsyncSimplifiedCollectionDep = Annotated[
    AsyncCollection, Depends(get_async_simplified_collection)
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
