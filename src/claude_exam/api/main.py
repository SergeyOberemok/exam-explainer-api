import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from pymongo import AsyncMongoClient, MongoClient

from claude_exam.api.batch import STALE_JOB_SECONDS
from claude_exam.api.config import get_settings
from claude_exam.api.ica import IcaClient
from claude_exam.api.routers.examination import router as examination_router
from claude_exam.api.routers.game import ROUND_TTL_SECONDS
from claude_exam.api.routers.game import router as game_router
from claude_exam.api.routers.ica import router as ica_router
from claude_exam.api.routers.questions import router as questions_router
from claude_exam.api.routers.search import router as search_router
from claude_exam.api.search import (
    TEXT_INDEX_KEYS,
    TEXT_INDEX_NAME,
    TEXT_INDEX_WEIGHTS,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Sync client for the remaining sync routes (questions list/get).
    # TODO: migrate those to async and drop this.
    client = MongoClient(settings.database_url)
    app.state.mongo_client = client
    # PyMongo Async client for new async routes (e.g. ICA → MongoDB upsert).
    async_mongo_client: AsyncMongoClient = AsyncMongoClient(settings.database_url)
    app.state.async_mongo_client = async_mongo_client
    # One simplified record per question; makes the simplify upsert race-safe.
    await async_mongo_client.get_default_database()[
        "examination_simplified"
    ].create_index("question_id", unique=True)
    # Game endpoints filter by these fields; avoids collection scans.
    db = async_mongo_client.get_default_database()
    for name in ("examination", "examination_simplified"):
        await db[name].create_index([("section", 1), ("assessment_type", 1)])
    # Weighted text index behind /search/questions (a collection allows only one).
    await db["examination"].create_index(
        TEXT_INDEX_KEYS,
        name=TEXT_INDEX_NAME,
        weights=TEXT_INDEX_WEIGHTS,
        default_language="english",
    )
    # Rounds are only needed while a game is played; Mongo drops them after a day.
    await db["game_rounds"].create_index(
        "created_at", expireAfterSeconds=ROUND_TTL_SECONDS
    )
    # Background jobs die with their process. Fail only jobs whose heartbeat has
    # gone quiet, so a worker starting up doesn't kill jobs live in another one.
    now = datetime.now(timezone.utc)
    await async_mongo_client.get_default_database()[
        "simplification_jobs"
    ].update_many(
        {
            "status": {"$in": ["queued", "running"]},
            "heartbeat_at": {"$lt": now - timedelta(seconds=STALE_JOB_SECONDS)},
        },
        {"$set": {"status": "failed", "finished_at": now}},
    )
    # Serializes simplification jobs within this process (limits ICA 429s).
    app.state.simplification_lock = asyncio.Semaphore(1)
    ica_client = IcaClient(settings.ica_url, settings.ica_api_key)
    app.state.ica_client = ica_client
    try:
        yield
    finally:
        client.close()
        await async_mongo_client.close()
        await ica_client.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(examination_router)
app.include_router(game_router)
app.include_router(ica_router)
app.include_router(questions_router)
app.include_router(search_router)
