import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from pymongo import AsyncMongoClient

from claude_exam.api.batch import STALE_JOB_SECONDS
from claude_exam.api.config import get_settings
from claude_exam.api.errors import install_error_handlers
from claude_exam.api.ica import IcaClient
from claude_exam.api.indexes import ensure_indexes
from claude_exam.api.routers.evaluate import router as evaluate_router
from claude_exam.api.routers.examination import router as examination_router
from claude_exam.api.routers.game_v2 import router as game_v2_router
from claude_exam.api.routers.game_v2 import search_router as search_v2_router
from claude_exam.api.routers.ica import router as ica_router
from claude_exam.api.routers.questions import router as questions_router
from claude_exam.api.routers.search import router as search_router

API_VERSION_PREFIX = "/v1"
# /v2/game serves the composed questions (`compose.py`). /v1 is left with the study search
# (`/v1/search/questions`), which the vault skills call.
API_V2_PREFIX = "/v2"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.api_key = settings.api_key
    mongo: AsyncMongoClient = AsyncMongoClient(settings.database_url)
    app.state.async_mongo_client = mongo
    db = mongo.get_default_database()
    await ensure_indexes(db)
    # Background jobs die with their process. Fail only jobs whose heartbeat has
    # gone quiet, so a worker starting up doesn't kill jobs live in another one.
    now = datetime.now(timezone.utc)
    await db["simplification_jobs"].update_many(
        {
            "status": {"$in": ["queued", "running"]},
            "heartbeat_at": {"$lt": now - timedelta(seconds=STALE_JOB_SECONDS)},
        },
        {"$set": {"status": "failed", "finished_at": now}},
    )
    # Serializes simplification jobs within this process (limits ICA 429s).
    app.state.simplification_lock = asyncio.Semaphore(1)
    app.state.pipeline_concurrency = settings.pipeline_concurrency
    ica_client = IcaClient(settings.ica_url, settings.ica_api_key)
    app.state.ica_client = ica_client
    try:
        yield
    finally:
        await mongo.close()
        await ica_client.aclose()


app = FastAPI(lifespan=lifespan)
install_error_handlers(app)

# The game and search routes are versioned: a shipped Unity build can't be updated, so a
# breaking change gets a new prefix.
app.include_router(search_router, prefix=API_VERSION_PREFIX)
app.include_router(game_v2_router, prefix=API_V2_PREFIX)
app.include_router(search_v2_router, prefix=API_V2_PREFIX)
app.include_router(examination_router)
app.include_router(evaluate_router)
app.include_router(ica_router)
app.include_router(questions_router)

# The prompt evaluation page: one static file over the /evaluate routes.
app.frontend("/evaluate/ui", directory=Path(__file__).parent / "static" / "evaluate")
