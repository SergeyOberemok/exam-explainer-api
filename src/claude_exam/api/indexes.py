"""Indexes the API relies on, created at startup.

Creating an index that already exists with the same spec is a no-op. An index whose options
changed (e.g. new text weights) makes MongoDB refuse with IndexOptionsConflict /
IndexKeySpecsConflict; that is logged instead of stopping the app, and the old index keeps
serving until it is dropped by hand (`db.examination.dropIndex("examination_text")`) and
the app restarts.
"""

import logging

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import OperationFailure

from claude_exam.api.game.domain import ROUND_TTL_SECONDS
from claude_exam.api.search import TEXT_INDEX_KEYS, TEXT_INDEX_NAME, TEXT_INDEX_WEIGHTS

logger = logging.getLogger(__name__)


async def ensure_indexes(db: AsyncDatabase) -> None:
    specs = [
        # One simplified record per question; makes the simplify upsert race-safe.
        ("examination_simplified", "question_id", {"unique": True}),
        # Game endpoints filter by these fields; avoids collection scans.
        ("examination", [("section", 1), ("assessment_type", 1)], {}),
        ("examination_simplified", [("section", 1), ("assessment_type", 1)], {}),
        # Weighted text index behind /search (a collection allows only one).
        (
            "examination",
            TEXT_INDEX_KEYS,
            {"name": TEXT_INDEX_NAME, "weights": TEXT_INDEX_WEIGHTS, "default_language": "english"},
        ),
        # Rounds are only needed while a game is played; Mongo drops them after a day.
        ("game_rounds", "created_at", {"expireAfterSeconds": ROUND_TTL_SECONDS}),
    ]
    for collection, keys, options in specs:
        try:
            await db[collection].create_index(keys, **options)
        except OperationFailure as exc:
            logger.error(
                "Index %s on %s not created (%s); drop the old index to apply the new spec.",
                options.get("name", keys),
                collection,
                exc.details.get("codeName") if exc.details else exc,
            )
