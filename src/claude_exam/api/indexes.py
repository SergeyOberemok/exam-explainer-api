"""Indexes the API relies on, created at startup.

Creating an index that already exists with the same spec is a no-op. An index whose options
changed (e.g. new text weights) makes MongoDB refuse with IndexOptionsConflict /
IndexKeySpecsConflict; that is logged instead of stopping the app, and the old index keeps
serving until it is dropped by hand (`db.examination.dropIndex("examination_text")`) and
the app restarts.
"""

import logging

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError, OperationFailure

from claude_exam.api.game.domain import ROUND_TTL_SECONDS
from claude_exam.api.scenario_catalog import text_hash
from claude_exam.api.search import TEXT_INDEX_KEYS, TEXT_INDEX_NAME, TEXT_INDEX_WEIGHTS

logger = logging.getLogger(__name__)


async def ensure_text_hashes(db: AsyncDatabase) -> int:
    """Give every stored question its `text_hash` (questions imported before the import
    stored one have none), so an import can tell what the bank already holds. Returns
    how many were filled in."""
    filled = 0
    async for doc in db["examination"].find({"text_hash": None}, {"prompt": 1}):
        try:
            await db["examination"].update_one(
                {"_id": doc["_id"]}, {"$set": {"text_hash": text_hash(doc)}}
            )
        except DuplicateKeyError:
            # The bank already holds this question under another id: leave this copy
            # without a hash (the unique sparse index ignores it) and say so, instead
            # of failing startup.
            logger.warning("question %s duplicates a stored question; text_hash not set", doc["_id"])
            continue
        filled += 1
    if filled:
        logger.info("text_hash added to %d stored questions", filled)
    return filled


async def ensure_indexes(db: AsyncDatabase) -> None:
    await ensure_text_hashes(db)
    specs = [
        # One stored question per text: the import's duplicate check, made race-safe.
        # Not created while the bank holds the same question twice (logged below).
        ("examination", "text_hash", {"unique": True, "sparse": True}),
        # One rephrase record per question; makes its upsert race-safe.
        ("examination_simplified", "question_id", {"unique": True}),
        # One objectives record per question; makes the tagging upsert race-safe.
        ("question_objectives", "question_id", {"unique": True}),
        # A round filtered by objective looks questions up by key.
        ("question_objectives", "objectives", {}),
        # One story record per question; makes the story upsert race-safe.
        ("question_story", "question_id", {"unique": True}),
        # A story round reads a chapter's stage in order.
        ("question_story", [("scenario", 1), ("stage", 1), ("order", 1)], {}),
        # One decomposition record per question; makes its upsert race-safe.
        ("question_decomposition", "question_id", {"unique": True}),
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
