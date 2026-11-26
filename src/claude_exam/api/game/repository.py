"""MongoDB access for the game rounds. The service decides; this class only reads and writes.

A `game_rounds` document:

    {
      "created_at": datetime,                 # TTL index: dropped after ROUND_TTL_SECONDS
      "filters": {"section": str, "assessment_type": str, "objective": str,
                  "scenario": str, "stage": str, "ordered": bool},
      "questions": {question_id: {correct, count, key}},  # never sent before answering
      "question_ids": [question_id, ...],     # every question in the round, in the order added
      "batches": [[question_id, ...], ...],   # batch 0 is the start; each later one is one /batches call
      "answers": {question_id: answer_index},
      "eliminated": {question_id: [index, index]},
      "fifty_fifty_remaining": int,
    }

`question_id` keys are validated ObjectId hex strings (`GameQuestionId`), so they are safe
inside field paths.
"""

from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.domain import FIFTY_FIFTY_CHARGES, MAX_ROUND_QUESTIONS


class RoundRepository:
    """Reads and writes `game_rounds`. Every write is one conditional update, so concurrent
    requests for the same round can't both apply; `None`/`False` means the condition failed
    and the caller reads the round to find out why."""

    def __init__(self, rounds: AsyncCollection) -> None:
        self.rounds = rounds

    async def create(self, filters: dict, entries: dict[str, dict]) -> str:
        ids = list(entries)
        result = await self.rounds.insert_one(
            {
                "created_at": datetime.now(timezone.utc),
                "filters": filters,
                "questions": entries,
                "question_ids": ids,
                "batches": [ids],
                "answers": {},
                "eliminated": {},
                "fifty_fifty_remaining": FIFTY_FIFTY_CHARGES,
            }
        )
        return str(result.inserted_id)

    async def get_for_question(self, round_id: ObjectId, question_id: str) -> dict | None:
        """The round's state for one question (and the 50/50 charges); None if no round."""
        return await self.rounds.find_one(
            {"_id": round_id},
            {
                f"questions.{question_id}": 1,
                f"answers.{question_id}": 1,
                f"eliminated.{question_id}": 1,
                "fifty_fifty_remaining": 1,
            },
        )

    async def get_layout(self, round_id: ObjectId) -> dict | None:
        """Which questions the round holds, its batches and filters, without the answer keys."""
        return await self.rounds.find_one(
            {"_id": round_id}, {"question_ids": 1, "batches": 1, "filters": 1}
        )

    async def get_entries(self, round_id: ObjectId, question_ids: list[str]) -> dict | None:
        """The stored entries of some questions; None if there is no round."""
        doc = await self.rounds.find_one(
            {"_id": round_id}, {f"questions.{i}": 1 for i in question_ids} or {"_id": 1}
        )
        return None if doc is None else (doc.get("questions") or {})

    async def record_answer(
        self, round_id: ObjectId, question_id: str, answer_index: int
    ) -> dict | None:
        """Records the answer if the question is in the round, unanswered, the index is in
        range and not eliminated. Returns the question's entry, or None."""
        doc = await self.rounds.find_one_and_update(
            {
                "_id": round_id,
                f"questions.{question_id}.count": {"$gt": answer_index},
                f"answers.{question_id}": {"$exists": False},
                f"eliminated.{question_id}": {"$ne": answer_index},
            },
            {"$set": {f"answers.{question_id}": answer_index}},
            projection={f"questions.{question_id}": 1},
        )
        return None if doc is None else doc["questions"][question_id]

    async def spend_fifty_fifty(
        self, round_id: ObjectId, question_id: str, eliminated: list[int]
    ) -> int | None:
        """Stores `eliminated` and uses a charge if the question is unanswered, not yet
        50/50'd and a charge is left. Returns the charges left, or None."""
        doc = await self.rounds.find_one_and_update(
            {
                "_id": round_id,
                f"questions.{question_id}": {"$exists": True},
                f"answers.{question_id}": {"$exists": False},
                f"eliminated.{question_id}": {"$exists": False},
                "fifty_fifty_remaining": {"$gt": 0},
            },
            {
                "$set": {f"eliminated.{question_id}": eliminated},
                "$inc": {"fifty_fifty_remaining": -1},
            },
            projection={"fifty_fifty_remaining": 1},
            return_document=ReturnDocument.AFTER,
        )
        return None if doc is None else doc["fifty_fifty_remaining"]

    async def add_question(self, round_id: ObjectId, question_id: str, entry: dict) -> bool:
        """Adds one question if the round doesn't have it and isn't full."""
        doc = await self.rounds.find_one_and_update(
            {
                "_id": round_id,
                f"questions.{question_id}": {"$exists": False},
                f"question_ids.{MAX_ROUND_QUESTIONS - 1}": {"$exists": False},
            },
            {
                "$set": {f"questions.{question_id}": entry},
                "$push": {"question_ids": question_id},
            },
            projection={"_id": 1},
        )
        return doc is not None

    async def add_batch(
        self, round_id: ObjectId, batch_index: int, entries: dict[str, dict]
    ) -> bool:
        """Adds `entries` as batch `batch_index` if that is the next batch, none of them is in
        the round yet and they fit."""
        ids = list(entries)
        doc = await self.rounds.find_one_and_update(
            {
                "_id": round_id,
                "batches": {"$size": batch_index},
                "question_ids": {"$nin": ids},
                f"question_ids.{MAX_ROUND_QUESTIONS - len(ids)}": {"$exists": False},
            },
            {
                "$set": {f"questions.{i}": e for i, e in entries.items()},
                "$push": {"question_ids": {"$each": ids}, "batches": ids},
            },
            projection={"_id": 1},
        )
        return doc is not None
