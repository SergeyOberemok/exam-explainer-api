"""MongoDB access for `/v2/game`: the composed `game_question` documents (`compose.py`).

A `game_question` document:

    {
      "question_id": str,                     # the examination _id as a string (unique)
      "held": bool,                           # held for manual processing: not served
      "section", "assessment_type", "scenario", "stage": str,
      "order": int,                           # position inside the chapter's stage
      "objectives": [str], "difficulty": int,
      "count": int,                           # number of answers
      "public": GameCard,                     # sent before answering
      "key": GameCardKey,                     # kept in the round entry until answered
      "text_hash", "input_key": str, "updated_at": datetime,
    }

Rounds are the same `game_rounds` documents as v1 (`RoundRepository`); their entries hold
`correct`, `count` and the `key`.
"""

import asyncio

from bson import ObjectId
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.scenario_catalog import BY_KEY, STAGE_KEYS

SERVED: dict = {"held": False}
# What a card query never needs to read back.
_NO_KEY = {"key": 0}


def _position(keys: tuple[str, ...], key: object) -> int:
    return keys.index(key) if key in keys else len(keys)


def story_order(doc: dict) -> tuple:
    """Where a document stands in the story: chapter, stage, `order`, then question id
    so the order is stable."""
    return (
        _position(tuple(BY_KEY), doc.get("scenario")),
        _position(STAGE_KEYS, doc.get("stage")),
        doc.get("order") or 0,
        str(doc.get("question_id")),
    )


def card_query(filters: dict, exclude: list[str] = ()) -> dict:
    """The served documents a round's filters allow (empty strings mean "any"), leaving
    out the question ids in `exclude`."""
    query: dict = dict(SERVED)
    for field in ("section", "assessment_type", "scenario", "stage"):
        if filters.get(field):
            query[field] = filters[field]
    if filters.get("objective"):
        query["objectives"] = filters["objective"]
    if exclude:
        query["question_id"] = {"$nin": list(exclude)}
    return query


class CardRepository:
    """Reads `game_question`. `examination` is only asked whether a question exists, so
    adding an unknown one is a 404 and an unserved one a 422."""

    def __init__(self, cards: AsyncCollection, examination: AsyncCollection | None = None) -> None:
        self.cards = cards
        self.examination = examination

    async def _fetch(self, query: dict, size: int, shuffle: bool, key: bool) -> list[dict]:
        pipeline: list[dict] = [{"$match": query}]
        if shuffle:
            pipeline.append({"$sample": {"size": size}})
        else:
            pipeline += [{"$sort": {"question_id": 1}}, {"$limit": size}]
        if not key:
            pipeline.append({"$project": _NO_KEY})
        cursor = await self.cards.aggregate(pipeline)
        return await cursor.to_list()

    async def sample(self, query: dict, size: int) -> list[dict]:
        """Up to `size` random documents, with their keys (for a round)."""
        return await self._fetch(query, size, shuffle=True, key=True)

    async def in_story_order(self, query: dict) -> list[dict]:
        """Every matching document in story order (a story round's pool is one chapter or
        stage)."""
        return sorted(await self.cards.find(query).to_list(), key=story_order)

    async def browse(self, query: dict, size: int, shuffle: bool) -> tuple[list[dict], int]:
        """Up to `size` documents without their keys, and how many match in all."""
        return await asyncio.gather(
            self._fetch(query, size, shuffle, key=False), self.cards.count_documents(query)
        )

    async def count(self, query: dict) -> int:
        return await self.cards.count_documents(query)

    async def get(self, question_id: str) -> dict | None:
        """A served document."""
        return await self.cards.find_one({"question_id": question_id, **SERVED})

    async def get_many(self, question_ids: list[str]) -> dict[str, dict]:
        """Documents by question id, served or not (a stored batch is shown again as it
        was handed out)."""
        if not question_ids:
            return {}
        cursor = self.cards.find({"question_id": {"$in": list(question_ids)}}, _NO_KEY)
        return {d["question_id"]: d for d in await cursor.to_list()}

    async def served_ids(self) -> list[str]:
        return [str(i) for i in await self.cards.distinct("question_id", SERVED)]

    async def question_exists(self, question_id: str) -> bool:
        if self.examination is None or not ObjectId.is_valid(question_id):
            return False
        return await self.examination.find_one({"_id": ObjectId(question_id)}, {"_id": 1}) is not None

    async def sections(self) -> list[str]:
        return await self.cards.distinct("section", SERVED)

    async def summaries(self) -> list[dict]:
        """The filter fields of every served document (for the objective and story
        counts)."""
        cursor = self.cards.find(
            SERVED, {"question_id": 1, "objectives": 1, "scenario": 1, "stage": 1}
        )
        return await cursor.to_list()
