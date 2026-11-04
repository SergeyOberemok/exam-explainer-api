import asyncio
import random
from datetime import datetime, timezone
from typing import Annotated, NoReturn

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncRoundsCollectionDep,
    AsyncSimplifiedCollectionDep,
    QuestionObjectIdDep,
    RoundObjectIdDep,
)
from claude_exam.api.schemas import (
    GameAnswer,
    GameAnswerRequest,
    GameAnswerResult,
    GameFiftyFiftyRequest,
    GameFiftyFiftyResult,
    GameQuestion,
    GameQuestionList,
    GameReview,
    GameRound,
    GameRoundQuestion,
    GameRoundQuestionRequest,
    GameRoundRequest,
    GameSectionList,
)

# Backend for the Unity game. Responses are flat, null-free objects (lists are
# wrapped in `items`) because UnityWebRequest + JsonUtility cannot parse a
# top-level JSON array or null values.
#
# The game plays rounds (`POST /game/rounds`). Questions go out without their
# correct answer, feedback or explanation; the round document keeps those, and
# the game gets them for one question by answering it, once.
router = APIRouter(prefix="/game", tags=["game"])

ROUND_TTL_SECONDS = 24 * 60 * 60
FIFTY_FIFTY_CHARGES = 2
FIFTY_FIFTY_ELIMINATES = 2
# The game has one card and one key per answer.
MAX_ANSWERS = 4

# Pre-filter for pick-one questions the game can show; `is_playable` has the final say.
PLAYABLE_QUERY: dict = {
    "correct_response": {"$size": 1},
    "prompt.answers.1": {"$exists": True},
    f"prompt.answers.{MAX_ANSWERS}": {"$exists": False},
}


def letter_to_index(letter: object) -> int:
    """'a' -> 0, 'b' -> 1, ...; -1 for anything that isn't a single letter
    (correct_response stores option letters, but stored data may be malformed)."""
    s = letter.strip().lower() if isinstance(letter, str) else ""
    return ord(s) - ord("a") if len(s) == 1 else -1


def answers_of(doc: dict) -> list:
    return (doc.get("prompt") or {}).get("answers") or []


def correct_indices(doc: dict) -> list[int]:
    count = len(answers_of(doc))
    return sorted(
        {
            i
            for i in map(letter_to_index, doc.get("correct_response") or [])
            if 0 <= i < count
        }
    )


def is_playable(doc: dict) -> bool:
    """Pick-one: 2..MAX_ANSWERS answers and exactly one valid correct answer."""
    return 2 <= len(answers_of(doc)) <= MAX_ANSWERS and len(correct_indices(doc)) == 1


def simplified_matches(original: dict, simplified: dict) -> bool:
    """A simplified version is only usable if it keeps the answers and the correct one."""
    return len(answers_of(simplified)) == len(answers_of(original)) and correct_indices(
        simplified
    ) == correct_indices(original)


def to_game_question(doc: dict, *, simplified: bool = False) -> GameQuestion:
    """Map an `examination` doc, or an `examination_simplified` doc when
    `simplified` (same prompt/correct_response layout, id in `question_id`)."""
    prompt = doc.get("prompt") or {}
    return GameQuestion(
        id=str(doc["question_id"] if simplified else doc["_id"]),
        simplified=simplified,
        section=doc.get("section") or "",
        assessment_type=doc.get("assessment_type") or "",
        question=prompt.get("question") or "",
        answers=[
            GameAnswer(index=i, text=text or "") for i, text in enumerate(answers_of(doc))
        ],
    )


def to_review(doc: dict) -> GameReview:
    """Explanation and one feedback per answer (empty where the source has none)."""
    prompt = doc.get("prompt") or {}
    feedbacks = prompt.get("feedbacks") or []
    return GameReview(
        explanation=prompt.get("explanation") or "",
        feedbacks=[
            (feedbacks[i] or "") if i < len(feedbacks) else ""
            for i in range(len(answers_of(doc)))
        ],
    )


def usable_simplified(original: dict, simplified: dict | None) -> dict | None:
    """`simplified` if the game can offer it for `original`, else None."""
    if simplified is not None and simplified_matches(original, simplified):
        return simplified
    return None


def round_entry(doc: dict, simplified: dict | None) -> dict:
    """What a round document keeps for one playable question: its answer key and reviews."""
    return {
        "correct": correct_indices(doc)[0],
        "count": len(answers_of(doc)),
        "original": to_review(doc).model_dump(),
        "simplified": to_review(simplified).model_dump() if simplified else None,
    }


def filter_query(section: str | None, assessment_type: str | None) -> dict:
    query: dict = {}
    if section:
        query["section"] = section
    if assessment_type:
        query["assessment_type"] = assessment_type
    return query


async def sample_docs(
    collection: AsyncCollection, query: dict, count: int, shuffle: bool
) -> tuple[list[dict], int]:
    pipeline: list[dict] = [{"$match": query}]
    if shuffle:
        pipeline.append({"$sample": {"size": count}})
    else:
        # Stable order, so the same call returns the same questions.
        pipeline += [{"$sort": {"_id": 1}}, {"$limit": count}]

    async def fetch_docs() -> list[dict]:
        cursor = await collection.aggregate(pipeline)
        return await cursor.to_list()

    return await asyncio.gather(fetch_docs(), collection.count_documents(query))


def dedupe(docs: list[dict], key: str) -> list[dict]:
    """$sample may return the same document twice."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        doc_id = str(doc[key])
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc)
    return unique


async def fetch_game_batch(
    collection: AsyncCollection,
    count: int,
    section: str | None,
    assessment_type: str | None,
    shuffle: bool,
    *,
    simplified: bool = False,
) -> GameQuestionList:
    docs, total = await sample_docs(
        collection, filter_query(section, assessment_type), count, shuffle
    )
    key = "question_id" if simplified else "_id"
    return GameQuestionList(
        items=[to_game_question(d, simplified=simplified) for d in dedupe(docs, key)],
        total=total,
    )


@router.post("/rounds", status_code=201)
async def create_game_round(
    body: GameRoundRequest,
    collection: AsyncCollectionDep,
    simplified_collection: AsyncSimplifiedCollectionDep,
    rounds: AsyncRoundsCollectionDep,
) -> GameRound:
    """Start a round: random playable questions and their simplified versions, in one
    response. The answer key stays in the round document."""
    query = {**filter_query(body.section, body.assessment_type), **PLAYABLE_QUERY}
    docs, total = await sample_docs(collection, query, body.count, shuffle=True)
    originals = [d for d in dedupe(docs, "_id") if is_playable(d)]

    ids = [str(d["_id"]) for d in originals]
    cursor = simplified_collection.find({"question_id": {"$in": ids}})
    simplified_by_id = {str(d["question_id"]): d for d in await cursor.to_list()}

    questions: dict[str, dict] = {}
    simplified_items: list[GameQuestion] = []
    for doc in originals:
        question_id = str(doc["_id"])
        simplified = usable_simplified(doc, simplified_by_id.get(question_id))
        questions[question_id] = round_entry(doc, simplified)
        if simplified is not None:
            simplified_items.append(to_game_question(simplified, simplified=True))

    result = await rounds.insert_one(
        {
            "created_at": datetime.now(timezone.utc),
            "questions": questions,
            "answers": {},
            "eliminated": {},
            "fifty_fifty_remaining": FIFTY_FIFTY_CHARGES,
        }
    )
    return GameRound(
        round_id=str(result.inserted_id),
        items=[to_game_question(d) for d in originals],
        simplified=simplified_items,
        total=total,
        fifty_fifty_charges=FIFTY_FIFTY_CHARGES,
    )


def raise_round_error(
    round_doc: dict | None, question_id: str, answer_index: int | None = None
) -> NoReturn:
    """Explain why a round update didn't apply; `answer_index` is set for an answer."""
    if round_doc is None:
        raise HTTPException(status_code=404, detail="round not found")
    question = (round_doc.get("questions") or {}).get(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question is not in this round")
    if question_id in (round_doc.get("answers") or {}):
        raise HTTPException(status_code=409, detail="question already answered")
    if answer_index is not None:
        if answer_index >= question["count"]:
            raise HTTPException(status_code=422, detail="answer index out of range")
        if answer_index in (round_doc.get("eliminated") or {}).get(question_id, []):
            raise HTTPException(status_code=422, detail="answer was eliminated by 50/50")
    else:
        if round_doc.get("fifty_fifty_remaining", 0) <= 0:
            raise HTTPException(status_code=409, detail="no 50/50 charges left")
        if question["count"] - 1 < FIFTY_FIFTY_ELIMINATES:
            raise HTTPException(status_code=409, detail="too few answers for 50/50")
    # The state changed between the write and this read; the client can retry.
    raise HTTPException(status_code=409, detail="round changed, try again")


def round_projection(question_id: str) -> dict:
    return {
        f"questions.{question_id}": 1,
        f"answers.{question_id}": 1,
        f"eliminated.{question_id}": 1,
        "fifty_fifty_remaining": 1,
    }


@router.post("/rounds/{round_id}/answers")
async def answer_game_round_question(
    body: GameAnswerRequest,
    round_oid: RoundObjectIdDep,
    rounds: AsyncRoundsCollectionDep,
) -> GameAnswerResult:
    """Answer one question of a round, once; sending the same answer again repeats the
    result. Returns whether it was right, the correct answer, and both versions'
    feedback and explanation."""
    question_id, index = body.question_id, body.answer_index
    # One atomic write, so two concurrent answers can't both count.
    doc = await rounds.find_one_and_update(
        {
            "_id": round_oid,
            f"questions.{question_id}.count": {"$gt": index},
            f"answers.{question_id}": {"$exists": False},
            f"eliminated.{question_id}": {"$ne": index},
        },
        {"$set": {f"answers.{question_id}": index}},
        projection={f"questions.{question_id}": 1},
    )
    if doc is None:
        doc = await rounds.find_one({"_id": round_oid}, round_projection(question_id))
        # The same answer again is a retry after a lost response: repeat the result.
        if (doc or {}).get("answers", {}).get(question_id) != index:
            raise_round_error(doc, question_id, index)

    question = doc["questions"][question_id]
    simplified = question.get("simplified")
    return GameAnswerResult(
        question_id=question_id,
        answer_index=index,
        is_correct=index == question["correct"],
        correct_index=question["correct"],
        original=GameReview(**question["original"]),
        has_simplified=simplified is not None,
        simplified=GameReview(**simplified) if simplified else GameReview(),
    )


@router.post("/rounds/{round_id}/fifty-fifty")
async def use_game_round_fifty_fifty(
    body: GameFiftyFiftyRequest,
    round_oid: RoundObjectIdDep,
    rounds: AsyncRoundsCollectionDep,
) -> GameFiftyFiftyResult:
    """Remove two random wrong answers from an unanswered question of the round.
    Once per question (asking again repeats the result), and the round has
    `fifty_fifty_charges` uses."""
    question_id = body.question_id
    round_doc = await rounds.find_one({"_id": round_oid}, round_projection(question_id))
    question = ((round_doc or {}).get("questions") or {}).get(question_id)
    already = (round_doc or {}).get("eliminated", {}).get(question_id)
    if question is not None and already is not None:
        # A retry after a lost response: repeat the result without using a charge.
        return GameFiftyFiftyResult(
            question_id=question_id,
            eliminated_indices=already,
            fifty_fifty_remaining=round_doc.get("fifty_fifty_remaining", 0),
        )
    usable = (
        question is not None
        and question_id not in (round_doc.get("answers") or {})
        and round_doc.get("fifty_fifty_remaining", 0) > 0
        and question["count"] - 1 >= FIFTY_FIFTY_ELIMINATES
    )
    if not usable:
        raise_round_error(round_doc, question_id)

    wrong = [i for i in range(question["count"]) if i != question["correct"]]
    eliminated = sorted(random.sample(wrong, FIFTY_FIFTY_ELIMINATES))
    # Conditional write: a concurrent answer or 50/50 makes it miss instead of double-spending.
    updated = await rounds.find_one_and_update(
        {
            "_id": round_oid,
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
    if updated is None:
        raise_round_error(
            await rounds.find_one({"_id": round_oid}, round_projection(question_id)),
            question_id,
        )

    return GameFiftyFiftyResult(
        question_id=question_id,
        eliminated_indices=eliminated,
        fifty_fifty_remaining=updated["fifty_fifty_remaining"],
    )


@router.post("/rounds/{round_id}/questions")
async def add_game_round_question(
    body: GameRoundQuestionRequest,
    round_oid: RoundObjectIdDep,
    collection: AsyncCollectionDep,
    simplified_collection: AsyncSimplifiedCollectionDep,
    rounds: AsyncRoundsCollectionDep,
) -> GameRoundQuestion:
    """Add one question (e.g. a search result) to a round, so it can be answered there.
    Adding a question the round already has changes nothing and returns it again.
    Like the round itself, the response carries no answer key."""
    question_id = body.question_id
    doc = await collection.find_one({"_id": ObjectId(question_id)})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    if not is_playable(doc):
        raise HTTPException(status_code=422, detail="question is not playable")
    simplified = usable_simplified(
        doc, await simplified_collection.find_one({"question_id": question_id})
    )

    # Conditional write: a question already in the round keeps its stored key.
    added = await rounds.find_one_and_update(
        {"_id": round_oid, f"questions.{question_id}": {"$exists": False}},
        {"$set": {f"questions.{question_id}": round_entry(doc, simplified)}},
        projection={"_id": 1},
    )
    if added is None:
        round_doc = await rounds.find_one(
            {"_id": round_oid}, {f"questions.{question_id}": 1}
        )
        if round_doc is None:
            raise HTTPException(status_code=404, detail="round not found")
        # Offer the simplified text only if the stored entry can review it.
        if (round_doc.get("questions") or {}).get(question_id, {}).get("simplified") is None:
            simplified = None

    return GameRoundQuestion(
        item=to_game_question(doc),
        simplified=[to_game_question(simplified, simplified=True)] if simplified else [],
    )


@router.get("/questions")
async def list_game_questions(
    collection: AsyncCollectionDep,
    count: Annotated[int, Query(ge=1, le=50)] = 10,
    section: Annotated[str | None, Query(max_length=200)] = None,
    assessment_type: Annotated[str | None, Query(max_length=100)] = None,
    shuffle: bool = True,
) -> GameQuestionList:
    """A batch of questions (random by default), without answer keys. To play, start a round."""
    return await fetch_game_batch(collection, count, section, assessment_type, shuffle)


@router.get("/simplified-questions")
async def list_simplified_game_questions(
    simplified_collection: AsyncSimplifiedCollectionDep,
    count: Annotated[int, Query(ge=1, le=50)] = 10,
    section: Annotated[str | None, Query(max_length=200)] = None,
    assessment_type: Annotated[str | None, Query(max_length=100)] = None,
    shuffle: bool = True,
) -> GameQuestionList:
    """A batch of plain-language questions (only those already simplified)."""
    return await fetch_game_batch(
        simplified_collection, count, section, assessment_type, shuffle, simplified=True
    )


@router.get("/simplified-questions/{question_id}")
async def get_simplified_game_question(
    question_id: str, simplified_collection: AsyncSimplifiedCollectionDep
) -> GameQuestion:
    """Simplified version of one question; `question_id` is the id `/game/questions` returned."""
    doc = await simplified_collection.find_one({"question_id": question_id})
    if doc is None:
        # Distinct from "question not found" so the game can fall back to the raw question.
        raise HTTPException(status_code=404, detail="simplified question not found")
    return to_game_question(doc, simplified=True)


@router.get("/questions/{question_id}")
async def get_game_question(
    object_id: QuestionObjectIdDep, collection: AsyncCollectionDep
) -> GameQuestion:
    doc = await collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    return to_game_question(doc)


@router.get("/sections")
async def list_game_sections(
    collection: AsyncCollectionDep,
    simplified_collection: AsyncSimplifiedCollectionDep,
    simplified: bool = False,
) -> GameSectionList:
    """Distinct sections, so the game can offer a category choice. With
    `simplified=true` only sections that have simplified questions."""
    sections = await (simplified_collection if simplified else collection).distinct(
        "section"
    )
    return GameSectionList(items=sorted(s for s in sections if s))
