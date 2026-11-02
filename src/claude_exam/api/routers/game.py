import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncSimplifiedCollectionDep,
    QuestionObjectIdDep,
)
from claude_exam.api.schemas import (
    GameAnswer,
    GameQuestion,
    GameQuestionList,
    GameSectionList,
)

# Backend for the Unity game. Responses are flat, null-free objects (lists are
# wrapped in `items`) because UnityWebRequest + JsonUtility cannot parse a
# top-level JSON array or null values.
router = APIRouter(prefix="/game", tags=["game"])


def letter_to_index(letter: object) -> int:
    """'a' -> 0, 'b' -> 1, ...; -1 for anything that isn't a single letter
    (correct_response stores option letters, but stored data may be malformed)."""
    s = letter.strip().lower() if isinstance(letter, str) else ""
    return ord(s) - ord("a") if len(s) == 1 else -1


def to_game_question(doc: dict, *, simplified: bool = False) -> GameQuestion:
    """Map an `examination` doc, or an `examination_simplified` doc when
    `simplified` (same prompt/correct_response layout, id in `question_id`)."""
    prompt = doc.get("prompt") or {}
    answers = prompt.get("answers") or []
    feedbacks = prompt.get("feedbacks") or []
    correct = sorted(
        {
            i
            for i in map(letter_to_index, doc.get("correct_response") or [])
            if 0 <= i < len(answers)
        }
    )
    return GameQuestion(
        id=str(doc["question_id"] if simplified else doc["_id"]),
        simplified=simplified,
        section=doc.get("section") or "",
        assessment_type=doc.get("assessment_type") or "",
        question=prompt.get("question") or "",
        explanation=prompt.get("explanation") or "",
        answers=[
            GameAnswer(
                index=i,
                text=text or "",
                is_correct=i in correct,
                feedback=(feedbacks[i] or "") if i < len(feedbacks) else "",
            )
            for i, text in enumerate(answers)
        ],
        correct_indices=correct,
    )


async def fetch_game_batch(
    collection: AsyncCollection,
    count: int,
    section: str | None,
    assessment_type: str | None,
    shuffle: bool,
    *,
    simplified: bool = False,
) -> GameQuestionList:
    query: dict = {}
    if section:
        query["section"] = section
    if assessment_type:
        query["assessment_type"] = assessment_type

    pipeline: list[dict] = [{"$match": query}]
    if shuffle:
        pipeline.append({"$sample": {"size": count}})
    else:
        # Stable order, so the same call returns the same questions.
        pipeline += [{"$sort": {"_id": 1}}, {"$limit": count}]

    async def fetch_docs() -> list[dict]:
        cursor = await collection.aggregate(pipeline)
        return await cursor.to_list()

    docs, total = await asyncio.gather(fetch_docs(), collection.count_documents(query))

    items: list[GameQuestion] = []
    seen: set[str] = set()
    for doc in docs:
        question = to_game_question(doc, simplified=simplified)
        # $sample may return the same document twice.
        if question.id not in seen:
            seen.add(question.id)
            items.append(question)
    return GameQuestionList(items=items, total=total)


@router.get("/questions")
async def list_game_questions(
    collection: AsyncCollectionDep,
    count: Annotated[int, Query(ge=1, le=50)] = 10,
    section: Annotated[str | None, Query(max_length=200)] = None,
    assessment_type: Annotated[str | None, Query(max_length=100)] = None,
    shuffle: bool = True,
) -> GameQuestionList:
    """A batch of questions (random by default) for one game round."""
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
