import asyncio

from fastapi import APIRouter, Depends, HTTPException

from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    PaginationDep,
    QuestionFiltersDep,
    QuestionObjectIdDep,
)
from claude_exam.api.schemas import (
    Question,
    QuestionPage,
    QuestionSummary,
)
from claude_exam.api.security import require_api_key

# Returns the answer key and feedback, so it needs the API key when one is configured.
router = APIRouter(
    prefix="/questions", tags=["questions"], dependencies=[Depends(require_api_key)]
)


def to_summary(doc: dict) -> QuestionSummary:
    return QuestionSummary(
        id=str(doc["_id"]),
        section=doc.get("section", ""),
        assessment_type=doc.get("assessment_type", ""),
        source=doc.get("source", ""),
        question=(doc.get("prompt") or {}).get("question", ""),
    )


def to_question(doc: dict) -> Question:
    prompt = doc.get("prompt") or {}
    return Question(
        **to_summary(doc).model_dump(),
        answers=prompt.get("answers", []),
        feedbacks=prompt.get("feedbacks", []),
        correct_response=doc.get("correct_response", []),
        explanation=prompt.get("explanation", ""),
        related_lectures=doc.get("related_lectures", []),
    )


@router.get("/")
async def list_questions(
    collection: AsyncCollectionDep,
    pagination: PaginationDep,
    filters: QuestionFiltersDep,
) -> QuestionPage:
    query = filters.to_mongo()
    cursor = (
        collection.find(query)
        .sort("_id", 1)
        .skip(pagination.offset)
        .limit(pagination.limit)
    )
    docs, total = await asyncio.gather(cursor.to_list(), collection.count_documents(query))
    return QuestionPage(
        items=[to_summary(doc) for doc in docs],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get("/{question_id}")
async def get_question(
    object_id: QuestionObjectIdDep, collection: AsyncCollectionDep
) -> Question:
    doc = await collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    return to_question(doc)
