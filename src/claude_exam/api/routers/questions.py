from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException

from claude_exam.api.dependencies import (
    AsyncCollectionDep,
    AsyncSimplifiedCollectionDep,
    ExaminationCollectionDep,
    IcaClientDep,
    PaginationDep,
    QuestionFiltersDep,
)
from claude_exam.api.schemas import (
    Question,
    QuestionPage,
    QuestionSummary,
    SimplifiedQuestion,
    SimplifyRequest,
)
from claude_exam.api.simplify import simplify_and_store

router = APIRouter(prefix="/questions", tags=["questions"])


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
def list_questions(
    collection: ExaminationCollectionDep,
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
    return QuestionPage(
        items=[to_summary(doc) for doc in cursor],
        total=collection.count_documents(query),
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get("/{question_id}")
def get_question(question_id: str, collection: ExaminationCollectionDep) -> Question:
    try:
        object_id = ObjectId(question_id)
    except InvalidId as exc:
        raise HTTPException(status_code=404, detail="question not found") from exc

    doc = collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")
    return to_question(doc)


@router.post("/{question_id}/simplify")
async def simplify_question(
    question_id: str,
    collection: AsyncCollectionDep,
    simplified: AsyncSimplifiedCollectionDep,
    ica: IcaClientDep,
    body: SimplifyRequest | None = None,
    force: bool = False,
) -> SimplifiedQuestion:
    """
    Simplify one stored question through ICA and upsert the result into
    `examination_simplified`, keyed by `question_id` (the examination `_id`).
    An already stored result is returned as-is (no ICA call) unless `force=true`,
    which replaces it and keeps `created_at`.
    """
    body = body or SimplifyRequest()
    try:
        object_id = ObjectId(question_id)
    except InvalidId as exc:
        raise HTTPException(status_code=404, detail="question not found") from exc

    doc = await collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="question not found")

    if not force:
        existing = await simplified.find_one({"question_id": question_id})
        if existing is not None:
            return SimplifiedQuestion(**existing)

    return await simplify_and_store(
        ica, simplified, doc, body.model_id, body.max_tokens
    )
