from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.routers.game import answers_of, correct_indices
from claude_exam.api.schemas import SearchHit, SearchResponse

# Ranking is done by MongoDB: a weighted text index (`TEXT_INDEX_*`, created in the
# app lifespan) scored with `$meta: "textScore"`.
TEXT_INDEX_NAME = "examination_text"
TEXT_INDEX_KEYS = [
    ("prompt.question", "text"),
    ("prompt.explanation", "text"),
    ("prompt.answers", "text"),
    ("section", "text"),
]
TEXT_INDEX_WEIGHTS = {
    "prompt.question": 10,
    "section": 3,
    "prompt.explanation": 2,
    "prompt.answers": 1,
}

SCORE = {"$meta": "textScore"}


def build_filter(query: str, section: str = "", assessment_type: str = "") -> dict:
    """`$text` match plus optional exact filters. Empty strings mean "any"."""
    flt: dict = {"$text": {"$search": query}}
    if section:
        flt["section"] = section
    if assessment_type:
        flt["assessment_type"] = assessment_type
    return flt


def to_hit(doc: dict) -> SearchHit:
    answers = [str(a) for a in answers_of(doc)]
    prompt = doc.get("prompt") or {}
    return SearchHit(
        id=str(doc["_id"]),
        score=float(doc.get("score") or 0.0),
        section=doc.get("section") or "",
        assessment_type=doc.get("assessment_type") or "",
        question=prompt.get("question") or "",
        answers=answers,
        correct_answers=[answers[i] for i in correct_indices(doc)],
        explanation=prompt.get("explanation") or "",
    )


async def search_questions(
    collection: AsyncCollection,
    query: str,
    *,
    section: str = "",
    assessment_type: str = "",
    limit: int = 5,
) -> SearchResponse:
    cursor = (
        collection.find(build_filter(query, section, assessment_type), {"score": SCORE})
        .sort({"score": SCORE})
        .limit(limit)
    )
    items = [to_hit(d) for d in await cursor.to_list()]
    return SearchResponse(items=items, total=len(items))
