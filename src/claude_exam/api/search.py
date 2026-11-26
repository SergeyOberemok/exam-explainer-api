import re

from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.domain import (
    PLAYABLE_QUERY,
    answers_of,
    correct_indices,
    ids_filter,
    is_playable,
    to_game_question,
)
from claude_exam.api.schemas import (
    GameCard,
    GameCardList,
    GameQuestionList,
    SearchHit,
    SearchResponse,
)

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

# In `$search`, `"..."` is a phrase every match must contain and a word starting with `-`
# is excluded. Pasted text is plain words: a pasted `--max-turns` would otherwise hide
# the very question that mentions it, and one quoted phrase would hide almost all.
_OPERATORS = re.compile(r'["\\“”„]|(?<!\S)[-‐-―−]+')


def plain_terms(text: str) -> str:
    """The words of `text` with the `$text` operators (phrases, negation) removed."""
    return " ".join(_OPERATORS.sub(" ", text).split())


def build_filter(query: str, section: str = "", assessment_type: str = "") -> dict:
    """`$text` match plus optional exact filters. Empty strings mean "any"."""
    flt: dict = {"$text": {"$search": plain_terms(query)}}
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


async def find_ranked(
    collection: AsyncCollection, flt: dict, limit: int
) -> list[dict]:
    cursor = collection.find(flt, {"score": SCORE}).sort({"score": SCORE}).limit(limit)
    return await cursor.to_list()


async def search_questions(
    collection: AsyncCollection,
    query: str,
    *,
    section: str = "",
    assessment_type: str = "",
    limit: int = 5,
) -> SearchResponse:
    docs = await find_ranked(collection, build_filter(query, section, assessment_type), limit)
    items = [to_hit(d) for d in docs]
    return SearchResponse(items=items, total=len(items))


async def search_game_questions(
    collection: AsyncCollection,
    simplified: AsyncCollection,
    query: str,
    *,
    section: str = "",
    assessment_type: str = "",
    limit: int = 5,
) -> GameQuestionList:
    """Like `search_questions`, but only questions the game can play (with a rephrase
    record in `simplified`), in the /game format: no answer key, so the player still has
    to answer them in a round."""
    rephrased = await simplified.distinct("question_id")
    flt = {
        **build_filter(query, section, assessment_type),
        **PLAYABLE_QUERY,
        "_id": ids_filter(str(i) for i in rephrased),
    }
    docs = await find_ranked(collection, flt, limit)
    items = [to_game_question(d) for d in docs if is_playable(d)]
    return GameQuestionList(items=items, total=len(items))


async def search_game_cards(
    collection: AsyncCollection,
    cards: AsyncCollection,
    query: str,
    *,
    section: str = "",
    assessment_type: str = "",
    limit: int = 5,
) -> GameCardList:
    """The `/v2` game search: the text is matched against `examination` (its text
    index), only for questions the game serves (a `game_question` document, not held),
    and the hits come back as their composed cards, best first, without the key."""
    served = await cards.distinct("question_id", {"held": False})
    flt = {**build_filter(query, section, assessment_type), "_id": ids_filter(served)}
    docs = await find_ranked(collection, flt, limit)
    ids = [str(d["_id"]) for d in docs]
    found = await cards.find({"question_id": {"$in": ids}}, {"key": 0}).to_list()
    by_id = {d["question_id"]: d for d in found}
    items = [GameCard.model_validate(by_id[i]["public"]) for i in ids if i in by_id]
    return GameCardList(items=items, total=len(items))
