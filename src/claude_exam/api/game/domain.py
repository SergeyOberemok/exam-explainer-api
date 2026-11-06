"""Pure game rules and mappings: no database, no HTTP.

Questions come from `examination` (id in `_id`) or `examination_simplified` (same
`prompt` / `correct_response` layout, examination id as a *string* in `question_id`;
the id types differ for historical reasons, so the code always joins through `str(_id)`).
"""

import logging

from bson import ObjectId
from pydantic import ValidationError

from claude_exam.api.schemas import (
    AnswerEntities,
    GameAnswer,
    GameQuestion,
    GameReview,
    PromptEntities,
)

logger = logging.getLogger(__name__)

ROUND_TTL_SECONDS = 24 * 60 * 60
FIFTY_FIFTY_CHARGES = 2
FIFTY_FIFTY_ELIMINATES = 2
# The game has one card and one key per answer.
MAX_ANSWERS = 4
# Upper bound on questions in one round (start, batches and search adds together). Keeps
# a round document far below MongoDB's 16 MB limit (an entry is a few KB at most).
MAX_ROUND_QUESTIONS = 500
# `$sample` runs on the pre-filter and `is_playable` has the last word, so ask for more
# than needed and trim: a round then rarely comes back short.
OVERSAMPLE_FACTOR = 2

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


def usable_simplified(original: dict, simplified: dict | None) -> dict | None:
    """`simplified` if the game can offer it for `original`, else None."""
    if simplified is not None and simplified_matches(original, simplified):
        return simplified
    return None


def entities_of(doc: dict) -> PromptEntities:
    """The stored visual entities of a simplified doc, with one `answers` item per
    answer (padded or trimmed). Entities are decoration: a malformed record gives none
    instead of failing the round."""
    try:
        entities = PromptEntities.model_validate(doc.get("entities") or {})
    except (ValidationError, KeyError, TypeError) as e:
        logger.warning("ignoring malformed entities of %s: %s", doc.get("question_id"), e)
        return PromptEntities()
    count = len(answers_of(doc))
    answers = entities.answers[:count]
    answers += [AnswerEntities() for _ in range(count - len(answers))]
    return PromptEntities(question=entities.question, answers=answers)


def to_game_question(doc: dict, *, simplified: bool = False) -> GameQuestion:
    """Map an `examination` doc, or an `examination_simplified` doc when `simplified`
    (only those carry entities)."""
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
        entities=entities_of(doc) if simplified else PromptEntities(),
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


def round_entry(doc: dict, simplified: dict | None) -> dict:
    """What a round document keeps for one playable question: its answer key and reviews."""
    return {
        "correct": correct_indices(doc)[0],
        "count": len(answers_of(doc)),
        "original": to_review(doc).model_dump(),
        "simplified": to_review(simplified).model_dump() if simplified else None,
    }


def filter_query(section: str | None, assessment_type: str | None) -> dict:
    """Exact filters; empty or None means "any" (JsonUtility sends "" for unset strings)."""
    query: dict = {}
    if section:
        query["section"] = section
    if assessment_type:
        query["assessment_type"] = assessment_type
    return query


def playable_query(
    section: str | None, assessment_type: str | None, exclude_ids: list[str] = ()
) -> dict:
    """Filters plus the playable pre-filter, leaving out `exclude_ids` (examination ids)."""
    query = {**filter_query(section, assessment_type), **PLAYABLE_QUERY}
    if exclude_ids:
        query["_id"] = {"$nin": [ObjectId(i) for i in exclude_ids]}
    return query


def dedupe(docs: list[dict], key: str) -> list[dict]:
    """`$sample` may return the same document twice."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        doc_id = str(doc[key])
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc)
    return unique


def pick_playable(docs: list[dict], count: int) -> list[dict]:
    """The first `count` distinct playable docs."""
    return [d for d in dedupe(docs, "_id") if is_playable(d)][:count]
