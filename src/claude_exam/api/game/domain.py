"""Pure game rules and mappings: no database, no HTTP.

Questions come from `examination` (id in `_id`) or `examination_simplified` (same
`prompt` / `correct_response` layout, examination id as a *string* in `question_id`;
the id types differ for historical reasons, so the code always joins through `str(_id)`).
"""

import logging

from bson import ObjectId
from pydantic import ValidationError

from claude_exam.api.icon_catalog import known as known_icon
from claude_exam.api.objective_catalog import known as known_objective
from claude_exam.api.scenario_catalog import (
    known_scenario,
    known_stage,
    scenario_of,
    split_scenario,
    stem_of,
    text_hash,
)
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
# Verdicts the check step stores on a simplified record (`check.verdict`): a model
# answering the game text without the key picked the right answer ("same"); it picked
# another one there but the right one on the original, so the rewrite is at fault
# ("changed", and the game does not offer that text); or it missed both, which says
# the question is hard, not that the rewrite is wrong ("hard").
CHECK_SAME, CHECK_CHANGED, CHECK_HARD = "same", "changed", "hard"

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
    """A simplified version is only usable if it keeps the answers and the correct one,
    and the check step (`check.py`) has not found that its wording changed which answer
    is right. A version that was never checked is usable."""
    if ((simplified.get("check") or {}).get("verdict")) == CHECK_CHANGED:
        return False
    return len(answers_of(simplified)) == len(answers_of(original)) and correct_indices(
        simplified
    ) == correct_indices(original)


def usable_simplified(original: dict, simplified: dict | None) -> dict | None:
    """`simplified` if the game can offer it for `original`, else None."""
    if simplified is not None and simplified_matches(original, simplified):
        return simplified
    return None


def entities_of(doc: dict) -> PromptEntities:
    """The stored visual entities and icons of a simplified doc, with one `answers` item
    per answer (padded or trimmed). Entities are decoration: a malformed record gives none
    instead of failing the round."""
    try:
        entities = PromptEntities.model_validate(doc.get("entities") or {})
    except (ValidationError, KeyError, TypeError) as e:
        logger.warning("ignoring malformed entities of %s: %s", doc.get("question_id"), e)
        return PromptEntities()
    count = len(answers_of(doc))
    answers = entities.answers[:count]
    answers += [AnswerEntities() for _ in range(count - len(answers))]
    # A stored key can outlive its catalog entry; the game then draws its default.
    for entity in [*entities.question, *(e for a in answers for e in a.entities)]:
        entity.icon = known_icon(entity.icon)
    for answer in answers:
        answer.icon = known_icon(answer.icon)
    return PromptEntities(
        question_icon=known_icon(entities.question_icon),
        question=entities.question,
        answers=answers,
    )


def objectives_of(tags: dict | None) -> list[str]:
    """The catalog keys of a `question_objectives` record, main one first; [] when the
    question is not tagged. A key that left the catalog is dropped."""
    raw = (tags or {}).get("objectives")
    return [k for k in raw if known_objective(k)] if isinstance(raw, list) else []


def difficulty_of(tags: dict | None) -> int:
    """1..3 from a `question_objectives` record; 0 when the question is not tagged."""
    value = (tags or {}).get("difficulty")
    return value if not isinstance(value, bool) and value in (1, 2, 3) else 0


def misconceptions_of(tags: dict | None, count: int, correct: int) -> list[str]:
    """One text per answer: what picking it suggests the player got wrong ("" for the
    correct answer and wherever the record has none). This is part of the answer key."""
    raw = (tags or {}).get("distractors")
    raw = raw if isinstance(raw, list) else []
    texts = []
    for i in range(count):
        item = raw[i] if i < len(raw) else None
        text = item.get("misconception") if isinstance(item, dict) else None
        texts.append(text if isinstance(text, str) and i != correct else "")
    return texts


def usable_story(original: dict, record: dict | None) -> dict | None:
    """`record` (a `question_story` record) if it was written for this text of the
    question and names a known stage, else None: a question edited since has no story."""
    if not isinstance(record, dict) or known_stage(record.get("stage")) is None:
        return None
    return record if record.get("text_hash") == text_hash(original) else None


def chapter_of(original: dict, record: dict | None) -> str:
    """The key of the question's chapter, "" when it has none. A scenario paragraph
    decides it; a question without a paragraph belongs to the chapter its (usable)
    story record names, which the model picked. A paragraph no chapter claims gives
    none, whatever a record says."""
    scenario = scenario_of(original)
    if scenario is not None:
        return scenario.key
    if split_scenario((original.get("prompt") or {}).get("question"))[0]:
        return ""
    return known_scenario((usable_story(original, record) or {}).get("scenario")) or ""


def story_of(original: dict, record: dict | None) -> dict:
    """What a question shows of the story before it is answered. For a question with a
    scenario paragraph the chapter and the stem come from its text, so they are there
    without a story record; the stage and the beat need one."""
    chapter = chapter_of(original, record)
    record = usable_story(original, record) if chapter else None
    return {
        "scenario": chapter,
        "stem": stem_of(original),
        "stage": record["stage"] if record else "",
        "beat": (record.get("beat") or "") if record else "",
    }


def outcomes_of(original: dict, record: dict | None) -> dict:
    """The story line for a correct and for a wrong answer ("" without a record). Part
    of what a round keeps back until the question is answered."""
    record = usable_story(original, record) if chapter_of(original, record) else None
    return {
        "win": (record.get("win") or "") if record else "",
        "fail": (record.get("fail") or "") if record else "",
    }


def to_game_question(
    doc: dict,
    *,
    simplified: bool = False,
    tags: dict | None = None,
    story: dict | None = None,
) -> GameQuestion:
    """Map an `examination` doc, or an `examination_simplified` doc when `simplified`
    (only those carry entities). `tags` is the question's `question_objectives` record
    and `story` what `story_of` returned for the original."""
    prompt = doc.get("prompt") or {}
    story = story or {}
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
        objectives=objectives_of(tags),
        difficulty=difficulty_of(tags),
        scenario=story.get("scenario", ""),
        stage=story.get("stage", ""),
        # A simplified text has the scenario woven in, so it has no separate stem.
        stem="" if simplified else story.get("stem", ""),
        beat=story.get("beat", ""),
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


def round_entry(
    doc: dict,
    simplified: dict | None,
    tags: dict | None = None,
    record: dict | None = None,
) -> dict:
    """What a round document keeps for one playable question: its answer key and reviews
    (`record` is its `question_story` record)."""
    correct = correct_indices(doc)[0]
    count = len(answers_of(doc))
    return {
        "correct": correct,
        "count": count,
        "original": to_review(doc).model_dump(),
        "simplified": to_review(simplified).model_dump() if simplified else None,
        "misconceptions": misconceptions_of(tags, count, correct),
        "outcomes": outcomes_of(doc, record),
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
    section: str | None,
    assessment_type: str | None,
    exclude_ids: list[str] = (),
    only_ids: list[str] | None = None,
) -> dict:
    """Filters plus the playable pre-filter, leaving out `exclude_ids` (examination ids)
    and, when `only_ids` is given (it may be empty), keeping only those."""
    query = {**filter_query(section, assessment_type), **PLAYABLE_QUERY}
    ids: dict = {}
    if only_ids is not None:
        ids["$in"] = [ObjectId(i) for i in only_ids if ObjectId.is_valid(i)]
    if exclude_ids:
        ids["$nin"] = [ObjectId(i) for i in exclude_ids]
    if ids:
        query["_id"] = ids
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
