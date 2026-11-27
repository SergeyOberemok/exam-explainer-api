"""Pure game rules and mappings: no database, no HTTP.

Questions come from `examination` (id in `_id`); the pipeline's records name them by
`question_id`, the examination id as a *string*, so the code always joins through
`str(_id)`.
"""

import logging

from bson import ObjectId
from pydantic import ValidationError

from claude_exam.api import decompose, hold
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
    DecisionFlow,
    GameFlow,
    GamePart,
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
    """Pick-one: 2..MAX_ANSWERS answers and exactly one valid correct answer, and not held
    for manual processing (`hold.is_held`)."""
    return (
        2 <= len(answers_of(doc)) <= MAX_ANSWERS
        and len(correct_indices(doc)) == 1
        and not hold.is_held(doc)
    )


def entities_of(record: dict | None, count: int) -> PromptEntities:
    """The visual entities and icons of a `question_decomposition` record, with one
    `answers` item per answer (`count`, padded or trimmed). Entities are decoration: a
    malformed record gives none instead of failing the round."""
    try:
        entities = PromptEntities.model_validate((record or {}).get("entities") or {})
    except (ValidationError, KeyError, TypeError) as e:
        logger.warning("ignoring malformed entities of %s: %s", (record or {}).get("question_id"), e)
        return PromptEntities()
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


def game_flow(flow: dict | None) -> GameFlow:
    """A stored `DecisionFlow` as the game reads it; empty when there is none or it is
    malformed (a flow is decoration, like the entities)."""
    if not flow:
        return GameFlow()
    try:
        parsed = DecisionFlow.model_validate(flow)
    except ValidationError as e:
        logger.warning("ignoring malformed flow: %s", e)
        return GameFlow()
    return GameFlow(
        primary=parsed.visualization.primary,
        secondary=list(parsed.visualization.secondary),
        reason=parsed.visualization.reason,
        trigger=parsed.trigger,
        # A stored key can outlive its catalog entry: serve it as "no icon".
        nodes=[n.model_copy(update={"icon": known_icon(n.icon) or ""}) for n in parsed.nodes],
        edges=parsed.edges,
    )


def _current_decomposition(original: dict, record: dict | None) -> dict | None:
    """The question's `question_decomposition` record when it was written for the
    question's text as it is (an edited question has none)."""
    return record if decompose.matches(record, original) else None


def flow_of(original: dict, record: dict | None) -> GameFlow:
    """The pre-answer flow of a question from its `question_decomposition` record. The
    flow, the parts and the entities are all on that record and a new decomposition
    clears the flow, so they always name the same ids."""
    current = _current_decomposition(original, record)
    if not decompose.has_flow(current):
        return GameFlow()
    return game_flow(current["flow"])


def parts_of(record: dict | None) -> list[GamePart]:
    """The semantic parts of a `question_decomposition` record ([] when it has none or
    they are malformed)."""
    raw = (record or {}).get("parts")
    if not isinstance(raw, list):
        return []
    try:
        return [GamePart.model_validate(p) for p in raw]
    except ValidationError:
        return []


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


def ids_filter(question_ids) -> dict:
    """`{"$in": [...]}` on examination `_id` for question ids given as strings (invalid
    ones left out)."""
    return {"$in": [ObjectId(i) for i in question_ids if ObjectId.is_valid(i)]}


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
