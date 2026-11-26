"""Whether a question's stored step results are current, by their input keys
(`step_key.py`): one rule for every step.

A step's result is current when the key stored with it equals the key of the messages
the step would send today, built from the question and the upstream results as they
are stored now: the same inputs the job's step reads (`batch._step`). So a changed
prompt, catalog, chapter, story or upstream result makes it outdated, and nothing has
to be bumped by hand. Pure: it works on records already read, so the job
(`batch.is_current`), the backfill routes and the reports share it.
"""

from typing import Literal, NamedTuple

from pydantic import ValidationError

from claude_exam.api import beat, check, compose, decompose, flow, icons, objectives, rephrase, story
from claude_exam.api.game.domain import story_of
from claude_exam.api.question_view import key_free
from claude_exam.api.schemas import DecisionFlow, JobStage

State = Literal["current", "outdated", "missing"]


class Records(NamedTuple):
    """One question's stored records (None when it has none)."""

    objectives: dict | None = None
    story: dict | None = None
    decomposition: dict | None = None
    simplified: dict | None = None
    # The question's game document (`compose.py`, collection `game_question`).
    game_question: dict | None = None


def _flow(records: Records) -> DecisionFlow | None:
    stored = (records.decomposition or {}).get("flow")
    try:
        return DecisionFlow.model_validate(stored) if stored else None
    except ValidationError:
        return None


# Where each step's key is stored: the record (a `Records` field) and the field path.
KEY_FIELD: dict[str, tuple[str, str]] = {
    "objectives": ("objectives", "input_key"),
    "story": ("story", "input_key"),
    "beat": ("story", "beat_input_key"),
    "decompose": ("decomposition", "input_key"),
    "icons": ("decomposition", "icons_input_key"),
    "flow": ("decomposition", "flow_input_key"),
    "resolve": ("decomposition", "resolution_input_key"),
    "rephrase": ("simplified", "input_key"),
    "check": ("simplified", "check.input_key"),
    "compose": ("game_question", "input_key"),
}


def _has_result(stage: JobStage, records: Records) -> bool:
    decomposition = records.decomposition or {}
    simplified = records.simplified or {}
    return bool(
        {
            "objectives": records.objectives,
            "story": records.story,
            "beat": (records.story or {}).get("beat"),
            "decompose": records.decomposition,
            "icons": decomposition.get("icons_ica"),
            "flow": decomposition.get("flow"),
            "resolve": decomposition.get("resolution"),
            "rephrase": records.simplified,
            "check": simplified.get("check"),
            "compose": records.game_question,
        }[stage]
    )


def stored_key(stage: JobStage, records: Records) -> str | None:
    """The input key stored with `stage`'s result (None: no result, or no key)."""
    if not _has_result(stage, records):
        return None
    record_name, path = KEY_FIELD[stage]
    value = getattr(records, record_name) or {}
    for name in path.split("."):
        value = (value or {}).get(name)
    return value


def expected_key(stage: JobStage, doc: dict, records: Records) -> str | None:
    """The input key `stage` would have if it ran today (None when it can't run: an
    input it needs is missing, or its builder refuses the question)."""
    if stage == "objectives":
        return objectives.input_key(doc)
    if stage == "story":
        return story.input_key(doc)
    if stage == "decompose":
        return decompose.input_key(doc)
    if stage == "compose":
        return compose.input_key(doc, records.objectives, records.story, records.decomposition)
    found = decompose.read_decomposition(records.decomposition, doc)
    if stage == "beat":
        return beat.input_key(doc, records.story, found)
    place = story_of(doc, records.story)
    if stage == "rephrase":
        return rephrase.input_key(doc, place, found)
    if stage == "check":
        return check.input_key(doc, records.simplified, place) if records.simplified else None
    if found is None:
        return None
    if stage == "icons":
        return icons.input_key(key_free(doc), icons.stored_entities(records.decomposition), found)
    if stage == "flow":
        return flow.flow_input_key(doc, found)
    drawn = _flow(records)
    if drawn is None or not is_current("flow", doc, records):
        return None
    return flow.resolve_input_key(doc, found, drawn, place)


def is_current(stage: JobStage, doc: dict, records: Records) -> bool:
    """Whether `stage`'s stored result was written from today's inputs. A seed story
    record and its beat are current as they are (they were written by hand)."""
    seed = (records.story or {}).get("origin") == "seed"
    if stage == "story" and seed or stage == "beat" and seed and _has_result(stage, records):
        return True
    stored = stored_key(stage, records)
    return stored is not None and stored == expected_key(stage, doc, records)


def state(stage: JobStage, doc: dict, records: Records) -> State:
    """"missing" without a stored result, else "current" or "outdated"."""
    if not _has_result(stage, records):
        return "missing"
    return "current" if is_current(stage, doc, records) else "outdated"
