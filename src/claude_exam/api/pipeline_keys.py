"""Whether a question's stored step results are current, by their input keys
(`step_key.py`): one rule for every step instead of a version scheme per step.

A step's result is current when the key stored with it equals the key of the messages
the step would send today, built from the question and the upstream results as they
are stored now: the same inputs the job's step reads (`batch._step`). So a changed
prompt, catalog, chapter, story or upstream result makes it outdated, and nothing has
to be bumped or cleared by hand. Pure: it works on records already read, so the job
(`batch.is_current`) and the report (`scripts/pipeline_report.py`) share it.
"""

from typing import Literal, NamedTuple

from pydantic import ValidationError

from claude_exam.api import (
    check,
    decompose,
    entities,
    flow,
    icons,
    objectives,
    simplify,
    story,
)
from claude_exam.api.game.domain import story_of
from claude_exam.api.schemas import (
    DecisionFlow,
    JobStage,
    PromptEntities,
    SimplifiedPrompt,
)

State = Literal["current", "outdated", "missing"]


class Records(NamedTuple):
    """One question's stored records (None when it has none)."""

    objectives: dict | None = None
    story: dict | None = None
    decomposition: dict | None = None
    simplified: dict | None = None


def _decomposition(doc: dict, records: Records):
    """The decomposition the later steps are built from (what `batch._decomposition`
    reads), or None."""
    return decompose.read_decomposition(records.decomposition, doc)


def _flow(records: Records) -> DecisionFlow | None:
    stored = (records.decomposition or {}).get("flow")
    try:
        return DecisionFlow.model_validate(stored) if stored else None
    except ValidationError:
        return None


def _prompt(records: Records) -> SimplifiedPrompt | None:
    stored = (records.simplified or {}).get("prompt")
    try:
        return SimplifiedPrompt(**stored) if stored else None
    except (TypeError, ValidationError):
        return None


def _entities(records: Records) -> PromptEntities:
    try:
        return PromptEntities.model_validate((records.simplified or {}).get("entities") or {})
    except ValidationError:
        return PromptEntities()


# Where each step's key is stored: the record (a `Records` field) and the field path.
KEY_FIELD: dict[str, tuple[str, str]] = {
    "objectives": ("objectives", "input_key"),
    "story": ("story", "input_key"),
    "decompose": ("decomposition", "input_key"),
    "flow": ("decomposition", "flow_input_key"),
    "resolve": ("decomposition", "resolution_input_key"),
    "simplify": ("simplified", "input_key"),
    "entities": ("simplified", "entities_input_key"),
    "icons": ("simplified", "icons_input_key"),
    "check": ("simplified", "check.input_key"),
}


def stored_key(stage: JobStage, records: Records) -> str | None:
    """The input key stored with `stage`'s result (None: no result, or no key)."""
    decomposition = records.decomposition or {}
    simplified = records.simplified or {}
    if stage == "objectives":
        return (records.objectives or {}).get("input_key")
    if stage == "story":
        return (records.story or {}).get("input_key")
    if stage == "decompose":
        return decomposition.get("input_key")
    if stage == "flow":
        return decomposition.get("flow_input_key") if decomposition.get("flow") else None
    if stage == "resolve":
        return decomposition.get("resolution_input_key") if decomposition.get("resolution") else None
    if stage == "simplify":
        return simplified.get("input_key")
    if stage == "entities":
        return simplified.get("entities_input_key") if simplified.get("entities_ica") else None
    if stage == "icons":
        return simplified.get("icons_input_key") if simplified.get("icons_ica") else None
    return (simplified.get("check") or {}).get("input_key")


def expected_key(stage: JobStage, doc: dict, records: Records) -> str | None:
    """The input key `stage` would have if it ran today (None when it can't run: an
    input it needs is missing, or its builder refuses the question)."""
    if stage == "objectives":
        return objectives.input_key(doc)
    if stage == "story":
        return story.input_key(doc)
    if stage == "decompose":
        return decompose.input_key(doc)
    found = _decomposition(doc, records)
    if stage == "flow":
        return flow.flow_input_key(doc, found) if found is not None else None
    if stage == "resolve":
        drawn = _flow(records)
        if found is None or drawn is None or not is_current("flow", doc, records):
            return None
        return flow.resolve_input_key(doc, found, drawn)
    place = story_of(doc, records.story)
    if stage == "simplify":
        return simplify.input_key(doc, place, found)
    prompt = _prompt(records)
    if prompt is None:
        return None
    if stage == "entities":
        return entities.input_key(doc, prompt, found)
    if stage == "icons":
        return icons.input_key((records.simplified or {}).get("section", ""), prompt, _entities(records))
    return check.input_key(doc, records.simplified, place)


def is_current(stage: JobStage, doc: dict, records: Records) -> bool:
    """Whether `stage`'s stored result was written from today's inputs. A seed story
    record is current as it is (it was written by hand)."""
    if stage == "story" and (records.story or {}).get("origin") == "seed":
        return True
    if not _readable(stage, doc, records):
        return False
    stored = stored_key(stage, records)
    return stored is not None and stored == expected_key(stage, doc, records)


def _readable(stage: JobStage, doc: dict, records: Records) -> bool:
    """Whether the readers still take the result: the game and the later steps read the
    decomposition, flow and resolution through their version fields
    (`decompose.read_decomposition`, `flow_is_current`, `resolution_is_current`), and
    serve a game text with the flow only when its `decomposition_version` is the
    decomposition's `fingerprint` (`game.domain._same_ids`). A result those refuse must
    be written again whatever its key says, even when its own inputs did not change (a
    new decomposition the game text's prompt reads the same). Goes when the readers move
    to the keys."""
    record = records.decomposition
    if stage == "decompose":
        return decompose.is_current(record, doc)
    if stage == "flow":
        return decompose.flow_is_current(record)
    if stage == "resolve":
        return decompose.resolution_is_current(record)
    if stage == "simplify":
        found = _decomposition(doc, records)
        fingerprint = decompose.fingerprint(found) if found is not None else None
        return (records.simplified or {}).get("decomposition_version") == fingerprint
    return True


def _has_result(stage: JobStage, records: Records) -> bool:
    decomposition = records.decomposition or {}
    simplified = records.simplified or {}
    return bool(
        {
            "objectives": records.objectives,
            "story": records.story,
            "decompose": records.decomposition,
            "flow": decomposition.get("flow"),
            "resolve": decomposition.get("resolution"),
            "simplify": records.simplified,
            "entities": simplified.get("entities_ica"),
            "icons": simplified.get("icons_ica"),
            "check": simplified.get("check"),
        }[stage]
    )


def state(stage: JobStage, doc: dict, records: Records) -> State:
    """"missing" without a stored result, else "current" or "outdated"."""
    if not _has_result(stage, records):
        return "missing"
    return "current" if is_current(stage, doc, records) else "outdated"
