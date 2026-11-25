"""Icon picking for a question (an ICA call, after its decomposition).

The model chooses, from the closed list in `icon_catalog.py`, one icon key for every
entity of the decomposition (`entities.derive_entities`), one for the question and one
for each answer. It reads the question through `KeyFreeQuestion`: icons are shown before
the player answers. Only keys are stored, on the `question_decomposition` record (the
`icon` fields inside `entities`, and `icons_ica`), so the picture a client draws for a
key can change without touching the data; a new decomposition clears them.
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.ica import IcaClient
from claude_exam.api.icon_catalog import ICONS, KIND_FALLBACK, known
from claude_exam.api.ica_json import ask_json
from claude_exam.api.question_view import KeyFreeQuestion, key_free
from claude_exam.api.schemas import (
    AnswerEntities,
    DecompositionRecord,
    IcaMeta,
    IcaModelId,
    PromptEntities,
    VisualEntity,
)
from claude_exam.api.step_key import step_key

logger = logging.getLogger(__name__)

# Bump when `parse_icons` changes what is stored from the same reply (the prompt and
# the catalog are in the input key, `step_key.py`).
CODE_VERSION = "1"

_SYSTEM_MESSAGE = """\
You choose icons from an icon catalog for one multiple-choice exam question, so a quiz \
game can show one picture on the question and one on each answer card. The exam is \
about Claude Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool \
use and agent design.

ICON CATALOG - "key: what it stands for". Use ONLY these keys, spelled exactly.
{{CATALOG}}

WHAT TO CHOOSE
1. For every entity in the ENTITIES list: the key that best shows that thing. The \
entity's kind (actor, object, action) and its other wordings are hints. Give the same \
entity one key; give different entities different keys where the catalog allows it. \
Use "actor", "object" or "action" only when no other key fits.
2. For the QUESTION: the key for what the scenario is mainly about (the system or \
situation being built or fixed).
3. For EACH ANSWER: the key for what THAT answer proposes to use or do - the thing \
that makes it different from the other answers, not the scenario they all share. \
If all answers mention the subagent but one changes a prompt, one adds a hook and one \
changes a setting, the icons are prompt, hook and settings. Answers of one question \
should get different keys whenever their proposals differ. An answer does not have to \
reuse the key of one of its entities.

Do not judge which answer is correct, and do not let the choice hint at it: pick by \
what the answer talks about, in the same way for every answer.

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{
  "entities": {"<entity name exactly as listed>": "<key>"},
  "question": "<key>",
  "answers": [{{ANSWERS_SCHEMA}}]
}

Rules:
- "entities" has one item per listed entity (an empty object when none are listed).
- "answers" has exactly {{N}} keys, index-aligned with the answers (A is index 0).
- Every value is a key from the catalog.
- Output valid JSON only.\
"""


def _build_system_message(n_answers: int) -> str:
    catalog = "\n".join(f"- {icon.key}: {icon.meaning}" for icon in ICONS)
    answers = ", ".join(f'"<key for {chr(65 + i)}>"' for i in range(n_answers))
    return (
        _SYSTEM_MESSAGE.replace("{{CATALOG}}", catalog)
        .replace("{{ANSWERS_SCHEMA}}", answers)
        .replace("{{N}}", str(n_answers))
    )


def _distinct(entities: PromptEntities) -> list[VisualEntity]:
    """Every entity once (by lower-cased name), in order of first mention."""
    seen: dict[str, VisualEntity] = {}
    for entity in [*entities.question, *(e for a in entities.answers for e in a.entities)]:
        seen.setdefault(entity.name.lower(), entity)
    return list(seen.values())


def _build_user_message(q: KeyFreeQuestion, entities: PromptEntities) -> str:
    answers = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(q.answers))
    listed = "\n".join(
        f"- {e.name} ({e.kind})" + (f" - also: {', '.join(e.aliases)}" if e.aliases else "")
        for e in _distinct(entities)
    )
    return (
        f"Section: {q.section}\n\n"
        f"QUESTION:\n{q.stem or q.question}\n\n"
        f"ANSWERS:\n{answers}\n\n"
        f"ENTITIES:\n{listed or '(none)'}"
    )


def _aligned(entities: PromptEntities, n_answers: int) -> PromptEntities:
    """A copy with exactly one `answers` item per answer."""
    answers = [a.model_copy(deep=True) for a in entities.answers[:n_answers]]
    answers += [AnswerEntities() for _ in range(n_answers - len(answers))]
    return PromptEntities(
        question_icon=entities.question_icon,
        question=[e.model_copy() for e in entities.question],
        answers=answers,
    )


def parse_icons(parsed: dict, entities: PromptEntities, n_answers: int) -> PromptEntities:
    """
    Validate the model's reply and return `entities` with the icons filled in.
    A reply of the wrong shape raises 502. A single bad pick does not: an entity
    without a known key gets the icon of its kind; an answer without one gets the icon
    of its least shared entity, the question that of its first entity (None when it
    has no entities, and the client then draws its default).
    """

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Icon output rejected: {msg}")

    by_name = parsed.get("entities")
    if not isinstance(by_name, dict):
        raise bad("'entities' must be an object of entity name -> key")
    raw_answers = parsed.get("answers")
    if not isinstance(raw_answers, list) or len(raw_answers) != n_answers:
        got = len(raw_answers) if isinstance(raw_answers, list) else type(raw_answers).__name__
        raise bad(f"'answers' must be a list of {n_answers} keys, got {got}")

    picks = {str(name).strip().lower(): known(key) for name, key in by_name.items()}
    result = _aligned(entities, n_answers)

    def assign(entity: VisualEntity) -> None:
        key = picks.get(entity.name.lower())
        if key is None:
            logger.info("no icon picked for entity %r; using its kind", entity.name)
        entity.icon = key or KIND_FALLBACK[entity.kind]

    for entity in result.question:
        assign(entity)
    for answer in result.answers:
        for entity in answer.entities:
            assign(entity)

    # How many answers mention each entity: the least shared one tells an answer apart.
    shared: dict[str, int] = {}
    for answer in result.answers:
        for name in {e.name.lower() for e in answer.entities}:
            shared[name] = shared.get(name, 0) + 1

    for answer, raw in zip(result.answers, raw_answers):
        answer.icon = known(raw)
        if answer.icon is None and answer.entities:
            answer.icon = min(answer.entities, key=lambda e: shared[e.name.lower()]).icon

    result.question_icon = known(parsed.get("question"))
    if result.question_icon is None and result.question:
        result.question_icon = result.question[0].icon
    return result


async def pick_icons(
    ica: IcaClient,
    q: KeyFreeQuestion,
    entities: PromptEntities,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[PromptEntities, IcaMeta]:
    """Ask the ICA chat model for the icons of one question and its entities; returns
    the entities with `icon` fields set."""
    system, user = messages(q, entities)
    parsed, meta = await ask_json(
        ica, model_id, user, system=system, what="icons", max_tokens=max_tokens
    )
    return parse_icons(parsed, entities, len(q.answers)), meta


def messages(q: KeyFreeQuestion, entities: PromptEntities) -> tuple[str, str]:
    """The (system, user) messages that pick the icons of a question. The entities'
    own icons are not in them, so the key is the same before and after picking."""
    return _build_system_message(len(q.answers)), _build_user_message(q, entities)


def input_key(q: KeyFreeQuestion, entities: PromptEntities) -> str:
    """The input key of the icons of a question and its entities (`step_key.py`)."""
    return step_key(CODE_VERSION, *messages(q, entities))


def stored_entities(record: dict | None) -> PromptEntities:
    """The entities of a `question_decomposition` record (none when malformed)."""
    try:
        return PromptEntities.model_validate((record or {}).get("entities") or {})
    except ValueError:
        return PromptEntities()


async def assign_and_store(
    ica: IcaClient,
    store: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> DecompositionRecord:
    """Pick icons for the entities of a decomposed question and store them on its
    `question_decomposition` record (the `icon` fields inside `entities`, and
    `icons_ica`). 404 without a decomposition; errors propagate."""
    existing = await store.find_one({"question_id": str(doc["_id"])})
    if existing is None:
        raise HTTPException(status_code=404, detail="question has no decomposition")
    q = key_free(doc)
    entities = stored_entities(existing)
    with_icons, meta = await pick_icons(ica, q, entities, model_id, max_tokens)
    # Conditional: the icons belong to the entities that were sent, so they must not
    # land on a record rewritten meanwhile (a new decomposition has new entities).
    stored = await store.find_one_and_update(
        {"question_id": existing["question_id"], "updated_at": existing["updated_at"]},
        {
            "$set": {
                "entities": with_icons.model_dump(),
                "icons_ica": meta.model_dump(),
                "icons_input_key": input_key(q, entities),
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(
            status_code=409,
            detail="decomposition changed while icons were picked; retry",
        )
    return DecompositionRecord(**stored)
