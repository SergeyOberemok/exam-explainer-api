"""Icon picking for a question (an ICA call, after its decomposition).

The model chooses, from the closed list in `icon_catalog.py`, one icon key for every
entity of the decomposition (concepts included), one for every action (the events of
the scenario, `listed_actions`), one for the question and one for each answer. It picks
by decomposition id. It reads the question through `KeyFreeQuestion` and the
decomposition, which was written without the key: icons are shown before the player
answers. Only keys are stored, on the `question_decomposition` record (`icons`, by id;
the same keys in the `icon` fields inside `entities`, which /v1 reads; `icons_ica`), so
the picture a client draws for a key can change without touching the data; a new
decomposition clears them.
"""

import logging
import re
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.decompose import matches, read_decomposition, scenario_view
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_json
from claude_exam.api.icon_catalog import (
    ACTION_FALLBACK,
    ICONS,
    KIND_FALLBACK,
    TYPE_FALLBACK,
    known,
)
from claude_exam.api.question_view import KeyFreeQuestion, key_free
from claude_exam.api.schemas import (
    AnswerEntities,
    DecompEvent,
    DecompositionRecord,
    IcaMeta,
    IcaModelId,
    PromptEntities,
    QuestionDecomposition,
    QuestionIcons,
    VisualEntity,
)
from claude_exam.api.step_key import step_key

logger = logging.getLogger(__name__)

# Bump when `parse_icons` changes what is stored from the same reply (the prompt and
# the catalog are in the input key, `step_key.py`).
CODE_VERSION = "3"

_SYSTEM_MESSAGE = """\
You choose icons from an icon catalog for one multiple-choice exam question, so a quiz \
game can show one picture on the question and one on each answer card. The exam is \
about Claude Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool \
use and agent design.

ICON CATALOG - "key: what it stands for". Use ONLY these keys, spelled exactly.
{{CATALOG}}

WHAT TO CHOOSE
1. For every entity in the ENTITIES list: the key that best shows that thing. The \
entity's type (actor, system, object, knowledge_source, artifact, external_service, \
concept) and its other wordings are hints. A concept (an option, a flag, a kind of \
state, a property) gets the key of what it is, not a generic idea: a resume option is \
"session", a --continue flag "cli-flag", state kept on a server "state". Give \
different entities different keys where the catalog allows it: the icons are what \
tells the answer cards apart. Use "actor", "object", "action" or "idea" only when no \
other key fits.
2. For every action in the ACTIONS list: the key of its verb, what is done (run, send, \
write, retry, review, ...), not of who does it.
3. For the QUESTION: the key for what the scenario is mainly about (the system or \
situation being built or fixed) - never the key of something only an answer mentions, \
and not the key of an answer card.
4. For EACH ANSWER: the key for what THAT answer proposes to use or do - the thing \
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
  "entities": {"<entity id as listed>": "<key>"},
  "actions": {"<action id as listed>": "<key>"},
  "question": "<key>",
  "answers": [{{ANSWERS_SCHEMA}}]
}

Rules:
- "entities" has one item per listed entity, "actions" one per listed action (an empty \
object when none are listed), keyed by the id before the colon.
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


def listed_actions(decomposition: QuestionDecomposition, q: KeyFreeQuestion) -> list[DecompEvent]:
    """The actions a question is drawn with: the events of the scenario as the pre-answer
    flow may use them (`decompose.scenario_view`: no reference to an entity only an
    answer names) that happened (with a `sequence`; one without is what should happen,
    left out), in order."""
    events = [e for e in scenario_view(decomposition, q).events if e.sequence is not None]
    return sorted(events, key=lambda e: e.sequence)


def _distinct(entities: PromptEntities) -> list[VisualEntity]:
    """Every entity once (by lower-cased name), in order of first mention."""
    seen: dict[str, VisualEntity] = {}
    for entity in [*entities.question, *(e for a in entities.answers for e in a.entities)]:
        seen.setdefault(entity.name.lower(), entity)
    return list(seen.values())


def _also(aliases: list[str]) -> str:
    return f" - also: {', '.join(aliases)}" if aliases else ""


def _build_user_message(
    q: KeyFreeQuestion, entities: PromptEntities, decomposition: QuestionDecomposition | None = None
) -> str:
    """The question, its answers, and what to pick for: every decomposition entity and
    action by id (without a decomposition, the visual entities, by name)."""
    answers = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(q.answers))
    if decomposition is None:
        listed = "\n".join(f"- {e.name}: {e.name} ({e.kind}){_also(e.aliases)}" for e in _distinct(entities))
        actions = ""
    else:
        names = {e.id: e.name for e in decomposition.entities}
        listed = "\n".join(f"- {e.id}: {e.name} ({e.type}){_also(e.aliases)}" for e in decomposition.entities)

        def who(event: DecompEvent) -> str:
            by = names.get(event.actor_id or "")
            on = names.get(event.target_id or "")
            return ", ".join([*([f"by {by}"] if by else []), *([f"on {on}"] if on else [])])

        actions = "\n".join(
            f"- {e.id}: {e.name}" + (f" ({who(e)})" if who(e) else "") for e in listed_actions(decomposition, q)
        )
    return (
        f"Section: {q.section}\n\n"
        f"QUESTION:\n{q.stem or q.question}\n\n"
        f"ANSWERS:\n{answers}\n\n"
        f"ENTITIES:\n{listed or '(none)'}\n\n"
        f"ACTIONS:\n{actions or '(none)'}"
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


def _answer_only_icon(key: str, q: KeyFreeQuestion) -> bool:
    """Whether the question icon `key` shows what only an answer talks about: a word of
    the key ("escalate") starts a word of an answer ("escalation criteria") and of no
    word of the scenario or stem."""
    stems = [w[:6] for w in key.split("-") if len(w) >= 4]
    if not stems:
        return False

    def says(text: str) -> bool:
        return any(re.search(rf"\b{re.escape(s)}", text, re.IGNORECASE) for s in stems)

    return not says(q.question) and any(says(a) for a in q.answers)


def parse_icons(
    parsed: dict,
    entities: PromptEntities,
    n_answers: int,
    q: KeyFreeQuestion | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> PromptEntities:
    """The visual entities with their icons (see `parse_picks`)."""
    return parse_picks(parsed, entities, n_answers, q, decomposition)[0]


def parse_picks(
    parsed: dict,
    entities: PromptEntities,
    n_answers: int,
    q: KeyFreeQuestion | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> tuple[PromptEntities, QuestionIcons]:
    """
    Validate the model's reply and return `entities` with the icons filled in, and the
    picks by id (`QuestionIcons`: every entity of `decomposition`, and with `q` every
    action). A reply of the wrong shape raises 502. A single bad pick does not: a
    decomposition entity without a known key gets the icon of its type
    (`TYPE_FALLBACK`), an action "action", a visual entity its decomposition entity's
    icon (by `ref`), else its pick by name, else the icon of its kind; an answer
    without one gets the icon of its least shared entity, the question that of its
    first entity (None when it has no entities, and the client then draws its
    default). The question's icon is shown before answering, so one that is also an
    answer card's, or (with `q`) that shows what only an answer talks about
    (`_answer_only_icon`), is replaced the same way.
    """

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Icon output rejected: {msg}")

    by_id = parsed.get("entities")
    if not isinstance(by_id, dict):
        raise bad("'entities' must be an object of entity id -> key")
    raw_actions = parsed.get("actions") or {}
    if not isinstance(raw_actions, dict):
        raise bad("'actions' must be an object of action id -> key")
    raw_answers = parsed.get("answers")
    if not isinstance(raw_answers, list) or len(raw_answers) != n_answers:
        got = len(raw_answers) if isinstance(raw_answers, list) else type(raw_answers).__name__
        raise bad(f"'answers' must be a list of {n_answers} keys, got {got}")

    picks = {str(name).strip().lower(): known(key) for name, key in by_id.items()}
    action_picks = {str(name).strip().lower(): known(key) for name, key in raw_actions.items()}

    by_ref: dict[str, str] = {}
    by_name: dict[str, str] = {}
    actions: dict[str, str] = {}
    if decomposition is not None:
        for e in decomposition.entities:
            key = picks.get(e.id.lower()) or picks.get(e.name.lower())
            if key is None:
                logger.info("no icon picked for entity %r; using its type", e.name)
            by_ref[e.id] = key or TYPE_FALLBACK.get(e.type, KIND_FALLBACK["object"])
            by_name.setdefault(e.name.lower(), by_ref[e.id])
        if q is not None:
            for event in listed_actions(decomposition, q):
                actions[event.id] = action_picks.get(event.id.lower()) or ACTION_FALLBACK

    result = _aligned(entities, n_answers)

    def assign(entity: VisualEntity) -> None:
        key = by_ref.get(entity.ref) or picks.get(entity.name.lower()) or by_name.get(entity.name.lower())
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
    answer_icons = {a.icon for a in result.answers}
    if result.question_icon is not None and (
        result.question_icon in answer_icons
        or (q is not None and _answer_only_icon(result.question_icon, q))
    ):
        logger.info("question icon %r points at an answer; using its first entity's", result.question_icon)
        result.question_icon = None
    if result.question_icon is None and result.question:
        result.question_icon = result.question[0].icon
    picked = QuestionIcons(
        question=result.question_icon,
        answers=[a.icon for a in result.answers],
        entities=by_ref,
        actions=actions,
    )
    return result, picked


async def pick_icons(
    ica: IcaClient,
    q: KeyFreeQuestion,
    entities: PromptEntities,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> tuple[PromptEntities, QuestionIcons, IcaMeta]:
    """Ask the ICA chat model for the icons of one question, its entities and actions;
    returns the entities with `icon` fields set and the picks by id."""
    system, user = messages(q, entities, decomposition)
    parsed, meta = await ask_json(
        ica, model_id, user, system=system, what="icons", max_tokens=max_tokens
    )
    with_icons, picked = parse_picks(parsed, entities, len(q.answers), q, decomposition)
    return with_icons, picked, meta


def messages(
    q: KeyFreeQuestion, entities: PromptEntities, decomposition: QuestionDecomposition | None = None
) -> tuple[str, str]:
    """The (system, user) messages that pick the icons of a question. The entities'
    own icons are not in them, so the key is the same before and after picking."""
    return _build_system_message(len(q.answers)), _build_user_message(q, entities, decomposition)


def input_key(
    q: KeyFreeQuestion, entities: PromptEntities, decomposition: QuestionDecomposition | None = None
) -> str:
    """The input key of the icons of a question, its entities and actions (`step_key.py`)."""
    return step_key(CODE_VERSION, *messages(q, entities, decomposition))


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
    `question_decomposition` record (`icons`, the `icon` fields inside `entities`, and
    `icons_ica`). 404 without a decomposition; errors propagate."""
    existing = await store.find_one({"question_id": str(doc["_id"])})
    # A record written for an earlier text of the question names other entities.
    if not matches(existing, doc):
        raise HTTPException(status_code=404, detail="question has no current decomposition")
    q = key_free(doc)
    entities = stored_entities(existing)
    decomposition = read_decomposition(existing, doc)
    with_icons, picked, meta = await pick_icons(ica, q, entities, model_id, max_tokens, decomposition)
    # Conditional: the icons belong to the entities that were sent, so they must not
    # land on a record rewritten meanwhile (a new decomposition has new entities).
    stored = await store.find_one_and_update(
        {"question_id": existing["question_id"], "updated_at": existing["updated_at"]},
        {
            "$set": {
                "entities": with_icons.model_dump(),
                "icons": picked.model_dump(),
                "icons_ica": meta.model_dump(),
                "icons_input_key": input_key(q, entities, decomposition),
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
