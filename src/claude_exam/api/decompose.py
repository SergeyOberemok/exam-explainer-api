"""Question decomposition (an ICA call): stage 1 of
`docs/question_decomposition_decision_flow_spec.md`.

The model reads the ORIGINAL question and its answers, and nothing of the answer key
(no correct letter, feedback or explanation), and writes what the question means:
entities with their roles, events, relationships, the problem (symptoms, causes,
consequences), conditions, goals, the question's intent, and what each answer proposes,
without judging it. Because it never saw the key, everything here may be shown before the
player answers: `flow.py` builds the pre-answer flow from it, and its semantic parts
(`parts_from`) and visual entities (`entities.derive_entities`) are built from it in
code and stored with it. The model reads the question through `KeyFreeQuestion`
(`question_view.py`), which has no field for the key.

A reply must pass `decomposition_errors` before it is stored. These are the spec's
validation checklist, made into code: ids are unique and every reference resolves, every
`explicit` item quotes its evidence from the question word for word, no number appears
that the question does not have, and there is one answer option per answer. A failing
reply is sent back with its faults (`ica_json.ask_validated`).

Stored in `question_decomposition` (`semantic`, `parts`, `entities`, `semantic_ica`,
`text_hash`, `input_key`), one record per question. Writing it clears what was built
on the previous one (icons, flow, resolution).
"""

import logging
import re
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.entities import derive_entities
from claude_exam.api.exam_terms import lost_numbers
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_validated, schema_skeleton
from claude_exam.api.question_view import KeyFreeQuestion, as_view, key_free, the_ask
from claude_exam.api.scenario_catalog import text_hash
from claude_exam.api.schemas import (
    DecompositionRecord,
    IcaMeta,
    IcaModelId,
    QuestionDecomposition,
    SemanticPart,
)
from claude_exam.api.step_key import key_or_none, step_key

logger = logging.getLogger(__name__)

# Bump when the gates or the parsing change what is stored from the same reply (the
# prompts themselves are in the input keys, `step_key.py`), AND when `parts_from` or
# `entities.derive_entities` change: the parts and entities are built in code when the
# decomposition is stored, so no key of their own covers how they are built. Bumping it
# asks the model for every decomposition again. The flow and resolution's code versions
# live in `flow.py`.
CODE_VERSION = "1"
MAX_ATTEMPTS = 3
# A question is a short scenario; more means the model is listing nouns.
MAX_ENTITIES = 12
_SYSTEM_MESSAGE = """\
You are a question decomposition engine for an exam about Claude Code, the Claude Agent \
SDK, the Claude API, MCP, prompt engineering, tool use, context management and agent \
design. You convert one multiple-choice question into a structured scenario model that \
an app turns into icons, plain sentences and a flow diagram for a learner.

You are NOT told which answer is correct, and you must not try to decide it. The \
answers are candidates, not facts: describe what each one proposes, never whether it \
works.

HOW TO WORK
1. Read the whole question before extracting anything.
2. Identify the question's actual task (what the reader must choose, explain or \
identify), not just its subject. A question may describe false positives but ask for \
an architecture: the task is the architecture.
3. Entities: who and what takes part, each with one type:
   - actor: person, role, team or autonomous agent that performs an action
   - system: software or service that performs operations
   - object: item created, changed, inspected or acted upon
   - knowledge_source: rules or information used to judge (guidelines, policy, spec)
   - artifact: produced or consumed information (report, comment, finding)
   - external_service: a separate service or integration
   - concept: an important abstract idea (confidence, ambiguity)
   Do not make an actor of every noun: guidelines are a knowledge_source. The name is \
the short noun a diagram box shows (1 to 4 words, Title Case: "Developer", "Project \
Skill", "Refund Policy"), never a description ("Developer with stale behavior"). Put \
exact identifiers the text uses for it (paths, tool, command and setting names, \
other wordings) in aliases, copied exactly; keep a product or mode name as the name \
when the question is about it (Claude Code, plan mode). At most {{MAX_ENTITIES}} \
entities; include the things the answer options rely on.
4. Events: meaningful actions or state changes, named as verb phrases ("Analyze pull \
request"), with who does them and on what.
5. Relationships connect entities, events and conditions by id. Use "causes" only when \
the question states or clearly establishes causality; otherwise "leads_to" or \
"associated_with" with source_status "inferred".
6. Keep separate: current_state (what happens now), the problem (symptoms = observable \
bad behavior; root_causes = why, if stated or inferable; consequences = the impact), \
and desired_state (what should happen). Do not take the first symptom for the root \
cause. Do not put a proposed solution into the current state. root_causes lists only \
causes the question STATES (explicit). When finding the cause is part of what the \
question asks, the answer options compete over it: leave root_causes empty and \
record the open cause in ambiguities_and_unknowns. Never use what you know about \
these products to fill in a cause, a check or a condition the question leaves open.
7. Conditions: rules, criteria, constraints and uncertainty that change what happens. \
Never invent a threshold, number, owner or rule.
8. answer_options: one per answer, same order (A is index 0): the mechanism it \
proposes in a short neutral phrase, and the ids of the entities it relies on.
9. The answer options are proposals, not part of the scenario. context, current and \
desired state, problem, events, relationships, conditions and goals describe only \
what the SCENARIO and QUESTION say: nothing an answer proposes (a separate instance, \
a score, a new step, a setting) appears there. The answers feed only answer_options \
and the entities those rely on.
10. Record anything important the question leaves open in ambiguities_and_unknowns.

SOURCE STATUS - every entity, event, condition and fact says where it comes from:
- "explicit": directly stated. Then "evidence" MUST be a phrase copied word for word \
(3 to 15 words; no paraphrase) from the SCENARIO or QUESTION - for an entity an \
answer option relies on, from that ANSWER OPTION.
- "inferred": a reasonable interpretation needed to connect the scenario. evidence "".
- "unknown": cannot be determined from the question. evidence "".
Never present an inference as explicit.

IDS - entities e1, e2...; events ev1, ev2...; conditions c1, c2... Unique, and every \
reference (actor_id, target_id, input_ids, output_ids, source_id, target_id, \
affected_entities, entity_ids) must be an id that exists. Use null for an unknown \
actor_id, target_id or sequence, and [] for empty lists.

STYLE - short, concrete labels a diagram can show (at most about 8 words). Numbers \
only as the question gives them. Be compact: a few items per list, only what explains \
the question.

OUTPUT FORMAT - respond with ONLY one JSON object of exactly this shape, every field \
present ([] for an empty list), no markdown fences, no prose:
{{SKELETON}}\
"""


def _build_system_message() -> str:
    return _SYSTEM_MESSAGE.replace("{{MAX_ENTITIES}}", str(MAX_ENTITIES)).replace(
        "{{SKELETON}}", schema_skeleton(QuestionDecomposition)
    )


def source_parts(question: KeyFreeQuestion | dict) -> tuple[str, str, list[str]]:
    """The scenario paragraph, the question itself and the answers, cleaned (from the
    key-free view, so a record passed here is reduced to one first)."""
    q = as_view(question)
    return q.paragraph, q.stem, list(q.answers)


def _build_user_message(q: KeyFreeQuestion) -> str:
    paragraph, stem, answers = source_parts(q)
    parts = [f"Section: {q.section}"]
    if paragraph:
        shared = q.shared
        note = (
            " (background shared by several questions: take from it only what this "
            "question needs)"
            if shared
            else ""
        )
        parts.append(f"SCENARIO{note}:\n{paragraph}")
    choices = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(answers))
    parts += [
        f"QUESTION:\n{stem}",
        f"ANSWER OPTIONS (candidates, not facts - do not judge them):\n{choices}",
    ]
    return "\n\n".join(parts)


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "`": ""})


def normalize(text: str) -> str:
    """Text as an evidence check compares it: lower case, straight quotes without
    backticks or double quotes, whitespace collapsed, no trailing punctuation."""
    text = text.translate(_QUOTES).replace('"', "").lower()
    return re.sub(r"\s+", " ", text).strip(" .,;:")


def _texts(decomposition: QuestionDecomposition) -> list[str]:
    """Every free-text field of a decomposition, for the number check."""
    d = decomposition
    facts = [
        *d.context.current_state,
        *d.context.desired_state,
        *d.problem.symptoms,
        *d.problem.root_causes,
        *d.problem.consequences,
        *d.goals_and_requirements.business_goals,
        *d.goals_and_requirements.behavioral_requirements,
        *d.goals_and_requirements.quality_requirements,
        *d.goals_and_requirements.constraints,
    ]
    return [
        d.question_summary,
        d.context.description,
        d.question_intent.question_task,
        d.question_intent.target_of_question,
        *d.question_intent.success_criteria,
        *(f.text for f in facts),
        *(e.name for e in d.entities),
        *(e.description for e in d.entities),
        *(e.name for e in d.events),
        *(c.description for c in d.conditions),
        *(o.mechanism for o in d.answer_options),
        *d.ambiguities_and_unknowns,
    ]


def invented_numbers(source: str, texts: list[str]) -> list[str]:
    """Numbers in `texts` that `source` does not contain."""
    return lost_numbers("\n".join(texts), source)


def decomposition_errors(doc: dict, parsed: dict) -> list[str]:
    """What makes a reply unusable, as sentences the model can act on (empty when it
    passes). See the module doc for the rules."""
    try:
        d = QuestionDecomposition.model_validate(parsed)
    except ValidationError as exc:
        return [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
        ]
    paragraph, stem, answers = source_parts(doc)
    source = "\n".join([paragraph, stem, *answers])
    haystack = normalize(source)
    errors: list[str] = []

    ids: dict[str, str] = {}
    for kind, items in (("entity", d.entities), ("event", d.events), ("condition", d.conditions)):
        for item in items:
            if item.id in ids:
                errors.append(f"id {item.id!r} is used twice ({ids[item.id]} and {kind})")
            ids[item.id] = kind
    entity_ids = {e.id for e in d.entities}

    def check_refs(where: str, refs: list[str | None], allowed: set[str]) -> None:
        missing = [r for r in refs if r and r not in allowed]
        if missing:
            errors.append(f"{where} refers to ids that do not exist: {', '.join(missing)}")

    for event in d.events:
        check_refs(
            f"event {event.id}",
            [event.actor_id, event.target_id, *event.input_ids, *event.output_ids],
            entity_ids,
        )
    for relationship in d.relationships:
        check_refs(
            f"relationship {relationship.source_id} {relationship.type}",
            [relationship.source_id, relationship.target_id],
            set(ids),
        )
    check_refs("problem.affected_entities", d.problem.affected_entities, entity_ids)
    for option in d.answer_options:
        check_refs(f"answer option {option.index}", option.entity_ids, entity_ids)

    if len(d.entities) > MAX_ENTITIES:
        errors.append(f"at most {MAX_ENTITIES} entities, got {len(d.entities)}")
    if sorted(o.index for o in d.answer_options) != list(range(len(answers))):
        errors.append(
            f"answer_options must have exactly one item per answer, with index 0 to "
            f"{len(answers) - 1}"
        )
    if not d.question_intent.question_task.strip():
        errors.append("question_intent.question_task must say what the question asks")

    # Only an entity may rest on an answer option; everything else describes the
    # scenario, and an answer's mechanism there would show it as a given fact.
    scenario = normalize("\n".join([paragraph, stem]))

    def check_evidence(where: str, status: str, evidence: str, entity: bool = False) -> None:
        if status != "explicit":
            return
        quote = normalize(evidence)
        if len(quote) < 3 or quote not in haystack:
            errors.append(
                f"{where} is explicit, so its evidence must be a phrase copied word for "
                f"word from the question (got {evidence!r}); copy it exactly or mark it "
                "inferred"
            )
        elif not entity and quote not in scenario:
            errors.append(
                f"{where} quotes an answer option ({evidence!r}): what the answers propose "
                "is not part of the scenario. Remove it, or describe only what the "
                "SCENARIO and QUESTION say"
            )

    for entity in d.entities:
        check_evidence(f"entity {entity.id}", entity.source_status, entity.evidence, entity=True)
    for event in d.events:
        check_evidence(f"event {event.id}", event.source_status, event.evidence)
    for condition in d.conditions:
        check_evidence(f"condition {condition.id}", condition.source_status, condition.evidence)
    groups = {
        "current_state": d.context.current_state,
        "desired_state": d.context.desired_state,
        "symptoms": d.problem.symptoms,
        "root_causes": d.problem.root_causes,
        "consequences": d.problem.consequences,
        "business_goals": d.goals_and_requirements.business_goals,
        "behavioral_requirements": d.goals_and_requirements.behavioral_requirements,
        "quality_requirements": d.goals_and_requirements.quality_requirements,
        "constraints": d.goals_and_requirements.constraints,
    }
    for name, facts in groups.items():
        for i, fact in enumerate(facts):
            check_evidence(f"{name}[{i}]", fact.source_status, fact.evidence)
    # An inferred cause is usually the answer, worked out from what the model knows: it
    # would show before answering what the question asks the player to find.
    for i, cause in enumerate(d.problem.root_causes):
        if cause.source_status != "explicit":
            errors.append(
                f"root_causes[{i}] is {cause.source_status}: list only causes the "
                "question states; an open cause belongs in ambiguities_and_unknowns"
            )

    # The scenario may not name what only an answer proposes, whatever its source
    # status: an inferred desired state "use a separate review instance" is the answer.
    # (Only entities an answer names: one no text names at all is the model's own word.)
    answer_text = normalize(" ".join(answers))
    answer_only = [
        e for e in answer_only_entities(d, doc) if _in_scenario(e, answer_text)
    ]
    scenario_texts = [
        ("context.description", d.context.description),
        *((f"{name}[{i}]", fact.text) for name, facts in groups.items() for i, fact in enumerate(facts)),
        *((f"event {e.id}", e.name) for e in d.events),
        *((f"condition {c.id}", c.description) for c in d.conditions),
    ]
    for where, text in scenario_texts:
        named = names_entity(text, answer_only)
        if named:
            errors.append(
                f"{where} names {named!r}, which only an answer option mentions: what "
                "the answers propose is not part of the scenario. Describe only what the "
                "SCENARIO and QUESTION say"
            )

    numbers = invented_numbers(source, _texts(d))
    if numbers:
        errors.append(
            f"these numbers are not in the question; remove them: {', '.join(numbers)}"
        )
    return errors


def messages(q: KeyFreeQuestion) -> tuple[str, str]:
    """The (system, user) messages that decompose the question; 422 without answers."""
    if not q.answers:
        raise HTTPException(status_code=422, detail="question has no answers to decompose")
    return _build_system_message(), _build_user_message(q)


def input_key(doc: dict) -> str | None:
    """The input key of the decomposition of `doc` (`step_key.py`)."""
    return key_or_none(lambda: step_key(CODE_VERSION, *messages(key_free(doc))))


async def decompose_item(
    ica: IcaClient,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[QuestionDecomposition, IcaMeta]:
    """Decompose one examination document (needs `prompt.question` and `prompt.answers`;
    the key is never read). 422 for a question without answers, 502 when the last reply
    still fails `decomposition_errors`."""
    system, user = messages(key_free(doc))
    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=system,
        what="decomposition",
        validate=lambda parsed: decomposition_errors(doc, parsed),
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        schema=QuestionDecomposition,
        label=doc.get("_id"),
    )
    return QuestionDecomposition.model_validate(parsed), meta


def matches(record: dict | None, doc: dict) -> bool:
    """Whether a stored decomposition was written for the question's text as it is (an
    edited question has none until it is decomposed again)."""
    return record is not None and record.get("text_hash") == text_hash(doc)


def _in_scenario(entity, scenario: str) -> bool:
    """Whether the scenario paragraph or the stem names this entity (by its evidence,
    its name or one of its aliases)."""
    terms = [entity.evidence, entity.name, *entity.aliases]
    return any(len(normalize(t)) >= 3 and normalize(t) in scenario for t in terms)


def names_entity(text: str, entities: list) -> str:
    """The first name or alias of `entities` that `text` contains ("" when none); terms
    under 4 characters are ignored, they match inside other words."""
    normalized = normalize(text)
    for entity in entities:
        for term in (entity.name, *entity.aliases):
            if len(normalize(term)) >= 4 and normalize(term) in normalized:
                return term
    return ""


def answer_only_entities(decomposition: QuestionDecomposition, doc: KeyFreeQuestion | dict) -> list:
    """The entities only an answer option names: what the answers propose (a separate
    instance, a personal skills folder). Shown before answering they point at one."""
    paragraph, stem, _answers = source_parts(doc)
    scenario = normalize("\n".join([paragraph, stem]))
    return [e for e in decomposition.entities if not _in_scenario(e, scenario)]


def scenario_view(
    decomposition: QuestionDecomposition, doc: KeyFreeQuestion | dict
) -> QuestionDecomposition:
    """The decomposition as the pre-answer flow may use it: without the answer options,
    the entities only they name (and the references to those), and the open questions
    and success criteria that name such an entity."""
    answer_only = answer_only_entities(decomposition, doc)
    gone = {e.id for e in answer_only}
    view = decomposition.model_copy(deep=True)
    view.answer_options = []
    view.entities = [e for e in view.entities if e.id not in gone]
    for event in view.events:
        event.actor_id = None if event.actor_id in gone else event.actor_id
        event.target_id = None if event.target_id in gone else event.target_id
        event.input_ids = [i for i in event.input_ids if i not in gone]
        event.output_ids = [i for i in event.output_ids if i not in gone]
    view.relationships = [
        r for r in view.relationships if r.source_id not in gone and r.target_id not in gone
    ]
    view.problem.affected_entities = [i for i in view.problem.affected_entities if i not in gone]
    view.ambiguities_and_unknowns = [
        a for a in view.ambiguities_and_unknowns if not names_entity(a, answer_only)
    ]
    view.question_intent.success_criteria = [
        s for s in view.question_intent.success_criteria if not names_entity(s, answer_only)
    ]
    return view


MAX_PARTS = 6


def _mentioned(text: str, entities: list) -> list[str]:
    """The ids of the entities `text` names (by name or alias, as `names_entity`)."""
    normalized = normalize(text)
    return [
        e.id
        for e in entities
        if any(len(normalize(t)) >= 4 and normalize(t) in normalized for t in (e.name, *e.aliases))
    ]


def parts_from(decomposition: QuestionDecomposition, question: KeyFreeQuestion) -> list[SemanticPart]:
    """The question as semantic parts, built in code from what may be shown before
    answering (`scenario_view`: no answer option and no entity only an answer names):
    the context, the current state, the symptoms, the consequences, the goals and the
    explicit conditions, in that order, then the question's own ask, word for word, as
    the last part. Each part names the ids of the entities its text mentions. At most
    `MAX_PARTS`: the ask always stays, the rest is cut from the end."""
    view = scenario_view(decomposition, question)
    found: list[tuple[str, str]] = []
    if view.context.description.strip():
        found.append(("context", view.context.description))
    found += [("current_state", f.text) for f in view.context.current_state]
    found += [("problem", f.text) for f in view.problem.symptoms]
    found += [("consequence", f.text) for f in view.problem.consequences]
    found += [("goal", f.text) for f in view.context.desired_state]
    found += [
        ("constraint", c.description) for c in view.conditions if c.source_status == "explicit"
    ]
    ask = the_ask(question) or question.stem
    parts = [
        SemanticPart(role=role, text=text.strip(), entity_ids=_mentioned(text, view.entities))
        for role, text in found
        if text.strip()
    ][: MAX_PARTS - 1]
    parts.append(SemanticPart(role="ask", text=ask, entity_ids=_mentioned(ask, view.entities)))
    return parts


def has_flow(record: dict | None) -> bool:
    """Whether the record holds a flow. Writing a decomposition clears its flow, so a
    stored flow is always one of the stored decomposition."""
    return bool(record) and record.get("flow") is not None


def has_resolution(record: dict | None) -> bool:
    """Whether the record holds a resolution (one of its flow: writing a flow clears it)."""
    return bool(record) and record.get("resolution") is not None


async def store_decomposition(
    store: AsyncCollection,
    doc: dict,
    decomposition: QuestionDecomposition,
    meta: IcaMeta,
) -> DecompositionRecord:
    """Upsert the decomposition of one question with its parts and entities (built
    from it in code); the icons, flow and resolution of an earlier decomposition go
    with it (they reference its ids)."""
    q = key_free(doc)
    now = datetime.now(timezone.utc)
    stored = await store.find_one_and_update(
        {"question_id": str(doc["_id"])},
        {
            "$set": {
                "text_hash": text_hash(doc),
                "semantic": decomposition.model_dump(),
                "semantic_ica": meta.model_dump(),
                "parts": [p.model_dump() for p in parts_from(decomposition, q)],
                "entities": derive_entities(decomposition, q).model_dump(),
                "icons_ica": None,
                "icons_input_key": None,
                "input_key": input_key(doc),
                "flow": None,
                "flow_ica": None,
                "flow_input_key": None,
                "resolution": None,
                "resolution_ica": None,
                "resolution_input_key": None,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return DecompositionRecord(**stored)


async def decompose_and_store(
    ica: IcaClient,
    store: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> DecompositionRecord:
    decomposition, meta = await decompose_item(ica, doc, model_id, max_tokens)
    return await store_decomposition(store, doc, decomposition, meta)


def read_decomposition(record: dict | None, doc: dict) -> QuestionDecomposition | None:
    """The stored decomposition of a question when it is current, else None (a
    malformed one reads as None too)."""
    if not matches(record, doc):
        return None
    try:
        return QuestionDecomposition.model_validate(record.get("semantic"))
    except ValidationError:
        return None
