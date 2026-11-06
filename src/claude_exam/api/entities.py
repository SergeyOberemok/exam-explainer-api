"""Visual entity extraction for simplified questions (second ICA call).

The prompt below is the machine-readable version of `docs/extracting_actor_entities.md`:
same selection rules, but the reply is one JSON object holding an entity catalog plus
references (entity ids) for the question text and for each answer, so the app can map
text to diagram elements by structure.
"""

import logging
import re
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.ica import IcaClient
from claude_exam.api.json_reply import extract_json
from claude_exam.api.schemas import (
    IcaMeta,
    IcaModelId,
    AnswerEntities,
    PromptEntities,
    SimplifiedPrompt,
    SimplifiedQuestion,
    VisualEntity,
)
from claude_exam.html_clean import clean_plain, clean_question

logger = logging.getLogger(__name__)

# A question is a short scenario; more than this means the model is listing nouns.
# Beyond the limits the reply is trimmed (in order), not rejected.
MAX_ENTITIES = 16
MAX_PER_ANSWER = 4
_KINDS = ("actor", "object", "action")
# One-word aliases that match too much text to identify a single entity.
_GENERIC_ALIASES = frozenset(
    {"refund", "policy", "tool", "tools", "message", "data", "system"}
)

# Entity names too generic to draw; references to them are dropped.
_GENERIC_NAMES = frozenset(
    {"tool", "tools", "tool set", "toolset", "data", "system", "information", "thing", "things"}
)
# Stable-name actors added deterministically wherever the text names them.
_STABLE_ACTORS = (
    ("Subagent", re.compile(r"\bsub-?agents?\b", re.I)),
    ("Main Agent", re.compile(r"\bmain agent\b", re.I)),
)

_SYSTEM_MESSAGE = """\
You extract reusable visual entities from one multiple-choice exam question so an app \
can draw them as diagram elements (icons, boxes, arrows). The exam is about Claude \
Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool use and \
agent design.

SOURCES
- ORIGINAL text is the primary source for what the question is about. The learner can \
read its exact names (tools, parameters, settings, files) there, so the entities do \
NOT repeat them one by one: they give the simplified version a basic, abstract picture \
of who and what is involved.
- SIMPLIFIED text is the plain-words retelling: use it for context and to find the \
broad things the original only implies.
- The two texts describe the same question and the same answers in the same order. \
An entity found in either version of a question or answer belongs to that question or \
answer, even if the other version leaves it out.
- Entities come from the question and from ALL answer options, correct or not. The \
feedback and explanation are context only; do not reference them.

WHAT TO EXTRACT
Concrete things that can be shown as a recognizable diagram element, in three kinds:
- "actor": performs or takes part in something (a person or role such as Customer or \
Engineer, an AI agent, a model, a subagent, a coordinator).
- "object": is processed, exchanged, accessed or represented (software, API, server, \
database, SDK, MCP server, tool, file, setting or mode, document, session, repository, \
business object, device).
- "action": an operation that can be one workflow step (Escalate to Human, Review). \
Use it sparingly, only for a step the question is really about.

BROAD LEVEL
Think of a simple diagram a beginner could read: a few boxes such as Customer, Order, \
Refund, Agent, Human Reviewer, Database, Website. Entities live at that level.
- Do NOT make an entity per tool, function, parameter, flag or setting name \
(get_customer, lookup_order, process_refund, tool_choice, allowedTools). Fold each one \
into the broad thing it works on or stands for: the customer tools -> "Customer", \
lookup_order -> "Order", process_refund and a refund decision -> "Refund", \
escalate_to_human -> "Human Reviewer". Put the exact name in "aliases" only if it is a \
useful label in that place. Several tools of one area are ONE entity.
- Keep a specific name as the entity name only when the name itself is what the \
question tests: a product or platform (Claude Code, Claude Agent SDK, MCP), or a mode \
or setting the options are chosen between (plan mode, acceptEdits).

GROUPING
- Merge synonyms and variants of the same thing into one entity (customer + customer \
details -> "Customer"; order + order lookup data -> "Order").
- Name = the shortest clear Title Case noun for the group (see BROAD LEVEL); keep the \
original technical term only for products, modes and settings the question tests \
(`acceptEdits`, plan mode, MCP). Put other wordings from the text in "aliases".
- Do not make separate entities for attributes, versions or properties of one thing.
- Reuse one name for the same thing everywhere in this reply. Use the stable names \
"Claude Agent SDK", "Claude Code", "Claude API", "MCP Server", "Subagent", \
"Main Agent" for those things instead of inventing variants - but ONLY where the \
text actually refers to that thing. Tools that are reached over MCP are tools, not \
an "MCP Server": add "MCP Server" only when a server (or the protocol itself) is \
really mentioned in that text.
- "aliases" are only other wordings of the SAME thing that appear in the original or \
simplified text: never the name itself, never counts or numbers ("18 tools"), never a \
different thing. A request, message, group or category is never an alias of a tool or \
of what it works on (wrong: "Refund" with alias "refund requests"; wrong: "MCP" with \
alias "backend systems").
- Never make an entity for a generic group or word that cannot be drawn: "Tools", \
"Tool Set", "Data", "System", "Information". Name the concrete things instead, or \
leave the place empty.
- An alias must be specific enough to point at this entity alone: never a single \
generic word such as "refund", "policy" or "tool" (use the fuller phrase from the \
text, e.g. "proposed refund decision").
- Coverage check before you answer: every stable-name thing (Subagent, Main Agent, \
MCP, ...) that the original or simplified text of the question or of an answer \
mentions - including as "the subagent", "specialized subagents" or "MCP tools" - must \
appear in that place's list. Re-read each answer once and add what you missed. When \
the text says "MCP tools" or "Model Context Protocol", include an "MCP" entity.
- When the text lists several categories of tools by role (for example shipping, \
loyalty and subscription tools), fold them into the broad things they work on \
(Shipment, Loyalty Account, Subscription) rather than one entity per tool; be \
consistent: cover all categories or none.
- A specific named instance of a stable-name thing (a subagent called \
"test-gap-analyzer") is an alias of that stable name ("Subagent"), not a replacement \
for it: keep the stable name as the entity name and reference it wherever the text \
says "subagent" or uses the specific name.
- Reference an entity from the question or from an answer only if its name or one of \
its aliases appears in that text (original or simplified version). Do not attach an \
entity to an answer because it is related to the topic.

EXCLUDE
Purely abstract ideas are never entities: confidence, accuracy, reasoning, \
independence, verification, bias, threshold, reliability, decision quality, \
performance metrics, cost, latency. Also leave out events, situations and facts \
(a billing dispute, a failed audit, misrouting), message or conversation concepts \
(context, history, a follow-up message) and generic groups of things already listed \
individually ("MCP Tools" when each tool is named). Do not invent anything the text \
does not support. Do not list every noun; keep a compact set (at most \
{{MAX_ENTITIES}} entities, at most {{MAX_PER_ANSWER}} per answer).

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{
  "entities": [
    {"id": "e1", "name": "<entity name>", "kind": "actor|object|action", \
"aliases": ["<other wording>"]}
  ],
  "question_entities": ["<entity id>"],
  {{ANSWERS_SCHEMA}}
}

Rules:
- ids are "e1", "e2", ... unique within this reply.
- "question_entities" lists the ids of entities mentioned in the ORIGINAL or the \
SIMPLIFIED question.
- "answers_entities" has exactly {{N}} lists, index-aligned with the answers (A is \
index 0, B is 1, ...; original and simplified answers share the same order). Each list \
holds the entities mentioned in the ORIGINAL or the SIMPLIFIED version of that answer. \
Use [] for an answer with no entity.
- Every id used in a reference must exist in "entities", and every entity must be \
referenced at least once.
- Output valid JSON only.\
"""


def _build_system_message(n_answers: int) -> str:
    letters = [chr(65 + i) for i in range(n_answers)]
    entries = ",\n    ".join(f'["<ids for answer {l}>"]' for l in letters)
    return (
        _SYSTEM_MESSAGE.replace(
            "{{ANSWERS_SCHEMA}}", f'"answers_entities": [\n    {entries}\n  ]'
        )
        .replace("{{N}}", str(n_answers))
        .replace("{{MAX_ENTITIES}}", str(MAX_ENTITIES))
        .replace("{{MAX_PER_ANSWER}}", str(MAX_PER_ANSWER))
    )


def _build_user_message(item: dict, simplified: SimplifiedPrompt) -> str:
    """Original and simplified texts side by side, labelled by source."""
    prompt = item.get("prompt") or {}
    original_answers = "\n".join(
        f"  {chr(65 + i)}. {clean_plain(a)}"
        for i, a in enumerate(prompt.get("answers", []))
    )
    simplified_answers = "\n".join(
        f"  {chr(65 + i)}. {a}" for i, a in enumerate(simplified.answers)
    )
    return (
        f"Section: {item.get('section', '')}\n\n"
        f"ORIGINAL QUESTION:\n{clean_question(prompt.get('question', ''))}\n\n"
        f"ORIGINAL ANSWERS:\n{original_answers}\n\n"
        f"SIMPLIFIED QUESTION:\n{simplified.question}\n\n"
        f"SIMPLIFIED ANSWERS:\n{simplified_answers}"
    )


def _mentions(entity: VisualEntity, text: str) -> bool:
    """True when the entity's name or one of its aliases occurs in `text`."""
    text = text.lower()
    return any(t.lower() in text for t in (entity.name, *entity.aliases))


def parse_entities(
    parsed: dict,
    n_answers: int,
    texts: tuple[str, list[str]] | None = None,
) -> PromptEntities:
    """
    Validate the model's reply and build the stored structure. The reply uses a
    catalog plus id references (keeps names consistent and the reply short); the
    stored form resolves them: `question: [entity]`, `answers: [{entities: [entity]}]`.
    Structural problems (bad kind, duplicate id, wrong answer count, wrong types)
    raise 502. Overflow is trimmed instead of failing the reply: entities beyond
    MAX_ENTITIES, references to ids that do not exist (or were trimmed), and more than
    MAX_PER_ANSWER entities on an answer.

    `texts` = (question text, answer texts), each the original and simplified version
    joined. When given, a reference is also dropped (and logged) if the entity's name
    or aliases do not occur in that text, and aliases found nowhere in the question
    are pruned. Without it only the structure is checked.
    """

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Entity output rejected: {msg}")

    raw_entities = parsed.get("entities")
    if not isinstance(raw_entities, list):
        raise bad("'entities' must be a list")

    entities: dict[str, VisualEntity] = {}
    for raw in raw_entities:
        if not isinstance(raw, dict):
            raise bad("every entity must be an object")
        entity_id, name, kind = raw.get("id"), raw.get("name"), raw.get("kind")
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise bad("entity 'id' must be a non-empty string")
        if entity_id in entities:
            raise bad(f"duplicate entity id {entity_id!r}")
        if not isinstance(name, str) or not name.strip():
            raise bad(f"entity {entity_id!r} needs a non-empty 'name'")
        if kind not in _KINDS:
            raise bad(f"entity {entity_id!r} has invalid kind {kind!r}")
        aliases = raw.get("aliases") or []
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise bad(f"entity {entity_id!r} 'aliases' must be a list of strings")
        name = name.strip()
        # Aliases are other wordings: not the name itself, no repeats.
        clean_aliases = list(
            dict.fromkeys(
                a.strip()
                for a in aliases
                if a.strip() and a.strip().lower() != name.lower()
            )
        )
        entities[entity_id] = VisualEntity(name=name, kind=kind, aliases=clean_aliases)
    entities = dict(list(entities.items())[:MAX_ENTITIES])

    if texts is not None:
        # Aliases must come from the text; judged against the whole question (the
        # name is kept even when only an alias occurs, e.g. "Main Agent" / "the agent").
        everything = "\n".join([texts[0], *texts[1]]).lower()
        for entity in entities.values():
            entity.aliases = [
                a
                for a in entity.aliases
                if a.lower() in everything and a.lower() not in _GENERIC_ALIASES
            ]

    def refs(value: object, where: str, text: str | None) -> list[VisualEntity]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise bad(f"{where} must be a list of entity ids")
        # Unknown ids and repeats are dropped; order is kept. Each place gets its own
        # copy of the entity (the stored shape has no shared catalog).
        out = []
        for i in dict.fromkeys(value):
            if i not in entities:
                continue
            if entities[i].name.lower() in _GENERIC_NAMES:
                logger.info("dropped generic entity %r from %s", entities[i].name, where)
                continue
            if text is not None and not _mentions(entities[i], text):
                logger.info(
                    "dropped ungrounded entity %r from %s", entities[i].name, where
                )
                continue
            out.append(entities[i].model_copy())
        return out

    def with_stable_actors(found: list[VisualEntity], text: str | None) -> list[VisualEntity]:
        """Put Subagent / Main Agent first wherever the text names them but the model
        left them out (the catalog entity is reused, else one is created)."""
        if text is None:
            return found
        have = {e.name.lower() for e in found}
        added = []
        for name, pattern in _STABLE_ACTORS:
            if name.lower() in have or not pattern.search(text):
                continue
            known = next((e for e in entities.values() if e.name.lower() == name.lower()), None)
            added.append((known or VisualEntity(name=name, kind="actor", aliases=[])).model_copy())
            logger.info("added missing stable entity %r", name)
        return added + found

    question_refs = with_stable_actors(
        refs(
            parsed.get("question_entities"),
            "'question_entities'",
            texts[0] if texts else None,
        ),
        texts[0] if texts else None,
    )
    raw_answers = parsed.get("answers_entities")
    if not isinstance(raw_answers, list) or len(raw_answers) != n_answers:
        got = len(raw_answers) if isinstance(raw_answers, list) else type(raw_answers).__name__
        raise bad(f"'answers_entities' must be a list of {n_answers} lists, got {got}")
    answers = []
    for i, r in enumerate(raw_answers):
        text = texts[1][i] if texts and i < len(texts[1]) else None
        found = refs(r, f"'answers_entities[{i}]'", text)
        answers.append(AnswerEntities(entities=with_stable_actors(found, text)[:MAX_PER_ANSWER]))
    return PromptEntities(question=question_refs, answers=answers)


def _source_texts(item: dict, simplified: SimplifiedPrompt) -> tuple[str, list[str]]:
    """Original and simplified text of the question and of each answer, joined, for
    the grounding check in `parse_entities`."""
    prompt = item.get("prompt") or {}
    originals = [clean_plain(a) for a in prompt.get("answers", [])]
    return (
        f"{clean_question(prompt.get('question', ''))}\n{simplified.question}",
        [
            f"{originals[i] if i < len(originals) else ''}\n{a}"
            for i, a in enumerate(simplified.answers)
        ],
    )


async def extract_entities(
    ica: IcaClient,
    item: dict,
    simplified: SimplifiedPrompt,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[PromptEntities, IcaMeta]:
    """
    Ask the ICA chat model for the visual entities of one question. `item` is the
    original examination document, `simplified` its simplified prompt (whose text the
    resolved entities are index-aligned with its answers).
    """
    n_answers = len(simplified.answers)
    data = await ica.chat_model(
        model_id,
        _build_user_message(item, simplified),
        system=_build_system_message(n_answers),
        max_tokens=max_tokens,
    )
    choice = (data.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        raise HTTPException(
            status_code=502,
            detail=f"ICA model stopped with finish_reason={finish_reason!r} — cannot parse entities",
        )
    parsed = extract_json((choice.get("message") or {}).get("content") or "")
    return (
        parse_entities(parsed, n_answers, _source_texts(item, simplified)),
        IcaMeta(
            model_id=model_id,
            finish_reason=finish_reason,
            total_tokens=(data.get("usage") or {}).get("total_tokens", 0),
        ),
    )


async def enrich_and_store(
    ica: IcaClient,
    simplified: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> SimplifiedQuestion:
    """
    Backfill: extract entities for an already simplified question and store only
    `entities` / `entities_ica` (text, `ica` and `created_at` stay as they are).
    Errors propagate, unlike inside `simplify_and_store`.
    """
    question_id = str(doc["_id"])
    existing = await simplified.find_one({"question_id": question_id})
    if existing is None:
        raise HTTPException(status_code=404, detail="simplified question not found")

    prompt = SimplifiedPrompt(**existing["prompt"])
    entities, meta = await extract_entities(ica, doc, prompt, model_id, max_tokens)
    # Conditional on the record not having been rewritten meanwhile: the references
    # index into the text that was sent, so they must not land on a newer text.
    stored = await simplified.find_one_and_update(
        {"question_id": question_id, "updated_at": existing["updated_at"]},
        {
            "$set": {
                "entities": entities.model_dump(),
                "entities_ica": meta.model_dump(),
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(
            status_code=409,
            detail="simplified question changed while entities were extracted; retry",
        )
    return SimplifiedQuestion(**stored)
