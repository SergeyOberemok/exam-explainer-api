"""Visual entity extraction for simplified questions (second ICA call).

The prompt below is the machine-readable version of `docs/extracting_actor_entities.md`:
same selection rules, but the reply is one JSON object holding an entity catalog plus
references (entity ids) for the question text and for each answer, so the app can map
text to diagram elements by structure.
"""

import logging
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

_SYSTEM_MESSAGE = """\
You extract reusable visual entities from one multiple-choice exam question so an app \
can draw them as diagram elements (icons, boxes, arrows). The exam is about Claude \
Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool use and \
agent design.

SOURCES
- ORIGINAL text is the primary source: take exact names of technologies, tools, \
settings, files, people and systems from it. Every tool, command, parameter, setting, \
mode, file or agent the original NAMES gets its own entity with that exact name, even \
when the simplified text drops it (for example a list "get_customer, lookup_order, \
process_refund" gives three entities, not one "Tools" entity). List these first.
- SIMPLIFIED text is the secondary source: use it to understand context and to find \
entities the original only implies. Never replace an original name with a paraphrase.
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
- "action": an operation that can be one workflow step (Process Refund, Lookup Order, \
Escalate to Human). A tool's name is an object; the operation it performs is an action.

GROUPING
- Merge synonyms and variants of the same thing into one entity (customer + customer \
details -> "Customer"; order + order lookup data -> "Order").
- Name = the original technical term when there is one (`stop_reason`, `acceptEdits`, \
plan mode, `git diff`, MCP); otherwise the shortest clear Title Case noun for the \
group. Put other wordings from the text in "aliases".
- Do not make separate entities for attributes, versions or properties of one thing.
- Reuse one name for the same thing everywhere in this reply. Use the stable names \
"Claude Agent SDK", "Claude Code", "Claude API", "MCP Server", "Subagent", \
"Main Agent" for those things instead of inventing variants.
- "aliases" are only other wordings of the SAME thing found in the text: never the \
name itself, never counts or numbers ("18 tools"), never a different thing.

EXCLUDE
Purely abstract ideas are never entities: confidence, accuracy, reasoning, \
independence, verification, bias, threshold, reliability, decision quality, \
performance metrics, cost, latency. Also leave out events, situations and facts \
(a billing dispute, a failed audit, misrouting), message or conversation concepts \
(context, history, a follow-up message) and generic groups of things already listed \
individually ("MCP Tools" when each tool is named). Do not invent anything the text \
does not support. Do not list every noun; keep a compact set (at most 16 entities, \
at most 4 per answer).

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
    return _SYSTEM_MESSAGE.replace(
        "{{ANSWERS_SCHEMA}}", f'"answers_entities": [\n    {entries}\n  ]'
    ).replace("{{N}}", str(n_answers))


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


def parse_entities(parsed: dict, n_answers: int) -> PromptEntities:
    """
    Validate the model's reply and build the stored structure. The reply uses a
    catalog plus id references (keeps names consistent and the reply short); the
    stored form resolves them: `question: [entity]`, `answers: [{entities: [entity]}]`.
    Structural problems (bad kind, duplicate id, wrong answer count, wrong types)
    raise 502. Overflow is trimmed instead of failing the reply: entities beyond
    MAX_ENTITIES, references to ids that do not exist (or were trimmed), and more than
    MAX_PER_ANSWER entities on an answer.
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

    def refs(value: object, where: str) -> list[VisualEntity]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise bad(f"{where} must be a list of entity ids")
        # Unknown ids and repeats are dropped; order is kept. Each place gets its own
        # copy of the entity (the stored shape has no shared catalog).
        return [entities[i].model_copy() for i in dict.fromkeys(value) if i in entities]

    question_refs = refs(parsed.get("question_entities"), "'question_entities'")
    raw_answers = parsed.get("answers_entities")
    if not isinstance(raw_answers, list) or len(raw_answers) != n_answers:
        got = len(raw_answers) if isinstance(raw_answers, list) else type(raw_answers).__name__
        raise bad(f"'answers_entities' must be a list of {n_answers} lists, got {got}")
    return PromptEntities(
        question=question_refs,
        answers=[
            AnswerEntities(entities=refs(r, f"'answers_entities[{i}]'")[:MAX_PER_ANSWER])
            for i, r in enumerate(raw_answers)
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
        parse_entities(parsed, n_answers),
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
