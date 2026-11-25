"""The visual entities of a question: who and what a diagram draws (no ICA call).

They are taken from the question's decomposition (`decompose.py`), which was written
without the answer key, so they may be shown before the player answers: its drawable
entities become the catalog, the question names all of them and each answer lists the
entities its answer option relies on first. `parse_entities` then grounds them in the
original wording, adds the stable actors, drops generic names and applies the limits.
Each entity keeps its decomposition id as `ref` (what a part's and a flow node's
`entity_ids` match). They are stored with the decomposition, so writing a new one
replaces them; `icons.py` adds a picture to each.
"""

import logging
import re

from fastapi import HTTPException

from claude_exam.api.question_view import KeyFreeQuestion
from claude_exam.api.schemas import (
    AnswerEntities,
    PromptEntities,
    QuestionDecomposition,
    VisualEntity,
)

logger = logging.getLogger(__name__)

# Entity types a diagram draws, and the `VisualEntity.kind` each becomes. A `concept`
# stays in the decomposition but is never drawn.
VISUAL_KIND = {
    "actor": "actor",
    "system": "object",
    "object": "object",
    "knowledge_source": "object",
    "artifact": "object",
    "external_service": "object",
}

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
        ref = raw.get("ref") if isinstance(raw.get("ref"), str) else ""
        entities[entity_id] = VisualEntity(
            name=name, kind=kind, aliases=clean_aliases, ref=ref
        )
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


def _source_texts(question: KeyFreeQuestion) -> tuple[str, list[str]]:
    """The question's text and each answer's, for the grounding check in
    `parse_entities` (an entity is kept only where its name or an alias occurs)."""
    return question.question, list(question.answers)


def derive_entities(decomposition: QuestionDecomposition, question: KeyFreeQuestion) -> PromptEntities:
    """The visual entities of a question taken from its decomposition (see the module
    docstring). `question` is its key-free view: the answers count and ground them."""
    catalog = [
        {
            "id": e.id,
            "name": e.name,
            "kind": VISUAL_KIND[e.type],
            "aliases": e.aliases,
            "ref": e.id,
        }
        for e in decomposition.entities
        if e.type in VISUAL_KIND
    ]
    ids = [e["id"] for e in catalog]
    options = {o.index: o.entity_ids for o in decomposition.answer_options}
    count = len(question.answers)
    answers = [
        list(dict.fromkeys([*(i for i in options.get(n, []) if i in ids), *ids]))
        for n in range(count)
    ]
    reply = {"entities": catalog, "question_entities": ids, "answers_entities": answers}
    return parse_entities(reply, count, _source_texts(question))
