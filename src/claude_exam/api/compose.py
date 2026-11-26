"""The game document of a question (`game_question`): the pipeline's last step, no ICA call.

The game reads one document per question and nothing else. This step puts it together
from what the earlier steps stored, so the rules for what is usable (a decomposition
written for the question's text as it is, a usable story record, icons still in the
catalog, ...) live here and not in the game's request handlers.

The document is shaped for how a player meets the question (`schemas.GameCard`):

1. `briefing`: the question's semantic parts in the order context, current state,
   constraint, problem, consequence, goal, shown one by one, then the `ask`.
2. `tools`: one per answer: its text, what it proposes (`label`, the decomposition's
   unjudged `answer_options[].mechanism`) and the `cast` members it is made of
   (`components`: concepts included, which tell the answers apart).
3. `map`: the pre-answer flow, unchanged.
4. After answering, `key.outcomes[answer]`: the answer's verdict, `impact` and
   `consequence` (written by the resolve step), its misconception and its gaps.

`public` is everything sent before answering; `key` stays in the round entry. Nothing
in `public` comes from the answer key: the parts, cast, tools and map are built from
the key-free steps, and `tests/test_compose.py` checks it with sentinel strings.

A question is composed only when everything is there: a playable question, a current
decomposition with icons, flow and a resolution with consequences, a usable story
record and objectives. Otherwise the step answers 422 with what is missing, and an
older document of the question is deleted (the game stops serving it).

The input key is a hash of the composed document itself, so the stored one is current
exactly while composing again would give the same document: any upstream change that
shows in the game (a new beat, a new icon, a new resolution, a catalog key that left)
outdates it, and nothing has to be bumped.
"""

import json
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api import hold
from claude_exam.api.decompose import has_flow, has_resolution, read_decomposition
from claude_exam.api.game.domain import (
    MAX_ANSWERS,
    answers_of,
    correct_indices,
    difficulty_of,
    entities_of,
    flow_of,
    game_flow,
    misconceptions_of,
    objectives_of,
    outcomes_of,
    parts_of,
    story_of,
    to_review,
    usable_story,
)
from claude_exam.api.leaks import answer_only_entities
from claude_exam.api.question_view import key_free, the_ask
from claude_exam.api.scenario_catalog import text_hash
from claude_exam.api.schemas import (
    GameAsk,
    GameCard,
    GameCardKey,
    GameCastMember,
    GameOutcome,
    GameTool,
    Resolution,
)
from claude_exam.api.step_key import step_key

# Bump when the same records compose into a different document.
CODE_VERSION = "1"

# The order of the briefing: where the player is, what holds, what they may not change,
# what goes wrong and what it costs, then what they want. The ask comes last, apart.
BRIEFING_ORDER = ("context", "current_state", "constraint", "problem", "consequence", "goal")
# An entity the icon step drew keeps its icon; the others (concepts, which it does not
# draw) get one by type.
_TYPE_ICON = {"actor": "actor", "concept": "idea"}
_DEFAULT_ICON = "object"


def _resolution(doc: dict, record: dict) -> Resolution | None:
    """The stored resolution if it fits the question: one option per answer, verdicts
    matching the key and a consequence for every option (one written before
    consequences existed is not enough for the game)."""
    try:
        resolution = Resolution.model_validate(record["resolution"])
    except (ValidationError, KeyError, TypeError):
        return None
    count = len(answers_of(doc))
    correct = set(correct_indices(doc))
    options = sorted(resolution.options, key=lambda o: o.index)
    if [o.index for o in options] != list(range(count)):
        return None
    if any((o.verdict == "correct") != (o.index in correct) for o in options):
        return None
    if any(not o.consequence.strip() for o in options):
        return None
    resolution.options = options
    return resolution


def missing(
    doc: dict, objectives: dict | None, story: dict | None, decomposition: dict | None
) -> list[str]:
    """What the question still lacks before it can be composed ([] when nothing)."""
    lacks = []
    if not (2 <= len(answers_of(doc)) <= MAX_ANSWERS and len(correct_indices(doc)) == 1):
        lacks.append("a playable question (2 to 4 answers, exactly one correct)")
    semantic = read_decomposition(decomposition, doc)
    if semantic is None:
        lacks.append("decomposition")
    else:
        if not (decomposition or {}).get("icons_ica"):
            lacks.append("icons")
        if not has_flow(decomposition):
            lacks.append("flow")
        elif not has_resolution(decomposition):
            lacks.append("resolution")
        elif not lacks and _resolution(doc, decomposition) is None:
            lacks.append("resolution with a consequence per answer")
    if usable_story(doc, story) is None:
        lacks.append("story")
    if not objectives_of(objectives):
        lacks.append("objectives")
    return lacks


def compose(
    doc: dict, objectives: dict | None, story: dict | None, decomposition: dict | None
) -> tuple[dict | None, list[str]]:
    """The game document of a question (`public`, `key` and the fields a round filters
    and orders by), or None and what is missing. Pure."""
    lacks = missing(doc, objectives, story, decomposition)
    if lacks:
        return None, lacks
    semantic = read_decomposition(decomposition, doc)
    resolution = _resolution(doc, decomposition)
    answers = answers_of(doc)
    count = len(answers)
    q = key_free(doc)
    place = story_of(doc, story)
    record = usable_story(doc, story) or {}

    entities = entities_of(decomposition, count)
    icon_by_ref: dict[str, str] = {}
    for entity in [*entities.question, *(e for a in entities.answers for e in a.entities)]:
        if entity.ref and entity.icon:
            icon_by_ref.setdefault(entity.ref, entity.icon)
    answer_only = {e.id for e in answer_only_entities(semantic, q)}
    cast = [
        GameCastMember(
            ref=e.id,
            name=e.name,
            type=e.type,
            aliases=list(e.aliases),
            icon=icon_by_ref.get(e.id) or _TYPE_ICON.get(e.type, _DEFAULT_ICON),
            scene=e.id not in answer_only,
        )
        for e in semantic.entities
    ]
    refs = {member.ref for member in cast}

    parts = parts_of(decomposition)
    asks = [p for p in parts if p.role == "ask"]
    rank = {role: i for i, role in enumerate(BRIEFING_ORDER)}
    briefing = sorted(
        (p for p in parts if p.role != "ask"), key=lambda p: rank.get(p.role, len(rank))
    )
    ask = (
        GameAsk(text=asks[-1].text, entity_ids=[i for i in asks[-1].entity_ids if i in refs])
        if asks
        else GameAsk(text=the_ask(q) or q.stem)
    )

    proposals = {o.index: o for o in semantic.answer_options}
    tools = []
    for i, text in enumerate(answers):
        option = proposals.get(i)
        tools.append(
            GameTool(
                index=i,
                text=text or "",
                label=option.mechanism if option else "",
                icon=entities.answers[i].icon or "",
                components=[r for r in (option.entity_ids if option else []) if r in refs],
            )
        )

    card = GameCard(
        id=str(doc["_id"]),
        section=doc.get("section") or "",
        assessment_type=doc.get("assessment_type") or "",
        scenario=place["scenario"],
        stage=place["stage"],
        objectives=objectives_of(objectives),
        difficulty=difficulty_of(objectives),
        icon=entities.question_icon or "",
        beat=place["beat"],
        question=(doc.get("prompt") or {}).get("question") or "",
        stem=place["stem"],
        briefing=briefing,
        ask=ask,
        cast=cast,
        tools=tools,
        map=flow_of(doc, decomposition),
    )

    correct = correct_indices(doc)[0]
    misconceptions = misconceptions_of(objectives, count, correct)
    lines = outcomes_of(doc, story)
    key = GameCardKey(
        correct=correct,
        review=to_review(doc),
        outcomes=[
            GameOutcome(
                index=o.index,
                verdict=o.verdict,
                impact=o.impact,
                consequence=o.consequence,
                misconception=misconceptions[o.index],
                reason=o.reason,
                gaps=list(o.gaps),
            )
            for o in resolution.options
        ],
        rule=resolution.rule,
        answer_requirements=list(resolution.answer_requirements),
        win=lines["win"],
        fail=lines["fail"],
        resolution_flow=game_flow(resolution.flow.model_dump()),
    )
    return {
        "section": card.section,
        "assessment_type": card.assessment_type,
        "scenario": card.scenario,
        "stage": card.stage,
        "order": record.get("order") if isinstance(record.get("order"), int) else 0,
        "objectives": card.objectives,
        "difficulty": card.difficulty,
        "count": count,
        "public": card.model_dump(),
        "key": key.model_dump(),
    }, []


def key_of(composed: dict) -> str:
    """The input key of a composed document: a hash of what it holds."""
    return step_key(CODE_VERSION, json.dumps(composed, sort_keys=True, ensure_ascii=False))


def input_key(
    doc: dict, objectives: dict | None, story: dict | None, decomposition: dict | None
) -> str | None:
    """The key the question's game document would have if it were composed today (None
    when it cannot be composed)."""
    composed, _ = compose(doc, objectives, story, decomposition)
    return None if composed is None else key_of(composed)


async def compose_and_store(
    store: AsyncCollection,
    doc: dict,
    objectives: dict | None,
    story: dict | None,
    decomposition: dict | None,
) -> dict:
    """Write the question's game document (upsert by `question_id`). When the question
    cannot be composed, its old document is deleted and the step answers 422 with what
    is missing (a job does not retry a 422)."""
    question_id = str(doc["_id"])
    composed, lacks = compose(doc, objectives, story, decomposition)
    if composed is None:
        await store.delete_one({"question_id": question_id})
        raise HTTPException(
            status_code=422, detail=f"not ready for the game, missing: {', '.join(lacks)}"
        )
    fields = {
        **composed,
        "question_id": question_id,
        "text_hash": text_hash(doc),
        "input_key": key_of(composed),
        "held": hold.is_held(doc),
        "updated_at": datetime.now(timezone.utc),
    }
    await store.find_one_and_update({"question_id": question_id}, {"$set": fields}, upsert=True)
    return fields


async def set_held(store: AsyncCollection | None, question_id: str, held: bool) -> None:
    """Mirror a question's hold (`hold.py`) on its game document, so a held question is
    not served."""
    if store is not None:
        await store.update_one({"question_id": question_id}, {"$set": {"held": held}})
