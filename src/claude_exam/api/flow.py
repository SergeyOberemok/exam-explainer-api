"""Decision flow and resolution of a question (two ICA calls): stage 2 of
`docs/question_decomposition_decision_flow_spec.md`, split by what the player may see.

- `flow` (before answering, key-free): built from the decomposition only. It shows the
  situation as nodes and edges (what happens, where it goes wrong, the conditions that
  branch it) and ends in exactly ONE decision node marked `is_question`: the question
  itself. That node has no outgoing edges; the game hangs the answer cards on it. So the
  flow cannot point at an answer, and the model that drew it never saw the key.
- `resolve` (after answering, sees the key, the feedback and the explanation): what a
  right answer has to do (`answer_requirements`), the flow again with the correct
  mechanism in place of the question node (its nodes and edges marked `added`), and
  each option's verdict, the requirements it meets and its gaps (a comparison matrix).
  This is part of the answer key: the game keeps it in the round entry and sends it only
  in the answer result.

Both replies must pass `graph_errors` (the spec's flow checklist as code): one start,
edges between existing nodes, every node reachable, every decision with two or more
labelled branches, every path ending in an end node (or the question node), ids that
resolve into the decomposition, short labels and no invented numbers. A failing reply is
sent back with its faults. Stored on the question's `question_decomposition` record.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.decompose import (
    FLOW_VERSION,
    RESOLUTION_VERSION,
    answer_only_entities,
    flow_is_current,
    resolution_is_current,
    invented_numbers,
    names_entity,
    read_decomposition,
    scenario_view,
    source_parts,
)
from claude_exam.api.game.domain import correct_indices
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_validated, schema_skeleton
from claude_exam.api.schemas import (
    DecisionFlow,
    DecompositionRecord,
    IcaMeta,
    IcaModelId,
    QuestionDecomposition,
    Resolution,
)
from claude_exam.html_clean import clean_plain

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_NODES = 14
# The prompt asks for labels of at most MAX_LABEL_WORDS words; the gate allows a few
# more, because a model cannot count words exactly and an 11-word label is no fault
# worth three replies.
MAX_LABEL_WORDS = 10
LABEL_WORD_LIMIT = 14
# Edge types that name their own branch; a `custom` or `next` branch of a decision
# needs a label.
_SELF_LABELLED = frozenset({"yes", "no", "unknown", "approved", "rejected"})

_FLOW_RULES = """\
FLOW RULES
- Node types: start (entry point), process (an action), decision (a question or test \
with branches), input (information a step needs), output (a result), human_review \
(human judgment or approval), end (a terminal state).
- Exactly one start node. Start with the event that initiates the scenario.
- Include only the steps needed to understand the question (at most {{MAX_NODES}} \
nodes). Labels are short (at most {{MAX_LABEL_WORDS}} words), concrete, in plain words.
- Conditions go in decision nodes, phrased as a question or test, never in process \
nodes. A decision has two or more outgoing edges, each with a type (yes, no, unknown, \
approved, rejected, custom, next) and a label that states the condition of that \
branch. Give a decision the id of the condition it tests (condition_id) when the \
decomposition has one, else "".
- Include an unknown or escalation branch when uncertainty matters. Do not equate \
uncertainty with a negative result. Keep automated actions apart from human judgment; \
never imply a human approved something the scenario does not say.
- Every branch reaches a meaningful end node (or the question node, see below). No \
dead ends, no edge to a node that does not exist.
- entity_ids name the decomposition entities a node shows. source_status: "explicit" \
when the question states the step, "inferred" when it only connects the scenario.
- Do not add facts, numbers, thresholds or actors the question does not give.
- Edges go from "source" to "target" (node ids n1, n2, ...).

VISUALIZATION - choose the primary view that matches the question's reasoning:
decision_flow (branching, routing, choosing an action), process_flow (ordered steps \
without meaningful branches), sequence_diagram (messages between participants), \
architecture_diagram (components and responsibilities), cause_effect_diagram (why a \
problem occurs), comparison_matrix (weighing alternatives), entity_relationship_diagram, \
timeline. The nodes and edges are drawn in that view: a question without branching \
gets a straight process or cause-and-effect chain, not invented decisions.\
"""

_FLOW_SYSTEM = """\
You draw the flow of one multiple-choice question of an exam about Claude Code, the \
Claude Agent SDK, the Claude API, MCP, prompt engineering, tool use and agent design. \
A learner sees it BEFORE answering, next to the question, to grasp the situation.

You get the question's decomposition (entities, events, problem, conditions, goals) \
and the question. You are not told which answer is correct, and the flow must not hint \
at any answer.

WHAT THE FLOW SHOWS
- The situation as it is now: what starts it, what happens, where it goes wrong \
(symptom, a cause only when the question states it, consequence) and the conditions \
the question states.
- Branch only on a condition the decomposition marks "explicit" (its id as \
condition_id). A cause, check or test the question leaves open is what the learner \
must work out: never draw it - let the flow reach the question node instead. Do not \
use what you know about these products to fill it in.
- It ends in exactly ONE decision node with "is_question": true: the question the \
learner must answer, in the question's own words - never a guess or a test of a \
cause ("Is it caused by X?"). That node has NO outgoing edges - the answer \
options are attached to it later. Every other path ends in an end node.
- Do not draw any answer option, mechanism or fix. "added" is false everywhere.

{{RULES}}

{{OUTPUT}}\
"""

_RESOLVE_SYSTEM = """\
You explain the resolution of one multiple-choice question of an exam about Claude \
Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool use and agent \
design. A learner sees it AFTER answering.

You get the question, its answers with the correct one marked, the per-answer feedback \
and the official explanation (the source of truth: do not invent technical facts), the \
question's decomposition and the flow the learner saw before answering, without the \
node where the learner had to choose (the nodes that led to it are named).

WRITE
1. answer_requirements: what any correct answer must achieve, as short phrases taken \
from the goals and the explanation (2 to 5).
2. flow: the flow the learner saw, continued. Keep all its nodes and edges with the \
same ids, and from the nodes the learner chose after, add the steps the CORRECT \
answer's mechanism adds, marked "added": true (nodes and the edges that touch them), \
leading to end nodes with the outcome the explanation gives. New node ids continue the \
numbering. No node has "is_question": true. Keep it small: add only what shows how \
the correct mechanism fixes the problem.
3. options: one per answer, same order (A is index 0): verdict ("correct" for the \
correct answer only), the requirements it supports (from answer_requirements), its \
gaps (why it falls short; [] for the correct one), and one sentence of reason.
4. rule: one line a learner can take to similar questions.

{{RULES}}

{{OUTPUT}}\
"""


def _system(template: str, model) -> str:
    """A system message with the flow rules and the exact shape of the reply."""
    output = (
        "OUTPUT FORMAT - respond with ONLY one JSON object of exactly this shape, every "
        "field present ([] for an empty list), no markdown fences, no prose:\n"
        + schema_skeleton(model)
    )
    return template.replace("{{RULES}}", _rules()).replace("{{OUTPUT}}", output)


def _rules() -> str:
    return _FLOW_RULES.replace("{{MAX_NODES}}", str(MAX_NODES)).replace(
        "{{MAX_LABEL_WORDS}}", str(MAX_LABEL_WORDS)
    )


def _compact(model) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False, separators=(",", ":"))


def _question_block(doc: dict, with_key: bool) -> str:
    paragraph, stem, answers = source_parts(doc)
    parts = []
    if paragraph:
        parts.append(f"SCENARIO (background):\n{paragraph}")
    parts.append(f"QUESTION:\n{stem}")
    if with_key:
        correct = set(correct_indices(doc))
        feedbacks = [clean_plain(f) for f in (doc.get("prompt") or {}).get("feedbacks", [])]
        lines = "\n".join(
            f"  {chr(65 + i)}. {a}{'  [CORRECT]' if i in correct else ''}"
            + (f"\n     feedback: {feedbacks[i]}" if i < len(feedbacks) else "")
            for i, a in enumerate(answers)
        )
        explanation = clean_plain((doc.get("prompt") or {}).get("explanation", ""))
        parts += [f"ANSWERS:\n{lines}", f"OFFICIAL EXPLANATION:\n{explanation}"]
    return "\n\n".join(parts)


def graph_errors(
    flow: DecisionFlow,
    decomposition: QuestionDecomposition,
    source: str,
    *,
    pre_answer: bool,
) -> list[str]:
    """The flow checklist of the spec as sentences the model can act on (empty when the
    flow passes). `pre_answer` adds the question-node rule and forbids `added`; without
    it the question node must be gone and something must be `added`."""
    errors: list[str] = []
    nodes = {}
    for node in flow.nodes:
        if node.id in nodes:
            errors.append(f"node id {node.id!r} is used twice")
        nodes[node.id] = node
    if not nodes:
        return ["the flow has no nodes"]
    if len(nodes) > MAX_NODES:
        errors.append(f"at most {MAX_NODES} nodes, got {len(nodes)}")

    out: dict[str, list] = defaultdict(list)
    incoming: dict[str, list] = defaultdict(list)
    for edge in flow.edges:
        missing = [n for n in (edge.source, edge.target) if n not in nodes]
        if missing:
            errors.append(
                f"edge {edge.source}->{edge.target} uses nodes that do not exist: "
                f"{', '.join(missing)}"
            )
            continue
        out[edge.source].append(edge)
        incoming[edge.target].append(edge)

    starts = [n for n in nodes.values() if n.type == "start"]
    if len(starts) != 1:
        errors.append(f"the flow needs exactly one start node, got {len(starts)}")
    questions = [n for n in nodes.values() if n.is_question]
    if pre_answer:
        if len(questions) != 1 or questions[0].type != "decision":
            errors.append(
                "the flow needs exactly one decision node with is_question true (the "
                "question itself)"
            )
        for q in questions:
            if out[q.id]:
                errors.append(
                    f"the question node {q.id} must have no outgoing edges: the answers "
                    "are attached to it later"
                )
        if any(n.added for n in nodes.values()) or any(e.added for e in flow.edges):
            errors.append("the flow before answering adds nothing: set every added to false")
    else:
        if questions:
            errors.append(
                "the resolved flow replaces the question node: no node may have "
                "is_question true"
            )
        if not any(n.added for n in nodes.values()):
            errors.append("mark the nodes the correct answer's mechanism adds with added true")

    terminal = {n.id for n in nodes.values() if n.type == "end" or n.is_question}
    if not terminal:
        errors.append("the flow needs at least one end node")
    entity_ids = {e.id for e in decomposition.entities}
    condition_ids = {c.id for c in decomposition.conditions}
    explicit_conditions = {
        c.id for c in decomposition.conditions if c.source_status == "explicit"
    }
    for node in nodes.values():
        words = len(node.label.split())
        if not node.label.strip():
            errors.append(f"node {node.id} needs a label")
        # The question node gets the question's own words (`with_question_label`).
        elif words > LABEL_WORD_LIMIT and not node.is_question:
            errors.append(f"node {node.id} label has {words} words; at most {MAX_LABEL_WORDS}")
        if node.type != "decision" and node.label.strip().endswith("?"):
            errors.append(
                f"node {node.id} asks a question but is a {node.type} node: only a "
                "decision node asks (and before answering, only one the question states)"
            )
        unknown = [i for i in node.entity_ids if i not in entity_ids]
        if unknown:
            errors.append(f"node {node.id} refers to entities that do not exist: {', '.join(unknown)}")
        if node.condition_id and node.condition_id not in condition_ids:
            errors.append(f"node {node.id} refers to a condition that does not exist: {node.condition_id}")
        if node.type == "end" and out[node.id]:
            errors.append(f"end node {node.id} must have no outgoing edges")
        if node.id not in terminal and not out[node.id]:
            errors.append(f"node {node.id} is a dead end: give it an outgoing edge or make it an end node")
        if node.type == "decision" and not node.is_question:
            branches = out[node.id]
            if len(branches) < 2:
                errors.append(f"decision {node.id} needs two or more outgoing branches")
            # Before answering, a branch rests only on what the question states: a test
            # of a cause it leaves open is the answer, worked out by the model.
            if pre_answer and node.condition_id not in explicit_conditions:
                errors.append(
                    f"decision {node.id} must test a condition the decomposition marks "
                    "explicit (its condition_id); the question does not state this one, "
                    "so leave the decision out and let the flow reach the question node"
                )
            for edge in branches:
                if edge.type not in _SELF_LABELLED and not edge.label.strip():
                    errors.append(
                        f"branch {edge.source}->{edge.target} needs a label stating its condition"
                    )

    if len(starts) == 1:
        reached, todo = {starts[0].id}, [starts[0].id]
        while todo:
            for edge in out[todo.pop()]:
                if edge.target not in reached:
                    reached.add(edge.target)
                    todo.append(edge.target)
        unreachable = [n for n in nodes if n not in reached]
        if unreachable:
            errors.append(f"nodes not reachable from the start: {', '.join(unreachable)}")
    # Every node must be able to reach a terminal (catches cycles without an exit).
    finishing, todo = set(terminal), list(terminal)
    while todo:
        for edge in incoming[todo.pop()]:
            if edge.source not in finishing:
                finishing.add(edge.source)
                todo.append(edge.source)
    stuck = [n for n in nodes if n not in finishing]
    if stuck and terminal:
        errors.append(f"no path to an end node from: {', '.join(stuck)}")

    numbers = invented_numbers(
        source,
        [flow.trigger, *(n.label for n in nodes.values()), *(e.label for e in flow.edges)],
    )
    if numbers:
        errors.append(f"these numbers are not in the question; remove them: {', '.join(numbers)}")
    return errors


def _parsed(model, parsed: dict) -> tuple[object | None, list[str]]:
    try:
        return model.model_validate(parsed), []
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
        ]


def _source(doc: dict) -> str:
    paragraph, stem, answers = source_parts(doc)
    return "\n".join([paragraph, stem, *answers])


def flow_errors(doc: dict, decomposition: QuestionDecomposition, parsed: dict) -> list[str]:
    """The gate of a pre-answer flow. It is checked against the decomposition's
    `scenario_view` (what the model was given), and no label may name an entity only an
    answer option mentions."""
    flow, errors = _parsed(DecisionFlow, parsed)
    if errors:
        return errors
    view = scenario_view(decomposition, doc)
    errors = graph_errors(flow, view, _source(doc), pre_answer=True)
    answer_only = answer_only_entities(decomposition, doc)
    texts = [
        ("the trigger", flow.trigger),
        *((f"node {n.id}", n.label) for n in flow.nodes),
        *((f"edge {e.source}->{e.target}", e.label) for e in flow.edges),
    ]
    for where, text in texts:
        named = names_entity(text, answer_only)
        if named:
            errors.append(
                f"{where} names {named!r}, which only an answer option mentions: the "
                "flow before answering shows the scenario, not what the answers propose"
            )
    return errors


def resolution_errors(
    doc: dict, decomposition: QuestionDecomposition, parsed: dict
) -> list[str]:
    resolution, errors = _parsed(Resolution, parsed)
    if errors:
        return errors
    source = "\n".join(
        [
            _source(doc),
            clean_plain((doc.get("prompt") or {}).get("explanation", "")),
            *(clean_plain(f) for f in (doc.get("prompt") or {}).get("feedbacks", [])),
        ]
    )
    errors = graph_errors(resolution.flow, decomposition, source, pre_answer=False)
    n_answers = len((doc.get("prompt") or {}).get("answers", []))
    if sorted(o.index for o in resolution.options) != list(range(n_answers)):
        errors.append(f"options must have one item per answer, with index 0 to {n_answers - 1}")
    correct = set(correct_indices(doc))
    wrong = [
        chr(65 + o.index)
        for o in resolution.options
        if (o.verdict == "correct") != (o.index in correct)
    ]
    if wrong:
        errors.append(
            f"the verdict of option(s) {', '.join(wrong)} does not match the correct answer"
        )
    if not resolution.answer_requirements:
        errors.append("answer_requirements must list what a correct answer has to achieve")
    if not resolution.rule.strip():
        errors.append("rule must be one line a learner can take to similar questions")
    return errors


_WORD = re.compile(r"[a-z][a-z0-9_-]{3,}")


def leak_warning(doc: dict, flow: DecisionFlow) -> str:
    """A soft check of a pre-answer flow (never rejects): when the flow's labels share
    clearly more words with the correct answer than with any other, the flow may point
    at it. "" when nothing stands out."""
    labels = set(_WORD.findall(" ".join(n.label for n in flow.nodes).lower()))
    answers = source_parts(doc)[2]
    stem = set(_WORD.findall(source_parts(doc)[1].lower()))
    overlap = [len((set(_WORD.findall(a.lower())) - stem) & labels) for a in answers]
    correct = correct_indices(doc)
    if len(correct) != 1 or len(overlap) < 2:
        return ""
    best = overlap[correct[0]]
    others = max(o for i, o in enumerate(overlap) if i != correct[0])
    if best >= 2 and best >= 2 * max(others, 1):
        return (
            f"the flow shares {best} words with the correct answer "
            f"{chr(65 + correct[0])} and at most {others} with the others"
        )
    return ""


def prune_after_question(parsed: dict) -> dict:
    """A pre-answer reply without what comes after the question node: its outgoing
    edges, and the nodes only they lead to. The model tends to draw the ways out of the
    question, and those are the answers, which must not show before answering; cutting
    them in code is surer than asking again. A reply of the wrong shape, or without a
    question node, is returned as it is (the gate then names the fault)."""
    nodes, edges = parsed.get("nodes"), parsed.get("edges")
    if not (isinstance(nodes, list) and isinstance(edges, list)):
        return parsed
    if not all(isinstance(n, dict) for n in nodes) or not all(isinstance(e, dict) for e in edges):
        return parsed
    questions = {n.get("id") for n in nodes if n.get("is_question") is True}
    if not questions:
        return parsed
    kept_edges = [e for e in edges if e.get("source") not in questions]
    starts = [n.get("id") for n in nodes if n.get("type") == "start"]
    reached, todo = set(starts), list(starts)
    while todo:
        current = todo.pop()
        for edge in kept_edges:
            if edge.get("source") == current and edge.get("target") not in reached:
                reached.add(edge.get("target"))
                todo.append(edge.get("target"))
    # Only nodes the cut edges led to go; a node that was unreachable anyway stays, so
    # the gate still reports a broken flow.
    before = set(starts)
    todo = list(starts)
    while todo:
        current = todo.pop()
        for edge in edges:
            if edge.get("source") == current and edge.get("target") not in before:
                before.add(edge.get("target"))
                todo.append(edge.get("target"))
    cut = (before - reached) - questions
    # A node without any edge left (often an end the model kept for an answer branch) is
    # debris, not a broken flow; the start and the question node always stay.
    touched = {e.get(k) for e in kept_edges for k in ("source", "target")}
    cut |= {
        n.get("id")
        for n in nodes
        if n.get("id") not in touched
        and n.get("id") not in questions
        and n.get("type") != "start"
    }
    if len(kept_edges) == len(edges) and not cut:
        return parsed
    logger.info("pruned %d edge(s) and %d node(s) after the question node",
                len(edges) - len(kept_edges), len(cut))
    return {
        **parsed,
        "nodes": [n for n in nodes if n.get("id") not in cut],
        "edges": [e for e in kept_edges if e.get("source") not in cut and e.get("target") not in cut],
    }


# A sentence ends at ".", "?" or "!" followed by space and a capital (or a quote), so a
# file name such as ".claude/rules/" inside a sentence does not end it.
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'`(])")


def the_ask(doc: dict) -> str:
    """The question's own ask: the last sentence of the stem that ends with "?"
    ("" when it has none)."""
    sentences = _SENTENCE_END.split(" ".join(source_parts(doc)[1].split()))
    asks = [s for s in sentences if s.endswith("?")]
    return asks[-1] if asks else ""


def with_question_label(flow: DecisionFlow, doc: dict) -> DecisionFlow:
    """The flow with its question node labelled with the question's own ask, word for
    word. The model's label for it tends to be a hypothesis ("Is the failure caused by
    the max_tokens stop?") that points at an answer; the original words cannot."""
    ask = the_ask(doc)
    if ask:
        for node in flow.nodes:
            if node.is_question:
                node.label = ask
    return flow


async def flow_item(
    ica: IcaClient,
    doc: dict,
    decomposition: QuestionDecomposition,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[DecisionFlow, IcaMeta]:
    """The pre-answer flow of one question. The model reads the scenario and the stem
    and the decomposition's `scenario_view`: never the key, the answers, or what only
    they mention."""
    user = (
        f"{_question_block(doc, with_key=False)}\n\n"
        f"DECOMPOSITION:\n{_compact(scenario_view(decomposition, doc))}"
    )
    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=_system(_FLOW_SYSTEM, DecisionFlow),
        what="flow",
        validate=lambda parsed: flow_errors(doc, decomposition, parsed),
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        schema=DecisionFlow,
        label=doc.get("_id"),
        transform=prune_after_question,
    )
    flow = with_question_label(DecisionFlow.model_validate(parsed), doc)
    warning = leak_warning(doc, flow)
    if warning:
        logger.info("flow of %s: %s", doc.get("_id"), warning)
    return flow, meta


async def resolve_item(
    ica: IcaClient,
    doc: dict,
    decomposition: QuestionDecomposition,
    flow: DecisionFlow,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[Resolution, IcaMeta]:
    """The resolution of one question: needs exactly one correct answer (422 otherwise,
    which a job does not retry)."""
    if len(correct_indices(doc)) != 1:
        raise HTTPException(status_code=422, detail="the resolution needs exactly one correct answer")
    questions = {n.id for n in flow.nodes if n.is_question}
    before = flow.model_copy(deep=True)
    before.nodes = [n for n in before.nodes if n.id not in questions]
    before.edges = [e for e in before.edges if e.target not in questions and e.source not in questions]
    chose_after = [e.source for e in flow.edges if e.target in questions]
    user = (
        f"{_question_block(doc, with_key=True)}\n\n"
        f"DECOMPOSITION:\n{_compact(decomposition)}\n\n"
        f"FLOW BEFORE ANSWERING (without the question node):\n{_compact(before)}\n\n"
        f"THE LEARNER CHOSE AN ANSWER AFTER: {', '.join(chose_after) or '(the start)'}"
    )
    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=_system(_RESOLVE_SYSTEM, Resolution),
        what="resolution",
        validate=lambda parsed: resolution_errors(doc, decomposition, parsed),
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        schema=Resolution,
        label=doc.get("_id"),
    )
    return Resolution.model_validate(parsed), meta


async def _record(store: AsyncCollection, doc: dict) -> tuple[dict, QuestionDecomposition]:
    record = await store.find_one({"question_id": str(doc["_id"])})
    decomposition = read_decomposition(record, doc)
    if decomposition is None:
        raise HTTPException(status_code=404, detail="question has no current decomposition")
    return record, decomposition


async def _conditional_set(store: AsyncCollection, record: dict, fields: dict) -> DecompositionRecord:
    # Conditional on the record not having been rewritten meanwhile: the flow references
    # the ids of the decomposition that was sent.
    stored = await store.find_one_and_update(
        {"question_id": record["question_id"], "updated_at": record["updated_at"]},
        {"$set": {**fields, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(
            status_code=409, detail="decomposition changed while the flow was written; retry"
        )
    return DecompositionRecord(**stored)


async def flow_and_store(
    ica: IcaClient,
    store: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> DecompositionRecord:
    """Write the pre-answer flow of a decomposed question; its resolution goes with
    the old flow. 404 when the question has no current decomposition."""
    record, decomposition = await _record(store, doc)
    flow, meta = await flow_item(ica, doc, decomposition, model_id, max_tokens)
    return await _conditional_set(
        store,
        record,
        {
            "flow": flow.model_dump(),
            "flow_version": FLOW_VERSION,
            "flow_ica": meta.model_dump(),
            "resolution": None,
            "resolution_version": None,
            "resolution_ica": None,
        },
    )


async def resolve_and_store(
    ica: IcaClient,
    store: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> DecompositionRecord:
    """Write the resolution of a question that has a current flow (404 otherwise)."""
    record, decomposition = await _record(store, doc)
    if not flow_is_current(record):
        raise HTTPException(status_code=404, detail="question has no current flow")
    flow = DecisionFlow.model_validate(record["flow"])
    resolution, meta = await resolve_item(ica, doc, decomposition, flow, model_id, max_tokens)
    return await _conditional_set(
        store,
        record,
        {
            "resolution": resolution.model_dump(),
            "resolution_version": RESOLUTION_VERSION,
            "resolution_ica": meta.model_dump(),
        },
    )


def mermaid(flow: DecisionFlow, answers: list[str] | None = None) -> str:
    """The flow as a Mermaid `flowchart TD`, for reading it in any Mermaid viewer.
    With `answers` the question node gets one dotted edge per answer."""
    shapes = {
        "start": ("([", "])"),
        "end": ("([", "])"),
        "decision": ("{", "}"),
        "input": ("[/", "/]"),
        "output": ("[/", "/]"),
        "human_review": ("[[", "]]"),
        "process": ("[", "]"),
    }

    def text(label: str) -> str:
        return '"' + label.replace('"', "'") + '"'

    lines = ["flowchart TD"]
    for node in flow.nodes:
        left, right = shapes[node.type]
        lines.append(f"  {node.id}{left}{text(node.label)}{right}")
    for edge in flow.edges:
        arrow = "==>" if edge.added else "-->"
        label = edge.label or (edge.type if edge.type != "next" else "")
        lines.append(f"  {edge.source} {arrow}{'|' + text(label) + '|' if label else ''} {edge.target}")
    for node in flow.nodes:
        if node.is_question and answers:
            for i, answer in enumerate(answers):
                short = " ".join(answer.split()[:8]) + ("..." if len(answer.split()) > 8 else "")
                lines.append(f"  {node.id} -.-> opt{i}[{text(chr(65 + i) + '. ' + short)}]")
    added = [n.id for n in flow.nodes if n.added]
    if added:
        lines.append("  classDef added stroke-width:3px")
        lines.append(f"  class {','.join(added)} added")
    return "\n".join(lines)
