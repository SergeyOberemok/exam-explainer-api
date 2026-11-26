"""The decomposition, flow and resolution steps: their gates, the semantic parts of the
game text, the entities derived from a decomposition, structured outputs with their
fallback, and the jobs and routes over them."""

import copy

import pytest
from fastapi import HTTPException
from test_jobs import (
    FakeMongo,
    ScriptedIca,
    _edge,
    _fact,
    _node,
    client_for,
    decomposition_reply,
    flow_reply,
    question,
    reply,
    resolution_reply,
    run_job,
)

from claude_exam.api import batch, decompose, ica_json
from claude_exam.api.decompose import MAX_PARTS, decomposition_errors, normalize, parts_from, scenario_view
from claude_exam.api.entities import derive_entities
from claude_exam.api.flow import (
    flow_errors,
    leak_warning,
    mermaid,
    prune_after_question,
    the_ask,
    with_question_label,
    resolution_errors,
)
from claude_exam.api.ica_json import ask_json, strict_schema
from claude_exam.api.schemas import (
    DecisionFlow,
    QuestionDecomposition,
)
from claude_exam.api.question_view import key_free

pytestmark = pytest.mark.anyio
MODEL = "claude-haiku-5-5"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(ica_json, "_NO_SCHEMA", set())


def review_question(**kw):
    """A question with a scenario, a number and four answers (correct: B)."""
    return question(
        correct_response=["b"],
        prompt={
            "question": (
                "Your review agent posts findings on pull requests. In 12% of reviews "
                "the team guidelines are silent, yet the agent posts a definitive "
                "finding that developers contest. Which design routes such cases?"
            ),
            "answers": [
                "Have the agent report a confidence score.",
                "Route findings the guidelines do not cover to a human reviewer.",
                "Add more examples to the `review` prompt.",
                "Run the review twice.",
            ],
            "feedbacks": ["Incorrect. a", "Correct. b", "Incorrect. c", "Incorrect. d"],
            "explanation": "Missing guidance needs human judgment.",
        },
        **kw,
    )


def explicit(**kw):
    d = decomposition_reply(4)
    d["entities"][0].update(
        name="Review Agent", source_status="explicit", evidence="your review agent posts findings"
    )
    d.update(kw)
    return d


# --- the decomposition gate -----------------------------------------------------------


def test_a_good_decomposition_passes_and_explicit_items_need_verbatim_evidence():
    q = review_question()
    assert decomposition_errors(q, explicit()) == []

    # Quotes, backticks, case and spacing do not count; the words do.
    assert normalize('The  "Review" `Agent`.') == "the review agent"
    paraphrase = explicit()
    paraphrase["entities"][0]["evidence"] = "the agent that reviews code"
    assert any("word for word" in e for e in decomposition_errors(q, paraphrase))
    empty = explicit()
    empty["problem"]["symptoms"][0].update(source_status="explicit", evidence="")
    assert any("symptoms[0]" in e for e in decomposition_errors(q, empty))


def test_decomposition_ids_references_options_and_numbers_are_checked():
    q = review_question()
    d = explicit()
    d["entities"][1]["id"] = "e1"
    d["events"] = [
        {"id": "ev1", "name": "Post finding", "actor_id": "e9", "target_id": None, "input_ids": [],
         "output_ids": [], "sequence": 1, "source_status": "inferred", "evidence": ""}
    ]
    d["answer_options"] = d["answer_options"][:3]
    d["question_summary"] = "In 15% of reviews the agent is wrong."
    errors = " | ".join(decomposition_errors(q, d))
    assert "'e1' is used twice" in errors
    assert "event ev1 refers to ids that do not exist: e9" in errors
    assert "one item per answer" in errors
    assert "15" in errors and "12" not in errors.split("numbers")[-1]

    # A reply of the wrong shape is reported, not raised.
    assert decomposition_errors(q, {"entities": "none"})


# --- the flow gates -------------------------------------------------------------------


def decomposition(q):
    return QuestionDecomposition.model_validate(explicit())


def test_a_pre_answer_flow_ends_at_the_question_node_and_adds_nothing():
    q = review_question()
    d = decomposition(q)
    assert flow_errors(q, d, flow_reply()) == []

    opened = flow_reply()
    opened["nodes"].append(_node("n4", "end", "Finding posted"))
    opened["edges"].append(_edge("n3", "n4"))
    assert any("must have no outgoing edges" in e for e in flow_errors(q, d, opened))

    hinted = flow_reply()
    hinted["nodes"][1]["added"] = True
    assert any("adds nothing" in e for e in flow_errors(q, d, hinted))

    none = flow_reply()
    none["nodes"][2]["is_question"] = False
    assert any("is_question true" in e for e in flow_errors(q, d, none))


def test_what_comes_after_the_question_node_is_cut_before_the_gate():
    q = review_question()
    d = decomposition(q)
    opened = flow_reply()
    opened["nodes"] += [_node("n4", "process", "Route to a human"), _node("n5", "end", "Reviewed")]
    opened["edges"] += [_edge("n3", "n4"), _edge("n4", "n5")]
    pruned = prune_after_question(opened)
    assert [n["id"] for n in pruned["nodes"]] == ["n1", "n2", "n3"]
    assert flow_errors(q, d, pruned) == []
    # A node without any edge is debris and goes; an unreachable piece that has edges is
    # kept, so the gate still sees the broken flow.
    debris = flow_reply()
    debris["nodes"].append(_node("n9", "end", "Orphan"))
    assert [n["id"] for n in prune_after_question(debris)["nodes"]] == ["n1", "n2", "n3"]
    broken = flow_reply()
    broken["nodes"] += [_node("n8", "process", "Lost"), _node("n9", "end", "Orphan")]
    broken["edges"].append(_edge("n8", "n9"))
    assert prune_after_question(broken) is broken
    assert any("not reachable" in e for e in flow_errors(q, d, broken))


def test_the_flow_shows_no_entity_only_an_answer_names():
    q = review_question()
    d = explicit()
    d["entities"][1].update(
        name="Human Reviewer", type="actor", source_status="explicit", evidence="a human reviewer"
    )
    decomposition_ = QuestionDecomposition.model_validate(d)
    # The model never gets it: the view keeps the scenario's entities only.
    view = scenario_view(decomposition_, q)
    assert [e.name for e in view.entities] == ["Review Agent"] and view.answer_options == []

    leaky = flow_reply()
    leaky["nodes"][1].update(label="Human reviewer judges the finding", entity_ids=["e2"])
    errors = " | ".join(flow_errors(q, decomposition_, leaky))
    assert "names 'Human Reviewer', which only an answer option mentions" in errors
    assert "entities that do not exist: e2" in errors

    asking = flow_reply()
    asking["nodes"][1].update(type="output", label="Do the guidelines cover it?")
    assert any("only a decision node asks" in e for e in flow_errors(q, decomposition_, asking))


def test_scenario_facts_may_not_quote_an_answer_option():
    q = review_question()
    d = explicit()
    d["conditions"] = [
        {"id": "c1", "description": "Uncovered cases go to a human", "condition_type": "policy",
         "evaluation": "true_false", "source_status": "explicit",
         "evidence": "Route findings the guidelines do not cover to a human reviewer"}
    ]
    d["entities"][1].update(type="actor", name="Human Reviewer", source_status="explicit", evidence="a human reviewer")
    errors = decomposition_errors(q, d)
    assert len(errors) == 1 and "condition c1 quotes an answer option" in errors[0]


def test_the_question_node_says_what_the_question_asks():
    q = review_question()
    assert the_ask(q) == "Which design routes such cases?"
    named = question(prompt={**question()["prompt"], "question": "It fails. Which line of .claude/rules/a.md applies? Pick one."})
    assert the_ask(named) == "Which line of .claude/rules/a.md applies?"
    hinted = flow_reply()
    hinted["nodes"][2]["label"] = "Is it caused by the guidelines?"
    flow = with_question_label(DecisionFlow.model_validate(hinted), q)
    assert flow.nodes[2].label == "Which design routes such cases?"


def test_flow_graph_rules():
    q = review_question()
    d = decomposition(q)
    bad = flow_reply()
    bad["nodes"] += [
        _node("n4", "decision", "Guidelines cover it?"),
        _node("n5", "process", "Post the finding"),
        _node("n6", "output", "Orphan"),
        _node("n7", "process", "Check 40 rules", entity_ids=["e7"], condition_id="c9"),
    ]
    bad["edges"] += [_edge("n2", "n4"), _edge("n4", "n5", type="custom"), _edge("n2", "n8")]
    errors = " | ".join(flow_errors(q, d, bad))
    assert "decision n4 needs two or more outgoing branches" in errors
    assert "branch n4->n5 needs a label" in errors
    assert "n5 is a dead end" in errors
    assert "not reachable from the start: n6, n7" in errors
    assert "edge n2->n8 uses nodes that do not exist: n8" in errors
    assert "entities that do not exist: e7" in errors and "condition that does not exist: c9" in errors
    assert "40" in errors

    loop = flow_reply()
    loop["nodes"].append(_node("n4", "process", "Retry"))
    loop["edges"] += [_edge("n2", "n4"), _edge("n4", "n2")]
    assert flow_errors(q, d, loop) == []  # a loop with an exit is fine

    two = flow_reply()
    two["nodes"].append(_node("n0", "start", "Another start"))
    two["edges"].append(_edge("n0", "n2"))
    assert any("exactly one start" in e for e in flow_errors(q, d, two))


def test_the_resolution_replaces_the_question_node_and_matches_the_key():
    q = review_question()
    d = decomposition(q)
    assert resolution_errors(q, d, resolution_reply(4, [1])) == []

    wrong = resolution_reply(4, [0])
    assert any("option(s) A, B" in e for e in resolution_errors(q, d, wrong))

    unresolved = resolution_reply(4, [1])
    unresolved["flow"] = flow_reply()
    errors = resolution_errors(q, d, unresolved)
    assert any("no node may have is_question" in e for e in errors)
    assert any("added true" in e for e in errors)


def test_leak_warning_and_mermaid():
    q = review_question()
    flow = DecisionFlow.model_validate(flow_reply())
    assert leak_warning(q, flow) == ""
    leaky = flow_reply()
    leaky["nodes"][1]["label"] = "Route findings guidelines cover to human reviewer"
    assert "correct answer B" in leak_warning(q, DecisionFlow.model_validate(leaky))

    text = mermaid(flow, ["Confidence score", "Route to a human"])
    assert text.startswith("flowchart TD") and 'n3{"Which mode?"}' in text
    assert 'n3 -.-> opt1["B. Route to a human"]' in text
    resolved = mermaid(DecisionFlow.model_validate(resolution_reply()["flow"]))
    assert "n2 ==> n4" in resolved and "class n4,n5 added" in resolved


# --- structured outputs ---------------------------------------------------------------


def test_strict_schema_closes_every_object_and_requires_every_field():
    schema = strict_schema(QuestionDecomposition)
    entity = schema["$defs"]["DecompEntity"]
    assert entity["additionalProperties"] is False
    assert set(entity["required"]) == set(entity["properties"])
    assert "default" not in str(schema)


async def test_ask_json_sends_the_schema_and_falls_back_when_it_is_refused():
    ica = ScriptedIca()
    await ask_json(ica, MODEL, "x", system="question decomposition engine", what="d", schema=QuestionDecomposition)
    assert ica.formats == ["json_schema"]

    class Refusing(ScriptedIca):
        async def chat_model(self, *args, response_format=None, **kwargs):
            if response_format:
                self.formats.append("refused")
                raise HTTPException(status_code=502, detail="ICA returned 400: unsupported response_format")
            return await super().chat_model(*args, **kwargs)

    refusing = Refusing()
    for _ in range(2):
        parsed, _meta = await ask_json(
            refusing, MODEL, "x", system="question decomposition engine", what="d", schema=QuestionDecomposition
        )
    # Refused once, then remembered: the second call goes straight to json_object.
    assert refusing.formats == ["refused", "json_object", "json_object"]
    assert parsed["question_summary"]


# --- semantic parts and derived entities ------------------------------------------------


def test_parts_are_built_in_code_from_what_may_be_shown_before_answering():
    q = question()
    d = decomposition_reply()
    d["problem"]["symptoms"] = [_fact("Reviews come too late.")]
    parts = parts_from(QuestionDecomposition.model_validate(d), key_free(q))
    assert [p.role for p in parts] == ["context", "current_state", "problem", "ask"]
    assert parts[0].text == "A team works with an agent."
    # The ask is the question's own, word for word, and it comes last.
    assert parts[-1].text == "Which mode?"


def test_parts_name_no_answer_only_entity_and_keep_the_ask_when_cut():
    q = question()
    d = decomposition_reply()
    d["entities"].append(
        {"id": "e3", "name": "plan mode", "type": "object", "aliases": [], "description": "", "source_status": "inferred", "evidence": ""}
    )
    d["problem"]["symptoms"] = [
        *(_fact(f"Symptom {i} with plan mode.") for i in range(3)),
        *(_fact(f"Symptom {i} as it is.") for i in range(9)),
    ]
    parts = parts_from(QuestionDecomposition.model_validate(d), key_free(q))
    assert len(parts) == MAX_PARTS and parts[-1].role == "ask"
    # "plan mode" is what an answer proposes: a part naming it gives that answer away,
    # so it is dropped (not only its entity id).
    assert all("plan mode" not in p.text for p in parts)
    assert all("e3" not in p.entity_ids for p in parts)


def test_parts_drop_a_repeated_fact_and_keep_the_goal_the_ask_refers_to():
    q = question()
    d = decomposition_reply()
    d["context"]["current_state"] = [
        _fact("Developers contest the posted findings."),
        _fact("Most false positives involve uncovered patterns."),
        _fact("Findings are posted as definitive."),
    ]
    d["problem"]["symptoms"] = [
        # The first current state again, in other words: dropped.
        _fact("Developers contest findings as false positives."),
        _fact("Reviews come too late."),
    ]
    d["context"]["desired_state"] = [_fact("Uncovered cases go to a human reviewer.")]
    parts = parts_from(QuestionDecomposition.model_validate(d), key_free(q))
    texts = [p.text for p in parts]
    assert "Developers contest findings as false positives." not in texts
    assert len(parts) == MAX_PARTS
    # Context, three current states and a symptom fill the five places; the goal takes
    # the symptom's, so the ask still has something to refer to.
    assert [p.role for p in parts][-2:] == ["goal", "ask"]
    assert "Reviews come too late." not in texts


def test_derived_entities_keep_refs_drop_concepts_and_follow_the_answer_options():
    q = review_question()
    d = explicit()
    d["entities"][0]["aliases"] = ["agent"]
    d["entities"].append(
        {"id": "e3", "name": "Human Reviewer", "type": "actor", "aliases": [], "description": "Judges",
         "source_status": "inferred", "evidence": ""}
    )
    d["answer_options"][1]["entity_ids"] = ["e3"]
    entities = derive_entities(QuestionDecomposition.model_validate(d), key_free(q))
    assert [(e.name, e.ref) for e in entities.question] == [("Review Agent", "e1")]
    assert [e.name for e in entities.answers[1].entities] == ["Human Reviewer"]
    # The concept ("Ambiguity") is never drawn, and answers only get what they mention.
    assert all(e.name != "Ambiguity" for a in entities.answers for e in a.entities)
    assert entities.answers[3].entities == []


# --- jobs and routes ------------------------------------------------------------------


async def test_decompose_job_stores_flow_and_resolution_and_resumes():
    q = review_question()
    mongo, ica = FakeMongo([q]), ScriptedIca(flow=[HTTPException(status_code=400, detail="x")])
    job = await run_job(mongo, ica, [q["_id"]], "decompose")
    assert ica.calls == ["decompose", "icons", "flow"] and job.stages_failed == {"flow": 1}
    db = mongo.get_default_database()
    record = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
    assert record["semantic"]["question_intent"]["question_task"] and record["flow"] is None

    ica.calls.clear()
    again = await run_job(mongo, ica, [q["_id"]], "decompose", only_missing=True)
    assert ica.calls == ["flow", "resolve"] and again.succeeded == 1
    record = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
    assert record["flow"]["nodes"][-1]["is_question"] is True
    assert [o["verdict"] for o in record["resolution"]["options"]] == [
        "incorrect", "correct", "incorrect", "incorrect"
    ]
    # Every new step asked for a strict json_schema reply.
    assert set(ica.formats[-2:]) == {"json_schema"}

    # The flow model never saw the key; the resolution model did.
    seen = {}

    class Spy(ScriptedIca):
        async def chat_model(self, model_id, prompt, system=None, **kwargs):
            system = system or ""
            kind = "decompose" if "decomposition engine" in system else "flow" if "draw the flow" in system else "icons" if "icon catalog" in system else "resolve"
            seen.setdefault(kind, prompt)
            return await super().chat_model(model_id, prompt, system=system, **kwargs)

    await run_job(FakeMongo([q]), Spy(), [q["_id"]], "decompose")
    pre_answer = seen["decompose"] + seen["flow"] + seen["icons"]
    assert "[CORRECT]" not in pre_answer and "Correct. b" not in pre_answer
    assert "[CORRECT]" in seen["resolve"]


async def test_a_new_decomposition_replaces_its_parts_and_clears_what_was_built_on_it(monkeypatch):
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "full")
    db = mongo.get_default_database()
    record = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
    assert record["parts"] and record["icons_ica"] and record["flow"] and record["resolution"]

    # The decomposition prompt changed (nothing bumped): it is written again, with its
    # parts and entities, and its rewrite clears the icons, flow and resolution.
    monkeypatch.setattr(decompose, "_SYSTEM_MESSAGE", decompose._SYSTEM_MESSAGE + "\nNew rule.")
    other = decomposition_reply()
    other["context"]["description"] = "Another reading."
    ica.queues["decompose"].append(reply(other))
    ica.calls.clear()
    await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    # (The story step, which keeps failing for this question, is left out.)
    assert [c for c in ica.calls if c != "story"] == ["decompose", "icons", "flow", "resolve"]
    record = await db["question_decomposition"].find_one({"question_id": str(q["_id"])})
    assert record["parts"][0]["text"] == "Another reading."


def test_single_question_routes():
    q = review_question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client = client_for(mongo, ica)
    qid = str(q["_id"])
    assert client.get(f"/examination/{qid}/decomposition").status_code == 404
    assert client.post(f"/examination/{qid}/flow").status_code == 404
    assert client.post(f"/examination/{qid}/decompose").status_code == 200
    assert client.post(f"/examination/{qid}/resolve").status_code == 404
    assert client.post(f"/examination/{qid}/flow").status_code == 200
    resolved = client.post(f"/examination/{qid}/resolve")
    assert resolved.status_code == 200 and resolved.json()["resolution"]["rule"]
    assert client.get(f"/examination/{qid}/decomposition").json()["flow"]["nodes"]
    assert ica.calls == ["decompose", "flow", "resolve"]


# --- review fixes: leaks, transform before repair, outdated records ---------------------


def human_reviewer(d):
    """`d` with an entity only answer B names (the human reviewer)."""
    d["entities"][1].update(
        name="Human Reviewer", type="actor", source_status="explicit", evidence="a human reviewer"
    )
    return d


def test_scenario_facts_may_not_name_what_only_an_answer_proposes_even_when_inferred():
    q = review_question()
    d = human_reviewer(explicit())
    d["context"]["desired_state"] = [
        {"text": "A human reviewer judges uncovered cases", "source_status": "inferred", "evidence": ""}
    ]
    errors = decomposition_errors(q, d)
    assert len(errors) == 1 and "desired_state[0] names 'Human Reviewer'" in errors[0]
    # An entity no text names (the model's own word) is no answer's proposal.
    assert decomposition_errors(q, explicit()) == []


def test_the_view_drops_open_questions_that_name_an_answer_only_entity():
    q = review_question()
    d = human_reviewer(explicit())
    d["ambiguities_and_unknowns"] = ["Whether a human reviewer is available", "Who owns guidelines"]
    d["question_intent"]["success_criteria"] = ["Human reviewer sees uncovered cases"]
    view = scenario_view(QuestionDecomposition.model_validate(d), q)
    assert view.ambiguities_and_unknowns == ["Who owns guidelines"]
    assert view.question_intent.success_criteria == []


def test_the_flow_trigger_and_edge_labels_may_not_name_an_answer_only_entity():
    q = review_question()
    d = QuestionDecomposition.model_validate(human_reviewer(explicit()))
    leaky = flow_reply()
    leaky["trigger"] = "Human reviewer gets a case"
    leaky["edges"][0]["label"] = "to the human reviewer"
    errors = " | ".join(flow_errors(q, d, leaky))
    assert "the trigger names 'Human Reviewer'" in errors and "edge n1->n2 names" in errors


async def test_the_repair_note_shows_the_reply_as_the_code_repaired_it():
    import json

    replies = [{"keep": 1, "cut": 2}, {"keep": 2, "cut": 3}]
    prompts = []

    class Ica:
        async def chat_model(self, model, prompt, **kw):
            prompts.append(prompt)
            body = json.dumps(replies[len(prompts) - 1])
            return {"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}

    parsed, _ = await ica_json.ask_validated(
        Ica(),
        MODEL,
        "go",
        system="s",
        what="x",
        validate=lambda p: [] if p["keep"] == 2 else ["keep must be 2"],
        transform=lambda p: {k: v for k, v in p.items() if k != "cut"},
    )
    assert parsed == {"keep": 2}
    assert '"keep": 1' in prompts[1] and '"cut"' not in prompts[1]


async def test_resolve_is_outdated_with_its_flow_and_an_old_record_shape_still_reads(monkeypatch):
    q = review_question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "decompose")
    store = mongo.get_default_database()["question_decomposition"]
    record = store.docs[0]
    assert record["parts"] and record["input_key"]
    stores = batch.Stores(None, None, None, store)
    assert await batch.is_current("resolve", stores, q)
    # A changed flow prompt outdates the flow, and so the resolution that continues it.
    from claude_exam.api import flow

    monkeypatch.setattr(flow, "_FLOW_SYSTEM", flow._FLOW_SYSTEM + "\nNew rule.")
    assert not await batch.is_current("flow", stores, q)
    assert not await batch.is_current("resolve", stores, q)

    # A flow of an older shape (edges once said from/to) reads as none, not as a 500.
    record["flow"] = {"edges": [{"from": "n1", "to": "n2"}]}
    client = client_for(mongo, ica)
    got = client.get(f"/examination/{q['_id']}/decomposition")
    assert got.status_code == 200 and got.json()["flow"] is None
    record["semantic"] = {"entities": "old"}
    assert client.get(f"/examination/{q['_id']}/decomposition").status_code == 404


def test_single_question_routes_answer_404_for_an_unknown_question():
    client = client_for(FakeMongo([]), ScriptedIca())
    for step in ("decompose", "flow", "resolve"):
        assert client.post(f"/examination/{'0' * 24}/{step}").status_code == 404


# --- what the resolution replaces, the code flow's events ------------------------------


def _branching_flow():
    """A pre-answer flow whose decision posts a finding anyway on one branch (the
    problem) and reaches the question on the other."""
    return DecisionFlow.model_validate({
        "visualization": {"primary": "decision_flow", "secondary": [], "reason": "Routing."},
        "trigger": "Work starts",
        "nodes": [
            _node("n1", "start", "Work starts"),
            _node("n2", "decision", "Covered by the rules?"),
            _node("n3", "process", "Posted anyway"),
            _node("n4", "end", "Developers contest it"),
            _node("n5", "decision", "Which mode?", is_question=True),
        ],
        "edges": [
            _edge("n1", "n2"),
            _edge("n2", "n3", type="no", label="Not covered"),
            _edge("n2", "n5", type="yes", label="Covered"),
            _edge("n3", "n4"),
        ],
    })


def _resolved(keep_old: bool):
    edges = [
        _edge("n1", "n2"),
        _edge("n2", "n6", type="yes", label="Covered", added=True),
        _edge("n2", "n7", type="no", label="Not covered: a human decides", added=True),
        _edge("n6", "n8", added=True),
        _edge("n7", "n8", added=True),
    ]
    if keep_old:
        edges += [_edge("n2", "n3", type="no", label="Not covered"), _edge("n3", "n4")]
    reply_ = resolution_reply()
    reply_["flow"] = {
        "visualization": {"primary": "decision_flow", "secondary": [], "reason": "Routing."},
        "trigger": "Work starts",
        "nodes": [
            _node("n1", "start", "Work starts"),
            _node("n2", "decision", "Covered by the rules?"),
            _node("n3", "process", "Posted anyway"),
            _node("n4", "end", "Developers contest it"),
            _node("n6", "process", "Posted", added=True),
            _node("n7", "human_review", "A human decides", added=True),
            _node("n8", "end", "Clear findings posted", added=True),
        ],
        "edges": edges,
    }
    return reply_


def test_a_resolution_may_not_keep_the_old_branch_next_to_its_fix():
    from claude_exam.api.flow import replaced_errors

    before = _branching_flow()
    errors = replaced_errors(before, DecisionFlow.model_validate(_resolved(keep_old=True)["flow"]))
    assert len(errors) == 1 and "decision n2" in errors[0] and "n2->n3" in errors[0]


def test_the_path_the_fix_replaced_is_pruned_in_code():
    from claude_exam.api.flow import prune_replaced, replaced_errors

    pruned = prune_replaced(_resolved(keep_old=False))
    ids = [n["id"] for n in pruned["flow"]["nodes"]]
    # The old branch is gone, so "posted anyway" and its end are no longer reached.
    assert "n3" not in ids and "n4" not in ids
    assert ids == ["n1", "n2", "n6", "n7", "n8"]
    flow = DecisionFlow.model_validate(pruned["flow"])
    assert replaced_errors(_branching_flow(), flow) == []
    q = question()
    assert resolution_errors(q, QuestionDecomposition.model_validate(decomposition_reply()), pruned, before=_branching_flow()) == []


def test_a_resolution_keeps_the_start_and_where_the_learner_chose():
    from claude_exam.api.flow import replaced_errors

    flow = DecisionFlow.model_validate(_resolved(keep_old=False)["flow"])
    flow.nodes = [n for n in flow.nodes if n.id != "n2"]
    flow.edges = [e for e in flow.edges if "n2" not in (e.source, e.target)]
    assert any("n2" in e and "chose after" in e for e in replaced_errors(_branching_flow(), flow))
