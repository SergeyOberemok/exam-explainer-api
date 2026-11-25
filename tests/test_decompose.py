"""The decomposition, flow and resolution steps: their gates, the semantic parts of the
game text, the entities derived from a decomposition, structured outputs with their
fallback, and the jobs and routes over them."""

import copy

import pytest
from fastapi import HTTPException
from test_jobs import (
    SIMPLE,
    FakeMongo,
    ScriptedIca,
    _edge,
    _node,
    client_for,
    decomposition_reply,
    flow_reply,
    question,
    reply,
    resolution_reply,
    run_job,
    stored,
)

from claude_exam.api import batch, ica_json
from claude_exam.api.decompose import PROMPT_VERSION as DECOMPOSE_VERSION
from claude_exam.api.decompose import decomposition_errors, normalize, scenario_view
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
    SimplifiedPrompt,
)
from claude_exam.api.simplify import joined, parts_errors, simplify_item

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
    assert "default" not in str(strict_schema(SimplifiedPrompt))


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


def parts_reply(*parts):
    base = {k: v for k, v in SIMPLE.items() if k != "question"}
    return {**base, "parts": [{"role": r, "text": t, "entity_ids": ids} for r, t, ids in parts]}


def test_parts_need_one_final_ask_and_known_entities():
    d = QuestionDecomposition.model_validate(decomposition_reply())
    good = parts_reply(("problem", "Edits land unreviewed.", ["e1"]), ("ask", "Which mode?", []))
    assert parts_errors(good, d) == []
    assert joined(good)["question"] == "Edits land unreviewed. Which mode?"
    late = parts_reply(("ask", "Which mode?", []), ("goal", "Review first.", ["e7"]))
    errors = " | ".join(parts_errors(late, d))
    assert "last one" in errors and "not listed: e7" in errors
    assert parts_errors({"parts": []}, d)


async def test_simplify_with_a_decomposition_writes_parts_and_repairs_bad_ones():
    q = question()
    d = QuestionDecomposition.model_validate(decomposition_reply())
    text = "Which mode, plan mode or acceptEdits mode, reads but never edits?"
    no_ask = reply(parts_reply(("problem", text, [])))
    good = reply(parts_reply(("context", "You review agent work.", ["e1"]), ("ask", text, [])))
    ica = ScriptedIca(simplify=[no_ask, good])
    prompt, _meta = await simplify_item(ica, q, MODEL, decomposition=d)
    assert ica.calls == ["simplify", "simplify"]
    assert [p.role for p in prompt.parts] == ["context", "ask"]
    assert prompt.question == f"You review agent work. {text}"


def test_derived_entities_keep_refs_drop_concepts_and_follow_the_answer_options():
    q = review_question()
    d = explicit()
    d["entities"][0]["aliases"] = ["agent"]
    d["entities"].append(
        {"id": "e3", "name": "Human Reviewer", "type": "actor", "aliases": [], "description": "Judges",
         "source_status": "inferred", "evidence": ""}
    )
    d["answer_options"][1]["entity_ids"] = ["e3"]
    prompt = SimplifiedPrompt(
        question="Your review agent posts findings. Which design routes such cases?",
        answers=[a for a in q["prompt"]["answers"]],
        feedbacks=["a", "b", "c", "d"],
        explanation="x",
    )
    entities = derive_entities(QuestionDecomposition.model_validate(d), q, prompt)
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
    assert ica.calls == ["decompose", "flow"] and job.stages_failed == {"flow": 1}
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
            seen.setdefault("decompose" if "decomposition engine" in (system or "") else "flow" if "draw the flow" in (system or "") else "resolve", prompt)
            return await super().chat_model(model_id, prompt, system=system, **kwargs)

    await run_job(FakeMongo([q]), Spy(), [q["_id"]], "decompose")
    assert "[CORRECT]" not in seen["decompose"] + seen["flow"] and "Correct. b" not in seen["flow"]
    assert "[CORRECT]" in seen["resolve"]


async def test_a_new_decomposition_redoes_the_parts_and_clears_the_flow():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "full")
    record = await stored(mongo, q)
    assert record["decomposition_version"].startswith(f"{DECOMPOSE_VERSION}:") and record["prompt"]["parts"]
    assert record["entities_source"] == "decomposition"

    db = mongo.get_default_database()
    await db["question_decomposition"].update_one(
        {"question_id": str(q["_id"])}, {"$set": {"semantic_version": "old"}}
    )
    # The same decomposition again changes nothing built on it...
    ica.calls.clear()
    await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    assert [c for c in ica.calls if c != "story"] == ["decompose", "flow", "resolve"]

    # ...another one redoes the parts.
    await db["question_decomposition"].update_one(
        {"question_id": str(q["_id"])}, {"$set": {"semantic_version": "old"}}
    )
    other = decomposition_reply()
    other["question_summary"] = "Another reading."
    ica.queues["decompose"].append(reply(other))
    ica.calls.clear()
    await run_job(mongo, ica, [q["_id"]], "full", only_missing=True)
    record = await stored(mongo, q)
    # Same text back, so the check is kept; the entities point into the decomposition,
    # so they are derived again (no call) and get new icons. (The story step, which
    # keeps failing for this question, is left out.)
    assert [c for c in ica.calls if c != "story"] == [
        "decompose", "simplify", "icons", "flow", "resolve"
    ]
    assert record["check"] is not None and record["entities_source"] == "decomposition"


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


async def test_resolve_is_outdated_with_its_flow_and_an_old_record_shape_still_reads():
    q = review_question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    await run_job(mongo, ica, [q["_id"]], "decompose")
    store = mongo.get_default_database()["question_decomposition"]
    record = store.docs[0]
    assert record["fingerprint"].startswith(f"{DECOMPOSE_VERSION}:")
    stores = batch.Stores(None, None, None, None, store)
    assert await batch.is_current("resolve", stores, q)
    record["flow_version"] = "an older flow"
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
