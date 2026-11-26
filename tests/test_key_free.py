"""No answer key in pre-answer content: every step whose output a player sees before
answering (the decomposition, the icons, the flow, the beat, the rephrase) and the check that
answers without the key build their messages from `KeyFreeQuestion`, which has no field
for the key, the feedback or the explanation. Sentinel strings in those fields must
never reach such a prompt; the steps that may see the key (objectives, story,
resolution) do get them."""

from dataclasses import fields

from test_jobs import decomposition_reply, flow_reply

from claude_exam.api import beat, check, decompose, flow, icons, objectives, rephrase, story
from claude_exam.api.question_view import KeyFreeQuestion, key_free
from claude_exam.api.schemas import DecisionFlow, PromptEntities, QuestionDecomposition

FEEDBACK = "SENTINEL-FEEDBACK"
EXPLANATION = "SENTINEL-EXPLANATION"
DOC = {
    "_id": "6ae4891ec11925a6a80c076b",
    "section": "Domain 3: Claude Code Configuration & Workflows",
    "correct_response": ["b"],
    "prompt": {
        "question": "SCENARIO: You are building a customer support resolution agent. "
        "QUESTION: Which mode reads files but never edits them?",
        "answers": ["acceptEdits mode", "plan mode"],
        "feedbacks": [f"{FEEDBACK} A", f"{FEEDBACK} B"],
        "explanation": EXPLANATION,
    },
}
PLACE = {"scenario": "support", "stage": "build", "beat": "The agent edits too much."}


def leaks(*messages: str) -> list[str]:
    text = "\n".join(messages)
    return [s for s in (FEEDBACK, EXPLANATION, "[CORRECT]", "CORRECT ANSWER") if s in text]


def test_the_view_has_no_field_for_the_key():
    names = {f.name for f in fields(KeyFreeQuestion)}
    assert not names & {"correct_response", "correct", "feedbacks", "explanation", "prompt"}
    q = key_free(DOC)
    assert q.stem == "Which mode reads files but never edits them?" and q.shared
    assert q.answers == ("acceptEdits mode", "plan mode")


def test_no_pre_answer_prompt_carries_the_key():
    q = key_free(DOC)
    d = QuestionDecomposition.model_validate(decomposition_reply())
    rephrased = {"prompt": {"question": "Which mode only reads?", "answers": ["acceptEdits mode", "plan mode"]}}
    assert leaks(*decompose.messages(q)) == []
    assert leaks(*flow.flow_messages(q, d)) == []
    assert leaks(*icons.messages(q, PromptEntities())) == []
    assert leaks(*rephrase.messages(q, PLACE)) == []
    assert leaks(*beat.messages(q, PLACE)) == []
    assert leaks(*check.messages(q, rephrased, PLACE)) == []


def test_the_steps_shown_after_answering_do_read_the_key():
    d = QuestionDecomposition.model_validate(decomposition_reply())
    drawn = DecisionFlow.model_validate(flow_reply())
    assert EXPLANATION in "\n".join(objectives.messages(DOC))
    assert "CORRECT ANSWER: B" in "\n".join(story.messages({**DOC, "_id": "x"}))
    assert "[CORRECT]" in "\n".join(flow.resolve_messages(DOC, d, drawn, None))


def test_the_beat_is_not_written_by_the_step_that_reads_the_key():
    # The story step (with the key) writes only the lines after the answer.
    system, _user = story.messages({**DOC, "_id": "x"})
    assert '"beat"' not in system and "BEFORE the question" not in system
    # The beat step reads the stage and chapter, not an earlier beat.
    assert PLACE["beat"] not in "\n".join(beat.messages(key_free(DOC), PLACE))
