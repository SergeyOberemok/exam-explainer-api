import json
import re
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.ica import IcaClient
from claude_exam.api.schemas import (
    IcaMeta,
    IcaModelId,
    SimplifiedPrompt,
    SimplifiedQuestion,
)
from claude_exam.html_clean import clean_plain, clean_question

# System message injected directly into /chat-models/chat/completions.
# Combines the full Exam Explainer Agent persona with a JSON output requirement
# appended at the end — so a single model call produces structured output.
_SYSTEM_MESSAGE = """\
You are a patient exam coach for the Claude Certified Architect / Developer \
certification. You have deep familiarity with Claude Code, the Claude Agent SDK, \
the Claude API, MCP, prompt engineering, tool use, context management, evals and \
agent design. Your job is for the learner to understand and remember, not just to \
see the key.

When given a multiple-choice exam question, do the following in your head:
1. Retell the question as a concrete story the learner can hold in their head.
2. Identify the core problem and goal.
3. For every answer option, determine: the mechanism it relies on, the expected \
outcome, what changes, and its main limitation.
4. Identify the correct answer and the single deciding fact that separates it from \
the rest.
5. Derive a one-line takeaway rule for similar questions.

Use the supplied explanation, per-option feedbacks, and correct_response as the \
source of truth. Do not invent technical facts.

Plain language rules: short sentences, everyday words, define jargon on first use, \
concrete over abstract. Active voice. Keep technical terms only when they are the \
concept being tested.

Exact terms: keep every product, mode, tool, setting, parameter and command name \
verbatim (for example `acceptEdits`, `dontAsk`, plan mode, `git diff`, MCP) — never \
replace one with a paraphrase. If a term needs explaining, put the plain-words \
explanation right next to the exact name, e.g. "`dontAsk` mode (silently denies \
anything not pre-approved)". Also keep every concrete detail that distinguishes one \
option from another (extra steps, conditions, qualifiers).

OUTPUT FORMAT — you MUST respond with ONLY a single valid JSON object and nothing \
else — no markdown fences, no prose before or after, no comments. \
Respond with exactly this schema:

{
  "question": "<simplified question text — plain text, no HTML. Keep the scenario \
context (what is being built, with which tools) and the actual ask; retell it in \
2-5 short sentences>",
  {{ANSWERS_SCHEMA}},
  {{FEEDBACKS_SCHEMA}},
  "explanation": "<2-3 sentences: first state the single deciding fact that makes \
the correct answer correct, then why the closest wrong option fails, then end with \
a one-line rule for similar questions>"
}

Rules:
- Preserve the number of answers and feedbacks exactly (same count as input).
- Keep the order: answers[i] and feedbacks[i] must rewrite the input option with \
the same letter (A is index 0, B is index 1, and so on). Never reorder or merge options.
- Do not change correct_response — you are only rewriting text.
- Output valid JSON. Do not wrap it in a code block or add any surrounding text.\
"""


def _build_system_message(n_answers: int) -> str:
    # Fill the {{ANSWERS_SCHEMA}} / {{FEEDBACKS_SCHEMA}} placeholders with the
    # correct number of entries so the model knows the exact expected array length.
    letters = [chr(65 + i) for i in range(n_answers)]
    answers_entries = ",\n    ".join(
        f'"<simplified answer {l} — plain words, one or two sentences>"' for l in letters
    )
    feedbacks_entries = ",\n    ".join(
        f'"<simplified feedback for {l} — why it is right or wrong, one or two '
        f'short sentences that keep the reasoning, not just the verdict>"'
        for l in letters
    )
    return _SYSTEM_MESSAGE.replace(
        "{{ANSWERS_SCHEMA}}", f'"answers": [\n    {answers_entries}\n  ]'
    ).replace(
        "{{FEEDBACKS_SCHEMA}}", f'"feedbacks": [\n    {feedbacks_entries}\n  ]'
    )


def _build_user_message(item: dict) -> str:
    """Render one practice-exam item as the user turn sent to the agent."""
    prompt = item.get("prompt") or {}
    question = clean_question(prompt.get("question", ""))
    answers = prompt.get("answers", [])
    feedbacks = prompt.get("feedbacks", [])
    correct = item.get("correct_response", [])
    explanation = clean_plain(prompt.get("explanation", ""))
    section = item.get("section", "")

    choices_block = "\n".join(
        f"  {chr(65 + i)}. {clean_plain(a)}" for i, a in enumerate(answers)
    )
    correct_labels = ", ".join(l.upper() for l in correct)
    feedbacks_block = "\n".join(
        f"  {chr(65 + i)}. {clean_plain(f)}" for i, f in enumerate(feedbacks)
    )

    return (
        f"Section: {section}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER CHOICES:\n{choices_block}\n\n"
        f"CORRECT ANSWER: {correct_labels}\n\n"
        f"PER-CHOICE FEEDBACK:\n{feedbacks_block}\n\n"
        f"OFFICIAL EXPLANATION:\n{explanation}"
    )


def _extract_json(raw: str) -> dict:
    """
    Parse the agent's reply as a JSON object.
    Strips an accidental markdown fence, then falls back to the span from the
    first "{" to the last "}" if prose surrounds the object.
    Raises HTTPException 502 if no JSON object can be parsed.
    """
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        try:
            if start == -1 or end <= start:
                raise exc
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail=f"Exam Explainer Agent returned non-JSON output: {exc}. Raw: {raw[:300]}",
            ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Exam Explainer Agent returned {type(parsed).__name__}, expected a JSON object",
        )
    return parsed


def _validate_shape(parsed: dict, n_answers: int) -> None:
    """Require the same structure as the source prompt; raise 502 otherwise."""

    def bad(msg: str) -> HTTPException:
        return HTTPException(
            status_code=502, detail=f"Exam Explainer output rejected: {msg}"
        )

    for key in ("question", "explanation"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise bad(f"{key!r} must be a non-empty string")
    for key in ("answers", "feedbacks"):
        value = parsed.get(key)
        if not isinstance(value, list) or len(value) != n_answers:
            got = len(value) if isinstance(value, list) else type(value).__name__
            raise bad(f"{key!r} must be a list of {n_answers} items, got {got}")
        if not all(isinstance(v, str) and v.strip() for v in value):
            raise bad(f"every item in {key!r} must be a non-empty string")
        if key == "answers" and len({v.strip().lower() for v in value}) != len(value):
            raise bad("answers must be distinct (options merged?)")


async def simplify_item(
    ica: IcaClient,
    item: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> tuple[SimplifiedPrompt, IcaMeta]:
    """
    Run one exam item (needs `prompt`, `correct_response`, `section`) through the
    ICA chat model and return the validated simplified prompt plus call metadata.
    """
    n_answers = len((item.get("prompt") or {}).get("answers", []))
    if n_answers == 0:
        raise HTTPException(
            status_code=422, detail="question has no answers to simplify"
        )
    data = await ica.chat_model(
        model_id,
        _build_user_message(item),
        system=_build_system_message(n_answers),
        max_tokens=max_tokens,
    )
    choice = (data.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        raise HTTPException(
            status_code=502,
            detail=f"ICA model stopped with finish_reason={finish_reason!r} — cannot parse output",
        )
    parsed = _extract_json((choice.get("message") or {}).get("content") or "")
    _validate_shape(parsed, n_answers)

    return (
        SimplifiedPrompt(
            question=parsed["question"],
            answers=parsed["answers"],
            feedbacks=parsed["feedbacks"],
            explanation=parsed["explanation"],
        ),
        IcaMeta(
            model_id=model_id,
            finish_reason=finish_reason,
            total_tokens=(data.get("usage") or {}).get("total_tokens", 0),
        ),
    )


async def simplify_and_store(
    ica: IcaClient,
    simplified: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> SimplifiedQuestion:
    """
    Simplify one examination document and upsert the result into
    `examination_simplified`, keyed by `question_id`. An existing record is
    replaced and keeps its `created_at`.
    """
    prompt, meta = await simplify_item(ica, doc, model_id, max_tokens)

    now = datetime.now(timezone.utc)
    fields = {
        "section": doc.get("section", ""),
        "assessment_type": doc.get("assessment_type", ""),
        "source": doc.get("source", ""),
        "correct_response": doc.get("correct_response", []),
        "related_lectures": doc.get("related_lectures", []),
        "prompt": prompt.model_dump(),
        "ica": meta.model_dump(),
        "updated_at": now,
    }
    stored = await simplified.find_one_and_update(
        {"question_id": str(doc["_id"])},
        {"$set": fields, "$setOnInsert": {"created_at": now}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return SimplifiedQuestion(**stored)
