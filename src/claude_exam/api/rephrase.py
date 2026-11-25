"""The rephrase of a question (an ICA call, optional): the same question and answers in
plain words, from another angle, told as a moment in the game's story.

The game is built from the original question, its decomposition, flow and entities; the
rephrase is a second wording a player may switch to when the original is hard to read.
The original stays what is scored. The model reads the question through
`KeyFreeQuestion` (`question_view.py`): it never sees the correct answer, the feedback or
the explanation, so it cannot steer the wording towards the key. The answers keep their
order (index-aligned with the original). Stored in `examination_simplified` (`prompt`:
`question`, `answers`), one record per question; the check step (`check.py`) then tests
that a model still finds the right answer in it.

Because a model writes it, a reply has to pass hard checks before it is stored
(`fidelity_errors`): the exact names the exam tests, the numbers of the question and
the length of each answer. A reply that fails is sent back with what was wrong, up to
`MAX_ATTEMPTS` replies in all; after that it is rejected (502, which a job retries).
"""

import hashlib
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.exam_terms import lost_identifiers, lost_names, lost_numbers
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_validated
from claude_exam.api.question_view import KeyFreeQuestion, as_view, key_free
from claude_exam.api.scenario_catalog import BY_KEY, COMPANY, COMPANY_INTRO, STAGES
from claude_exam.api.schemas import IcaMeta, IcaModelId, Rephrase, SimplifiedQuestion
from claude_exam.api.step_key import key_or_none, step_key

logger = logging.getLogger(__name__)

# Bump when the gates (`fidelity_errors`, `_validate_shape`) change what is stored from
# the same reply (the prompt itself is in the input key, `step_key.py`).
CODE_VERSION = "1"
# How much longer than its original a rewritten answer may be before it is rejected.
MAX_ANSWER_GROWTH = 1.25
ANSWER_SLACK_CHARS = 15
# Attempts per call of `rephrase_item`: the first reply, then repairs.
MAX_ATTEMPTS = 3

_SYSTEM_MESSAGE = """\
You are a patient exam coach for the Claude Certified Architect / Developer \
certification and the writer of a quiz game built on its questions. You have deep \
familiarity with Claude Code, the Claude Agent SDK, the Claude API, MCP, prompt \
engineering, tool use, context management, evals and agent design. In the game every \
exam question happens inside one company, and the player is that company's Claude \
architect.

For one multiple-choice question you write a REPHRASE: the same question with the same \
answers, in plain words and from another angle, told as a moment in the game's story, \
so a learner who found the original hard to read understands what is asked. You are \
not told which answer is correct; do not try to decide it, and do not make any answer \
look better or worse than in the original.

THE STORY
{{COMPANY_INTRO}}
{{CHAPTER}}

THE QUESTION
- 2 to 4 short sentences, then the ask as a question. Address the player as "you".
- Sentence one sets the moment in the company. {{PEOPLE_RULE}}
- The story adds only who is asking and where. Every fact of the original stays: \
each number and percentage as digits, each limit, tool, setting, condition and \
symptom. Add no technical fact, cause or hint the original does not state.
- The ask asks exactly what the original asks: do not widen or change it.
{{BACKGROUND_RULE}}

THE ANSWERS
- An answer is the same proposal in plain words: one or two short sentences, no \
people, no story. It must not be longer than the original answer.
- Keep each detail that tells one option from another (extra steps, conditions, \
qualifiers).

PLAIN LANGUAGE
Short sentences (at most about 20 words), everyday words, active voice, concrete over \
abstract. Rewrite in your own words: never copy an original sentence of more than 20 \
words word for word.

EXACT TERMS
Keep every product, mode, tool, setting, parameter, hook, file and command name \
exactly as written (for example `acceptEdits`, `dontAsk`, plan mode, `PreToolUse`, \
`.mcp.json`, `claude -p`, MCP, subagent) - never replace one with a paraphrase, and \
keep it in the same answer it came from. If a term needs explaining, put the plain \
words next to the exact name: "`dontAsk` mode (silently denies anything not \
pre-approved)".

EXAMPLE - one question of another chapter, to show the level of story and of plain \
language. Never reuse its content.
Original question: The system prompt for this agent currently instructs it to \
"escalate to a human whenever the issue seems too complex to resolve or the customer \
appears frustrated." Production transcripts show the agent escalates simple billing \
corrections when customers use strong language, yet attempts genuinely ambiguous \
policy-exception disputes on its own, and first-contact resolution is stuck at 64%. \
Which prompt change would most effectively make the escalation behavior consistent \
and improve resolution rates?
Rephrase: Mara, head of support, shows you the transcripts. The agent's system \
prompt says to escalate "whenever the issue seems too complex to resolve or the \
customer appears frustrated." So it escalates simple billing corrections when \
customers use strong language, yet handles unclear policy-exception disputes alone. \
First-contact resolution is stuck at 64%. Which prompt change makes escalation \
consistent and raises the resolution rate?
Original answer: Have the agent output a self-reported confidence score with each \
resolution decision and escalate any case where the confidence score falls below a \
fixed threshold.
Rephrase: Have the agent report a confidence score with each decision and escalate \
when the score is below a fixed threshold.

OUTPUT FORMAT - you MUST respond with ONLY a single valid JSON object and nothing \
else - no markdown fences, no prose before or after, no comments. \
Respond with exactly this schema:

{
  "question": "<the rephrased question - plain text, no HTML>",
  {{ANSWERS_SCHEMA}}
}

Rules:
- Preserve the number of answers exactly (same count as input).
- Keep the order: answers[i] must rewrite the input option with the same letter (A is \
index 0, B is index 1, and so on). Never reorder or merge options.
- Output valid JSON. Do not wrap it in a code block or add any surrounding text.\
"""

_WITH_CHAPTER = (
    "- The player has read the chapter's introduction above and the line shown before "
    "the question (given with the question). Do not retell either, and do not open "
    "with the words of that line. The question's SCENARIO paragraph is that same "
    "background: take from it only what this question cannot be understood without."
)
_WITHOUT_CHAPTER = (
    "- The player has read nothing about this system. If the question comes with a "
    "SCENARIO paragraph, keep what the question needs from it: what is being built, "
    "with which tools, and the goal."
)


def opening(q: KeyFreeQuestion, chapter: str | None) -> str:
    """Who brings the question to the player, as a line for the model. Decided here and
    not by the model, which would open every question with a person: about one question
    in three gets one of the chapter's people (always the same one for the same text),
    the others start with the situation."""
    cast = BY_KEY[chapter].cast if chapter in BY_KEY else ()
    text = "\n".join([q.question, *q.answers])
    number = int(hashlib.sha1(text.encode()).hexdigest()[:16], 16)
    if not cast or number % 3:
        return "OPENING: no person. Start with the situation itself."
    name, role = cast[(number // 3) % len(cast)]
    return f"OPENING: {name} ({role}) brings this to you."


def _build_system_message(n_answers: int, chapter: str | None = None) -> str:
    answers = ",\n    ".join(
        f'"<rephrase of answer {chr(65 + i)} - plain words, one or two sentences>"'
        for i in range(n_answers)
    )
    scenario = BY_KEY.get(chapter) if chapter else None
    if scenario is not None:
        people = ", ".join(f"{name} ({role})" for name, role in scenario.cast)
        block = f'\nTHIS QUESTION\'S CHAPTER - "{scenario.title}"\n{scenario.intro}'
        if people:
            block += f"\nPeople of the chapter: {people}"
        background = _WITH_CHAPTER
    else:
        block = ""
        background = _WITHOUT_CHAPTER
    people_rule = (
        "The OPENING line given with the question says who brings the problem. If it "
        "names a person, that person brings it: use the first name and the role, once. "
        "If it says no person, name nobody: start with the situation itself, as the "
        f"team at {COMPANY} or you would see it."
    )
    return (
        _SYSTEM_MESSAGE.replace("{{COMPANY_INTRO}}", COMPANY_INTRO)
        .replace("{{CHAPTER}}", block)
        .replace("{{PEOPLE_RULE}}", people_rule)
        .replace("{{BACKGROUND_RULE}}", background)
        .replace("{{ANSWERS_SCHEMA}}", f'"answers": [\n    {answers}\n  ]')
    )


def _build_user_message(q: KeyFreeQuestion, place: dict | None = None) -> str:
    """The user turn: where the question stands in the story, the scenario paragraph,
    the question and its answers. No key."""
    place = place or {}
    stage = next((s for s in STAGES if s.key == place.get("stage")), None)
    parts = [f"Section: {q.section}"]
    if stage is not None:
        parts.append(f"STAGE OF THE CHAPTER: {stage.title} - {stage.description}")
    if place.get("beat"):
        parts.append(f"LINE SHOWN BEFORE THE QUESTION:\n{place['beat']}")
    parts.append(opening(q, chapter_of(place)))
    if q.paragraph:
        parts.append(f"SCENARIO:\n{q.paragraph}")
    choices = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(q.answers))
    parts += [f"QUESTION:\n{q.stem}", f"ANSWER CHOICES:\n{choices}"]
    return "\n\n".join(parts)


def chapter_of(place: dict | None) -> str | None:
    """The chapter a rephrase is set in: the one `game.domain.story_of` placed the
    question in (None without one)."""
    key = (place or {}).get("scenario") or ""
    return key if key in BY_KEY else None


_COPY_RATIO = 0.85


def fidelity_errors(question: "KeyFreeQuestion | dict", parsed: dict) -> list[str]:
    """
    What makes a shape-valid reply unusable, as sentences the model can act on (empty
    when it passes). A rephrase may change the words, never what the exam tests:
    - every identifier of an original answer (`exam_terms.lost_identifiers`) is in the
      rewrite of that same answer, and every identifier and number of the question
      itself in the rewritten question (the scenario paragraph is background and may
      be left out);
    - every named concept of the question and its answers (`exam_terms.lost_names`)
      is still named somewhere in the rewritten question and answers;
    - no answer has grown past `MAX_ANSWER_GROWTH`.
    """
    q = as_view(question)
    originals, stem = list(q.answers), q.stem
    errors = []
    for i, (original, rewrite) in enumerate(zip(originals, parsed["answers"])):
        letter = chr(65 + i)
        lost = lost_identifiers(original, rewrite)
        if lost:
            errors.append(f"answer {letter} must keep these exact terms: {', '.join(lost)}")
        if len(rewrite) > len(original) * MAX_ANSWER_GROWTH + ANSWER_SLACK_CHARS:
            errors.append(
                f"answer {letter} is longer than the original "
                f"({len(rewrite)} characters, the original has {len(original)})"
            )
    lost = lost_identifiers(stem, parsed["question"])
    if lost:
        errors.append(f"the question must keep these exact terms: {', '.join(lost)}")
    lost = lost_names(
        "\n".join([stem, *originals]), "\n".join([parsed["question"], *parsed["answers"]])
    )
    if lost:
        errors.append(
            f"the question and its answers must still name: {', '.join(lost)} "
            "(use the exact word where the original uses it)"
        )
    numbers = lost_numbers(stem, parsed["question"])
    if numbers:
        errors.append(f"the question must keep these numbers as digits: {', '.join(numbers)}")
    return errors


def quality_warnings(question: "KeyFreeQuestion | dict", parsed: dict) -> list[str]:
    """
    Everything a report should show about a shape-valid reply: the hard problems of
    `fidelity_errors` (a stored record has none, unless it predates them) and one soft
    one that never rejects a reply, an answer that is (nearly) a copy of the original.
    Many answers of this bank are already short and exact, so a copy is not an error.
    """
    q = as_view(question)
    warnings = fidelity_errors(q, parsed)
    for i, (orig, new) in enumerate(zip(q.answers, parsed["answers"])):
        if SequenceMatcher(None, orig.lower(), new.lower()).ratio() >= _COPY_RATIO:
            warnings.append(f"answer {chr(65 + i)} is not rephrased (near copy)")
    return warnings


def _validate_shape(parsed: dict, n_answers: int) -> None:
    """Require the same structure as the source question; raise 502 otherwise."""

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Rephrase rejected: {msg}")

    if not isinstance(parsed.get("question"), str) or not parsed["question"].strip():
        raise bad("'question' must be a non-empty string")
    value = parsed.get("answers")
    if not isinstance(value, list) or len(value) != n_answers:
        got = len(value) if isinstance(value, list) else type(value).__name__
        raise bad(f"'answers' must be a list of {n_answers} items, got {got}")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise bad("every item in 'answers' must be a non-empty string")
    if len({v.strip().lower() for v in value}) != len(value):
        raise bad("answers must be distinct (options merged?)")


def messages(q: KeyFreeQuestion, place: dict | None = None) -> tuple[str, str]:
    """The (system, user) messages that rephrase the question, set where `place` (what
    `game.domain.story_of` returned) puts it; 422 without answers."""
    if not q.answers:
        raise HTTPException(status_code=422, detail="question has no answers to rephrase")
    return _build_system_message(len(q.answers), chapter_of(place)), _build_user_message(q, place)


def input_key(doc: dict, place: dict | None = None) -> str | None:
    """The input key of the rephrase of `doc` (`step_key.py`). It covers the chapter's
    intro and people and the story line, since they are in the messages."""
    return key_or_none(lambda: step_key(CODE_VERSION, *messages(key_free(doc), place)))


async def rephrase_item(
    ica: IcaClient,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    place: dict | None = None,
) -> tuple[Rephrase, IcaMeta]:
    """Rephrase one examination question and return the validated text plus the call
    metadata (counting the tokens of every attempt). A reply that breaks
    `fidelity_errors` is sent back with what was wrong; when the last of `MAX_ATTEMPTS`
    replies still breaks it the call raises 502."""
    q = key_free(doc)
    system, user = messages(q, place)

    def validate(parsed: dict) -> list[str]:
        _validate_shape(parsed, len(q.answers))
        return fidelity_errors(q, parsed)

    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=system,
        what="rephrase",
        validate=validate,
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        label=doc.get("_id"),
    )
    for warning in quality_warnings(q, parsed):
        logger.info("rephrase %s: %s", doc.get("_id"), warning)
    return Rephrase(question=parsed["question"], answers=parsed["answers"]), meta


async def rephrase_and_store(
    ica: IcaClient,
    simplified: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    place: dict | None = None,
) -> SimplifiedQuestion:
    """Rephrase one examination question and upsert it into `examination_simplified`
    (keyed by `question_id`; an existing record keeps its `created_at`). A check of an
    earlier text goes with it unless the text came back the same."""
    rephrase, meta = await rephrase_item(ica, doc, model_id, max_tokens, place)
    question_id = str(doc["_id"])
    existing = await simplified.find_one({"question_id": question_id})
    old = (existing or {}).get("prompt") or {}
    same_text = (
        existing is not None
        and old.get("question") == rephrase.question
        and old.get("answers") == rephrase.answers
    )
    now = datetime.now(timezone.utc)
    fields = {
        "section": doc.get("section", ""),
        "assessment_type": doc.get("assessment_type", ""),
        "source": doc.get("source", ""),
        "correct_response": doc.get("correct_response", []),
        "related_lectures": doc.get("related_lectures", []),
        "prompt": rephrase.model_dump(),
        "ica": meta.model_dump(),
        "input_key": input_key(doc, place),
        "updated_at": now,
    }
    if not same_text:
        fields["check"] = None
    stored = await simplified.find_one_and_update(
        {"question_id": question_id},
        {"$set": fields, "$setOnInsert": {"created_at": now}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return SimplifiedQuestion(**stored)
