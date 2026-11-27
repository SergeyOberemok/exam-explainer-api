"""The beat of a question (an ICA call, key-free): the one story line shown BEFORE it.

The story step (`story.py`) places a question in its chapter and stage and writes the
lines shown after the answer; it reads the key, which those lines need. The beat is read
before answering, so it is written here, from the
`KeyFreeQuestion`: the model sees the company, the chapter (with its people, when it
has any), the stage
the story step chose, the question and the answers the player will see, and never the
correct answer, the feedback or the explanation.

A reply must pass `beat_errors`: one line of at most `story.MAX_LINE_WORDS` words that
gives no answer away (`leaks.leak`, which knows the answer-only entities when the
question has a decomposition). A failing reply is sent back with its faults.

Stored on the question's `question_story` record (`beat`, `beat_ica`, `beat_input_key`),
conditional on the record not having changed meanwhile; writing the story clears it. A
seed record keeps its hand-written beat (current as it is).
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.exam_terms import widened
from claude_exam.api.game.domain import story_of, usable_story
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_validated
from claude_exam.api.leaks import answer_only, leak_errors, terms_key
from claude_exam.api.question_view import KeyFreeQuestion, key_free
from claude_exam.api.scenario_catalog import BY_KEY, COMPANY, COMPANY_INTRO, STAGES
from claude_exam.api.schemas import IcaMeta, IcaModelId, QuestionDecomposition, QuestionStory
from claude_exam.api.step_key import key_or_none, step_key
from claude_exam.api.story import (
    MAX_LINE_CHARS,
    MAX_LINE_WORDS,
    PROMPT_LINE_WORDS,
    people_line,
    person_rule,
)

logger = logging.getLogger(__name__)

# Bump when `beat_errors` changes what is stored from the same reply (the prompt is in
# the input key, `step_key.py`).
CODE_VERSION = "2"
MAX_ATTEMPTS = 3

_SYSTEM_MESSAGE = """\
You write one line of a quiz game's story: the line shown just BEFORE a multiple-choice \
exam question. The exam is about Claude Code, the Claude Agent SDK, the Claude API, MCP, \
prompt engineering, tool use and agent design.

THE STORY
{{COMPANY_INTRO}}

{{CHAPTER}}

THIS QUESTION'S STAGE - {{STAGE}}

WHAT TO WRITE
The beat: what lands on the player's desk. State the problem or the decision as the \
team sees it, with the one or two concrete facts of the question that make it real (a \
number, a symptom). You do not know which answer is correct, and the line must not \
hint at any answer: do not name what an answer proposes, do not repeat an answer, and \
do not say what should be done.

STYLE
- At most {{LINE_WORDS}} words, plain English, present tense, about {{COMPANY}}'s \
people and systems.
{{PERSON_RULE}}
- No exclamation marks, no "you".
- Do not repeat the question's sentences. The line is a symptom or a situation, not a \
summary.
- Keep each fact's scope and wording: "most" stays "most", "silent or ambiguous" does \
not become "never covers", and no "all", "every" or "never" the question does not use.

EXAMPLES OF THE STYLE - from other questions; never reuse their content.
- "In 9% of runs the report is written before synthesis has finished, and citations \
are missing."
- "On pull requests with 40 files the reviewer contradicts itself and misses changes \
that span files."
- "The support team wants every pull request on the refund tools reviewed by Claude in CI, with a \
script deciding pass or fail."

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{"beat": "<one line>"}\
"""


def _chapter(key: str) -> str:
    scenario = BY_KEY.get(key)
    if scenario is None:
        return f"{COMPANY}'s team (no chapter)"
    return (
        f'THIS QUESTION\'S CHAPTER - "{scenario.title}"\n{scenario.intro}'
        f"{people_line(scenario)}"
    )


def messages(q: KeyFreeQuestion, place: dict) -> tuple[str, str]:
    """The (system, user) messages that write the beat, set where `place` (what
    `game.domain.story_of` returned) puts the question. Only its chapter and stage are
    read, never an earlier beat. 422 without answers or without a stage (no usable
    story record)."""
    if not q.answers:
        raise HTTPException(status_code=422, detail="question has no answers")
    stage = next((s for s in STAGES if s.key == place.get("stage")), None)
    if stage is None:
        raise HTTPException(status_code=422, detail="question has no story stage")
    scenario = BY_KEY.get(place.get("scenario") or "")
    system = (
        _SYSTEM_MESSAGE.replace("{{COMPANY_INTRO}}", COMPANY_INTRO)
        .replace("{{COMPANY}}", COMPANY)
        .replace("{{CHAPTER}}", _chapter(place.get("scenario") or ""))
        .replace("{{STAGE}}", f"{stage.title}: {stage.description}")
        .replace("{{LINE_WORDS}}", str(PROMPT_LINE_WORDS))
        .replace("{{PERSON_RULE}}", person_rule(bool(scenario and scenario.cast)))
    )
    choices = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(q.answers))
    user = (
        f"QUESTION:\n{q.stem or q.question}\n\n"
        "ANSWERS THE PLAYER WILL SEE AFTER THE LINE (name or hint none of them):\n"
        f"{choices}"
    )
    return system, user


def input_key(
    doc: dict, story_record: dict | None, decomposition: QuestionDecomposition | None = None
) -> str | None:
    """The input key of the beat of `doc` (`step_key.py`); None when it cannot have one.
    It also covers the answer-only entities the leak gate tested (`leaks.terms_key`), so
    a new decomposition that names one outdates the beat."""
    q = key_free(doc)

    def build() -> str:
        parts = list(messages(q, story_of(doc, story_record)))
        terms = terms_key(answer_only(q, decomposition).narrow())
        return step_key(CODE_VERSION, *parts, *([terms] if terms else []))

    return key_or_none(build)


def beat_errors(
    q: KeyFreeQuestion, parsed: dict, decomposition: QuestionDecomposition | None = None
) -> list[str]:
    """What makes a reply unusable (empty when it passes)."""
    text = parsed.get("beat")
    text = " ".join(text.split()) if isinstance(text, str) else ""
    if not text:
        return ['"beat" must be one sentence']
    errors = []
    if len(text.split()) > MAX_LINE_WORDS or len(text) > MAX_LINE_CHARS:
        errors.append(f"the beat is longer than {MAX_LINE_WORDS} words")
    sweeping = widened(q.question, text)
    if sweeping:
        errors.append(
            f"the beat says {', '.join(sweeping)}, which the question does not: keep "
            "its own scope (most, some, these), never widen a fact"
        )
    errors += leak_errors(
        [("the beat", text)],
        answer_only(q, decomposition).narrow(),
        "the line before the question must not say what any answer proposes",
    )
    return errors


async def beat_item(
    ica: IcaClient,
    doc: dict,
    place: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> tuple[str, IcaMeta]:
    """The beat of one question (502 when the last reply still fails `beat_errors`)."""
    q = key_free(doc)
    system, user = messages(q, place)
    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=system,
        what="beat",
        validate=lambda parsed: beat_errors(q, parsed, decomposition),
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        label=doc.get("_id"),
    )
    return " ".join(parsed["beat"].split()), meta


async def beat_and_store(
    ica: IcaClient,
    story: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> QuestionStory:
    """Write the beat of a question onto its story record. 404 without a usable story
    record (the stage comes from it); 409 when the record changed meanwhile."""
    record = await story.find_one({"question_id": str(doc["_id"])})
    if usable_story(doc, record) is None:
        raise HTTPException(status_code=404, detail="question has no current story record")
    beat, meta = await beat_item(
        ica, doc, story_of(doc, record), model_id, max_tokens, decomposition
    )
    stored = await story.find_one_and_update(
        {"question_id": record["question_id"], "updated_at": record["updated_at"]},
        {
            "$set": {
                "beat": beat,
                "beat_ica": meta.model_dump(),
                "beat_input_key": input_key(doc, record, decomposition),
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(
            status_code=409, detail="story changed while the beat was written; retry"
        )
    return QuestionStory(**stored)
