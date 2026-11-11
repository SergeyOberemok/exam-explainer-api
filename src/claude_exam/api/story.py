"""The story around a question: where it sits in its chapter and what the game says
before and after it (an ICA call of its own, or the hand-written seed).

The chapter comes from the question's text (`scenario_catalog.scenario_of`); for a
question without a scenario paragraph the model picks one of the chapters. This step
adds the rest: the `stage` of the chapter, a `beat` shown before the question, and the
`win` / `fail` line shown after the answer. The question and its answers are never
changed. Records live in `question_story`, one per question.

A record has an `origin`. "seed" records come from `data/story_seed.json`
(`scripts/load_story_seed.py`) and a backfill leaves them alone; "ica" records are
written here and are redone when the prompt or their chapter's key, marker or the stage
list changes.
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.domain import answers_of, correct_indices
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_json
from claude_exam.api.scenario_catalog import (
    CHAPTER_VERSIONS,
    COMPANY,
    COMPANY_INTRO,
    SCENARIOS,
    STAGES,
    ScenarioDef,
    chapter_version,
    known_scenario,
    known_stage,
    scenario_of,
    split_scenario,
    stem_of,
    text_hash,
    unknown_paragraph,
)
from claude_exam.api.schemas import IcaMeta, IcaModelId, QuestionStory

logger = logging.getLogger(__name__)

# Bump when the prompt or the meaning of a stored field changes: "ica" records of
# another version count as missing for a story backfill.
PROMPT_VERSION = "2"
# The prompt asks for less than the parser allows: a model counts words loosely.
PROMPT_LINE_WORDS = 16
MAX_LINE_WORDS = 20
# Hand-written seed lines may run a little longer.
SEED_LINE_WORDS = 30
MAX_LINE_CHARS = 220
# An answer this long found inside a line means the line gives the answer away.
MIN_LEAK_CHARS = 20
# Seed records are ordered by hand from 1. Records written by the model come after them,
# easier questions first: `ICA_ORDER` plus the question's difficulty (0 when untagged).
ICA_ORDER = 1000

_SYSTEM_MESSAGE = """\
You place one multiple-choice exam question into the story of a quiz game and write \
three short story lines for it. The exam is about Claude Code, the Claude Agent SDK, \
the Claude API, MCP, prompt engineering, tool use and agent design.

THE STORY
{{COMPANY_INTRO}}

{{CHAPTER}}

STAGES OF A CHAPTER - "key: when a question belongs to it"
{{STAGES}}

WHAT TO WRITE
{{PICK}}"stage": the one stage key that fits what happens in the question. Decide from \
the situation the question describes, not from its topic.
- "beat": shown BEFORE the question: what lands on the player's desk. State the \
problem or the decision as the team sees it, with the one or two concrete facts that \
make it real (a number, a symptom). It must not hint at the correct answer, name any \
answer, or say what should be done.
- "win": shown AFTER a correct answer: what is better in the project now, as an \
observable result. Do not restate the answer.
- "fail": shown AFTER a wrong answer: what keeps going wrong, as an observable \
result. Do not name or hint at the correct answer.

STYLE
- Each line is at most {{LINE_WORDS}} words, plain English, present tense, about \
{{COMPANY}}'s people and systems.
- Use a person only where it adds something, and then only one of the people listed \
for the chapter, by first name. Most lines need no person at all.
- No exclamation marks, no praise or blame for the player, no "you".
- Do not repeat the question's sentences. A line is a consequence or a symptom, not a \
summary.

EXAMPLES OF THE STYLE - from other questions; never reuse their content.
- beat: "In 9% of runs the report is written before synthesis has finished, and \
citations are missing."
  win: "The report writer cannot start until synthesis has delivered."
  fail: "The ordering still slips now and then, and reports go out incomplete."
- beat: "On pull requests with 40 files the reviewer contradicts itself and misses \
changes that span files."
  win: "Each file gets full attention, and what spans files is caught as well."
  fail: "The reviewer still flags a pattern in one file and approves it in the next."
- beat: "Omar wants every pull request on the refund tools reviewed by Claude in CI, \
with a script deciding pass or fail."
  win: "The pipeline reads the review and fails the build on critical findings, with \
nobody involved."
  fail: "The job waits for input nobody will give, or produces text the script cannot \
read."

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{
{{SCENARIO_FIELD}}  "stage": {{STAGE_KEYS}},
  "beat": "<one line>",
  "win": "<one line>",
  "fail": "<one line>"
}\
"""


def _people(scenario: ScenarioDef) -> str:
    return ", ".join(f"{name} ({role})" for name, role in scenario.cast) or "(none named)"


def needs_chapter_pick(doc: dict) -> bool:
    """True for a question without a scenario paragraph: the model then picks the
    chapter. A paragraph no chapter claims is not that case (it needs a new chapter)."""
    return not split_scenario((doc.get("prompt") or {}).get("question"))[0]


def _build_system_message(doc: dict) -> str:
    scenario = scenario_of(doc)
    if scenario is not None:
        chapter = (
            f'THIS QUESTION\'S CHAPTER - "{scenario.title}"\n{scenario.intro}\n'
            f"People: {_people(scenario)}"
        )
        pick, field = "- ", ""
    else:
        chapter = 'CHAPTERS - "key: what the team builds. People"\n' + "\n".join(
            f"- {s.key}: {s.intro} People: {_people(s)}" for s in SCENARIOS
        )
        pick = (
            '- "scenario": the key of the one chapter whose system the question is '
            "about. When the question names no system, take the chapter where its "
            "situation would most naturally come up.\n- "
        )
        field = "  \"scenario\": " + " | ".join(f'"{s.key}"' for s in SCENARIOS) + ",\n"
    return (
        _SYSTEM_MESSAGE.replace("{{COMPANY_INTRO}}", COMPANY_INTRO)
        .replace("{{COMPANY}}", COMPANY)
        .replace("{{CHAPTER}}", chapter)
        .replace("{{PICK}}", pick)
        .replace("{{SCENARIO_FIELD}}", field)
        .replace("{{LINE_WORDS}}", str(PROMPT_LINE_WORDS))
        .replace("{{STAGES}}", "\n".join(f"- {s.key}: {s.description}" for s in STAGES))
        .replace("{{STAGE_KEYS}}", " | ".join(f'"{s.key}"' for s in STAGES))
    )


def _build_user_message(doc: dict, correct: list[int]) -> str:
    lines = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(answers_of(doc)))
    question = stem_of(doc) or (doc.get("prompt") or {}).get("question") or ""
    return (
        f"QUESTION:\n{question}\n\n"
        f"ANSWERS:\n{lines}\n\n"
        f"CORRECT ANSWER: {', '.join(chr(65 + i) for i in correct)}"
    )


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def parse_story(
    parsed: dict,
    answers: list,
    *,
    correct: list[int] = (),
    max_words: int = MAX_LINE_WORDS,
    pick_chapter: bool = False,
) -> dict:
    """
    Validate a story reply (or a seed item) and return the fields to store: `stage`,
    `beat`, `win`, `fail`, and with `pick_chapter` the `scenario` the model chose.
    Anything wrong rejects it (502, so a job retries it): an unknown stage or chapter,
    an empty or over-long line, a beat that repeats an answer, or a `win` / `fail`
    line that repeats a correct one (`correct`: the indices of the correct answers).
    """

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Story output rejected: {msg}")

    stage = known_stage(parsed.get("stage"))
    if stage is None:
        raise bad(f"'stage' must be a stage key, got {parsed.get('stage')!r}")

    fields: dict = {"stage": stage}
    if pick_chapter:
        scenario = known_scenario(parsed.get("scenario"))
        if scenario is None:
            raise bad(f"'scenario' must be a chapter key, got {parsed.get('scenario')!r}")
        fields["scenario"] = scenario
    for name in ("beat", "win", "fail"):
        text = parsed.get(name)
        text = " ".join(text.split()) if isinstance(text, str) else ""
        if not text:
            raise bad(f"'{name}' must be a sentence")
        if len(text.split()) > max_words or len(text) > MAX_LINE_CHARS:
            raise bad(f"'{name}' is longer than {max_words} words")
        fields[name] = text

    squashed = [_squash(a) if isinstance(a, str) else "" for a in answers]
    for name in ("beat", "win", "fail"):
        line = _squash(fields[name])
        for i, answer in enumerate(squashed):
            # The beat is read before answering, so no answer may be in it; the lines
            # after it only must not spell out the correct one.
            if name != "beat" and i not in correct:
                continue
            if len(answer) >= MIN_LEAK_CHARS and answer in line:
                raise bad(f"'{name}' repeats answer {chr(65 + i)}")
    return fields


def record_fields(doc: dict, fields: dict, origin: str, order: int) -> dict:
    """What a `question_story` record holds besides its dates and ICA metadata. The
    chapter is the one the question's text names, else the one in `fields`."""
    fields = dict(fields)
    picked = fields.pop("scenario", None)
    scenario = scenario_of(doc)
    key = scenario.key if scenario else picked
    return {
        **fields,
        "text_hash": text_hash(doc),
        "scenario": key,
        # "text": read from the scenario paragraph; "model": picked for a question
        # without one.
        "scenario_origin": "text" if scenario else "model",
        "order": order,
        "origin": origin,
        "catalog_version": chapter_version(key),
        "prompt_version": PROMPT_VERSION,
    }


async def write_story(
    ica: IcaClient, doc: dict, model_id: IcaModelId, max_tokens: int | None = None
) -> tuple[dict, IcaMeta]:
    """Ask the ICA chat model for the story fields of one `examination` question."""
    correct = correct_indices(doc)
    # Not retried by a job (unlike 502): asking again cannot help.
    if not answers_of(doc) or not correct:
        raise HTTPException(status_code=422, detail="question has no answers or no answer key")
    if unknown_paragraph(doc):
        raise HTTPException(
            status_code=422,
            detail="the question's scenario is not in scenario_catalog.py",
        )

    parsed, meta = await ask_json(
        ica,
        model_id,
        _build_user_message(doc, correct),
        system=_build_system_message(doc),
        what="the story",
        max_tokens=max_tokens,
    )
    fields = parse_story(
        parsed, answers_of(doc), correct=correct, pick_chapter=needs_chapter_pick(doc)
    )
    return fields, meta


def current_query(question_id: str) -> dict:
    """Matches the question's record if a backfill has nothing to do for it: it is a
    seed record, or the model wrote it with today's prompt for a chapter as it is today."""
    return {
        "question_id": question_id,
        "$or": [
            {"origin": "seed"},
            {
                "catalog_version": {"$in": sorted(CHAPTER_VERSIONS)},
                "prompt_version": PROMPT_VERSION,
            },
        ],
    }


def upsert_update(fields: dict, ica: IcaMeta | None, now: datetime) -> dict:
    """The update that stores a record's fields, for the model and the seed alike."""
    return {
        "$set": {**fields, "ica": ica.model_dump() if ica else None, "updated_at": now},
        "$setOnInsert": {"created_at": now},
    }


async def story_and_store(
    ica: IcaClient,
    story: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    difficulty: int = 0,
) -> QuestionStory:
    """Write the story of one `examination` question and store it, replacing an earlier
    record of the same question (a seed record too: load the seed again to restore it).
    `difficulty` (1-3, 0 when unknown) places it among the model-written records of its
    stage. Errors propagate and leave the earlier record as it was."""
    fields, meta = await write_story(ica, doc, model_id, max_tokens)
    stored = await story.find_one_and_update(
        {"question_id": str(doc["_id"])},
        upsert_update(
            record_fields(doc, fields, "ica", ICA_ORDER + difficulty),
            meta,
            datetime.now(timezone.utc),
        ),
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return QuestionStory(**stored)
