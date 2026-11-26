"""Learning-objective tagging of exam questions (an ICA call of its own).

For one question the model picks, from the closed list in `objective_catalog.py`, the
objective it tests (and up to two more it touches), a thinking level, which wrong
answers are tempting (the difficulty follows from that), and for every wrong answer
what a learner who picks it gets wrong. The tags describe the
ORIGINAL question and its answer key, so they are stored in their own collection
(`question_objectives`, one record per question).

Unlike icons, nothing here is repaired with a guess: a reply without a usable main
objective is rejected, and the question stays untagged until it is tagged again
(`scripts/objective_coverage.py` lists what is untagged).
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.domain import answers_of, correct_indices, letter_to_index
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_json
from claude_exam.api.objective_catalog import (
    OBJECTIVES,
    for_section,
    known,
    normalize_section,
)
from claude_exam.api.schemas import IcaMeta, IcaModelId, QuestionObjectives
from claude_exam.api.step_key import key_or_none, step_key

logger = logging.getLogger(__name__)

# Bump when `parse_objectives` changes what is stored from the same reply (the prompt
# itself is in the input key, `step_key.py`).
CODE_VERSION = "1"
LEVELS = ("remember", "understand", "apply", "analyze")
MAX_SECONDARY = 2
MAX_MISCONCEPTION_CHARS = 160

_SYSTEM_MESSAGE = """\
You tag one multiple-choice exam question with the learning objective it tests, so a \
quiz game can group questions and tell a learner what a wrong answer reveals. The exam \
is about Claude Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, \
tool use and agent design.

OBJECTIVES OF THIS QUESTION'S SECTION - "key: what the learner can do". Use ONLY keys \
from this list and the next one, spelled exactly.
{{SECTION_OBJECTIVES}}

OBJECTIVES OF OTHER SECTIONS - the exam's sections overlap, so a question can test one \
of these instead.
{{OTHER_OBJECTIVES}}

WHAT TO CHOOSE
1. "primary": the one objective a learner must have reached to pick the correct answer \
for the right reason. Decide from the correct answer and the explanation, not from the \
story the question is set in. Prefer an objective of this question's section; take one \
of another section only when it clearly fits better.
2. "secondary": up to {{MAX_SECONDARY}} other objectives the question also needs, from \
any section. Usually none.
3. "level": the thinking the question asks for.
   - "remember": recall a name, flag, value or fact.
   - "understand": explain what a feature does or how two things differ.
   - "apply": use a known rule or feature in the given situation.
   - "analyze": find the cause of a problem or weigh workable options against each other.
4. "tempting": the letters of the WRONG answers that are tempting: an answer that \
would partly work, or that a prepared learner could defend for a moment. An answer \
that is plainly off (wrong tool for the job, solves another problem, makes it worse) \
is not tempting. Judge each wrong answer on its own; the list is often empty or has \
one letter, and has two or three only when the options are real trade-offs.
5. "distractors": one entry per WRONG answer, under that answer's letter. The CORRECT \
answer gets no entry. Each entry is an object with:
   - "misconception": one sentence of at most 15 words stating the wrong belief \
itself, as a plain claim, e.g. "A system prompt rule guarantees the order of tool \
calls." Do not start with "Believes" or "The learner", do not say that the answer is \
wrong, and do not name the correct answer. Every wrong answer gets one, however close \
to correct it is.
   - "confused_with": the key of the objective this wrong answer belongs to, when it \
would be right for a different objective; otherwise null.

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{
  "primary": "<key>",
  "secondary": ["<key>"],
  "level": "remember" | "understand" | "apply" | "analyze",
  "tempting": ["<letter>"],
  "distractors": {{{DISTRACTORS_SCHEMA}}}
}

Rules:
- "distractors" has exactly these keys: {{LETTERS}}. Each entry is about the answer with \
that letter, so check the letter against the answer list before you write it.
- Every key is copied from the lists above.
- Output valid JSON only.\
"""


def difficulty_of(level: str, tempting: int) -> int:
    """1 (easy) to 3 (hard), from what the model observed instead of its own estimate
    (which came out as 2 for nearly every question): how many wrong answers tempt, and
    the thinking asked for. The steps are relative to this exam, whose questions nearly
    all apply a rule to a situation and have at least one tempting wrong answer:
    - 3: every wrong answer tempts, or two do and the question asks to analyze;
    - 1: at most one tempting answer and no analysis asked;
    - 2: everything else."""
    if tempting >= 3 or (tempting == 2 and level == "analyze"):
        return 3
    if tempting <= 1 and level != "analyze":
        return 1
    return 2


def _listing(objectives) -> str:
    return "\n".join(f"- {o.key}: {o.statement}" for o in objectives) or "(none)"


def _build_system_message(section: str, n_answers: int, correct: list[int]) -> str:
    own = for_section(section)
    others = [o for o in OBJECTIVES if o not in own]
    wrong = [chr(65 + i) for i in range(n_answers) if i not in correct]
    items = ", ".join(
        f'"{letter}": {{"misconception": "<for {letter}>", "confused_with": "<key>" | null}}'
        for letter in wrong
    )
    return (
        _SYSTEM_MESSAGE.replace("{{SECTION_OBJECTIVES}}", _listing(own))
        .replace("{{OTHER_OBJECTIVES}}", _listing(others))
        .replace("{{MAX_SECONDARY}}", str(MAX_SECONDARY))
        .replace("{{DISTRACTORS_SCHEMA}}", items)
        .replace("{{LETTERS}}", ", ".join(wrong))
    )


def _build_user_message(doc: dict, correct: list[int]) -> str:
    prompt = doc.get("prompt") or {}
    answers = answers_of(doc)
    feedbacks = prompt.get("feedbacks") or []
    lines = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(answers))
    notes = "\n".join(
        f"  {chr(65 + i)}. {feedbacks[i]}" for i in range(len(answers)) if i < len(feedbacks) and feedbacks[i]
    )
    return (
        f"Section: {normalize_section(doc.get('section'))}\n\n"
        f"QUESTION:\n{prompt.get('question') or ''}\n\n"
        f"ANSWERS:\n{lines}\n\n"
        f"CORRECT ANSWER: {', '.join(chr(65 + i) for i in correct)}\n\n"
        f"FEEDBACK PER ANSWER:\n{notes or '(none)'}\n\n"
        f"EXPLANATION:\n{prompt.get('explanation') or '(none)'}"
    )


def parse_objectives(parsed: dict, section: str, n_answers: int, correct: list[int]) -> dict:
    """
    Validate the model's reply and return the fields to store: `objectives` (main one
    first), `level`, `tempting`, `difficulty` (derived, see `difficulty_of`),
    `distractors`. Anything that decides how the question
    is grouped or what a wrong answer means must be right, or the reply is rejected
    (502, so a job retries it). Only the optional extras are trimmed: an unknown
    secondary key or `confused_with` is dropped. The main objective may be of another
    section than the question's: the source's sections overlap, and the model reads the
    question while the section is only a label.
    """

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Objectives output rejected: {msg}")

    primary = known(parsed.get("primary"))
    if primary is None:
        raise bad(f"'primary' must be a key of the objective catalog, got {parsed.get('primary')!r}")
    if primary not in {o.key for o in for_section(section)}:
        # Allowed: the source's sections overlap. The coverage report counts these.
        logger.info("main objective %r is of another section than %r", primary, section)

    raw_secondary = parsed.get("secondary") or []
    if not isinstance(raw_secondary, list):
        raise bad("'secondary' must be a list of keys")
    secondary: list[str] = []
    for key in raw_secondary:
        if known(key) and key != primary and key not in secondary:
            secondary.append(key)
        else:
            logger.info("dropping secondary objective %r", key)

    level = parsed.get("level")
    if level not in LEVELS:
        raise bad(f"'level' must be one of {', '.join(LEVELS)}, got {level!r}")
    raw_tempting = parsed.get("tempting")
    if not isinstance(raw_tempting, list):
        raise bad("'tempting' must be a list of answer letters")
    # Only wrong answers can tempt; anything else in the list is dropped.
    tempting = sorted(
        {
            i
            for i in map(letter_to_index, raw_tempting)
            if 0 <= i < n_answers and i not in correct
        }
    )

    raw = parsed.get("distractors")
    if isinstance(raw, dict):
        # Keyed by answer letter (what the prompt asks for: a position in a list proved
        # easy to get wrong). An entry for a letter that is no answer is an error.
        letters = [chr(65 + i) for i in range(n_answers)]
        unknown = sorted(set(raw) - set(letters))
        if unknown:
            raise bad(f"'distractors' has entries for {', '.join(map(str, unknown))}, which are no answers")
        raw = [raw.get(letter) for letter in letters]
    if not isinstance(raw, list) or len(raw) != n_answers:
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise bad(f"'distractors' must be a list of {n_answers} items, got {got}")
    distractors: list[dict | None] = []
    for i, item in enumerate(raw):
        letter = chr(65 + i)
        if i in correct:
            if item is not None:
                raise bad(f"distractor {letter} must be null: it is the correct answer")
            distractors.append(None)
            continue
        text = item.get("misconception") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise bad(f"distractor {letter} needs a 'misconception' sentence")
        confused = known(item.get("confused_with"))
        distractors.append(
            {
                "confused_with": None if confused == primary else confused,
                "misconception": text.strip()[:MAX_MISCONCEPTION_CHARS],
            }
        )

    return {
        "objectives": [primary, *secondary[:MAX_SECONDARY]],
        "level": level,
        "tempting": tempting,
        "difficulty": difficulty_of(level, len(tempting)),
        "distractors": distractors,
    }


def messages(doc: dict) -> tuple[str, str]:
    """The (system, user) messages that tag `doc`; 422 when it cannot be tagged."""
    section = doc.get("section") or ""
    n_answers = len(answers_of(doc))
    correct = correct_indices(doc)
    # Not retried by a job (unlike 502): asking again cannot help.
    if not n_answers or not correct:
        raise HTTPException(status_code=422, detail="question has no answers or no answer key")
    if not for_section(section):
        raise HTTPException(
            status_code=422,
            detail=f"no objectives for section {normalize_section(section)!r}; "
            "add them to objective_catalog.py",
        )
    return _build_system_message(section, n_answers, correct), _build_user_message(doc, correct)


def input_key(doc: dict) -> str | None:
    """The input key of the objectives of `doc` (`step_key.py`); None when it cannot be
    tagged."""
    return key_or_none(lambda: step_key(CODE_VERSION, *messages(doc)))


async def tag_question(
    ica: IcaClient, doc: dict, model_id: IcaModelId, max_tokens: int | None = None
) -> tuple[dict, IcaMeta]:
    """Ask the ICA chat model for the objective tags of one `examination` question."""
    system, user = messages(doc)
    parsed, meta = await ask_json(
        ica, model_id, user, system=system, what="objectives", max_tokens=max_tokens
    )
    section = doc.get("section") or ""
    return parse_objectives(parsed, section, len(answers_of(doc)), correct_indices(doc)), meta


async def tag_and_store(
    ica: IcaClient,
    objectives: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
) -> QuestionObjectives:
    """Tag one `examination` question and store the result, replacing an earlier record
    of the same question. Errors propagate and leave the earlier record as it was."""
    fields, meta = await tag_question(ica, doc, model_id, max_tokens)
    now = datetime.now(timezone.utc)
    stored = await objectives.find_one_and_update(
        {"question_id": str(doc["_id"])},
        {
            "$set": {
                **fields,
                "section": doc.get("section") or "",
                "input_key": input_key(doc),
                "ica": meta.model_dump(),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return QuestionObjectives(**stored)

