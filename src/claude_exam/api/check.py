"""Does the rephrase of a question still have the same right answer? (an ICA call)

`rephrase.py` lets a model rewrite a question and its answers. Its own checks cover
names, numbers and length; they cannot tell whether the rewrite moved the right answer,
by dropping the fact that decided it or by making a wrong option sound right. This step
tests that the way a player would meet it: the model gets the rephrase and its answers
WITHOUT the key and picks one (its messages are built from `KeyFreeQuestion`).

- It picks the correct answer: verdict "same".
- It picks another one: the ORIGINAL question is asked the same way. Right there means
  the rewrite is at fault: "changed", and the game stops offering that text
  (`game.domain.simplified_matches`). Wrong there too means the question is hard for the
  model in any wording: "hard", which says nothing against the rewrite.

The result is stored on the rephrase record (`check`); `rephrase_and_store` clears it
when the text changes.
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.game.domain import (
    CHECK_CHANGED,
    CHECK_HARD,
    CHECK_SAME,
    answers_of,
    correct_indices,
    letter_to_index,
)
from claude_exam.api.ica import IcaClient
from claude_exam.api.ica_json import ask_json
from claude_exam.api.scenario_catalog import BY_KEY, COMPANY_INTRO
from claude_exam.api.schemas import IcaMeta, IcaModelId, SimplifiedQuestion
from claude_exam.api.question_view import KeyFreeQuestion, key_free
from claude_exam.api.rephrase import chapter_of
from claude_exam.api.step_key import key_or_none, step_key

logger = logging.getLogger(__name__)

# Bump when the verdict logic changes (the prompt is in the input key, `step_key.py`).
CODE_VERSION = "1"

_SYSTEM_MESSAGE = """\
You answer one multiple-choice question of an exam about Claude Code, the Claude \
Agent SDK, the Claude API, MCP, prompt engineering, tool use and agent design. \
Exactly one answer is correct. You are checking the question, so answer as a \
well-prepared candidate would, from what the question says and what you know of \
these products.

Some questions are set in a company; then its background is given first. It is \
background only.

OUTPUT FORMAT - respond with ONLY a single valid JSON object and nothing else - no \
markdown fences, no prose, no comments. Exactly this schema:

{
  "reason": "<one sentence: the fact that decides it>",
  "answer": {{LETTERS}}
}\
"""


def _build_system_message(n_answers: int) -> str:
    letters = " | ".join(f'"{chr(65 + i)}"' for i in range(n_answers))
    return _SYSTEM_MESSAGE.replace("{{LETTERS}}", letters)


def _build_user_message(question: str, answers: list, background: str = "") -> str:
    lines = "\n".join(f"  {chr(65 + i)}. {a}" for i, a in enumerate(answers))
    head = f"BACKGROUND:\n{background}\n\n" if background else ""
    return f"{head}QUESTION:\n{question}\n\nANSWERS:\n{lines}"


def parse_pick(parsed: dict, n_answers: int) -> int:
    """The index of the answer the model picked; 502 when it named none."""
    index = letter_to_index(parsed.get("answer"))
    if not 0 <= index < n_answers:
        raise HTTPException(
            status_code=502,
            detail=f"Check output rejected: 'answer' must be one of the answer letters, "
            f"got {parsed.get('answer')!r}",
        )
    return index


async def _pick(
    ica: IcaClient,
    model_id: IcaModelId,
    system: str,
    user: str,
    n_answers: int,
    max_tokens: int | None,
) -> tuple[int, IcaMeta]:
    parsed, meta = await ask_json(
        ica, model_id, user, system=system, what="the check", max_tokens=max_tokens
    )
    return parse_pick(parsed, n_answers), meta


def messages(
    q: KeyFreeQuestion, simplified: dict, story: dict | None = None
) -> tuple[str, str, str, str]:
    """The messages of both checks, key-free: (system, user) of the rephrase, with the
    company and chapter as background, then of the original (asked only when the first
    pick is wrong; its messages don't depend on that reply). 422 when the rephrase has
    another number of answers."""
    answers = answers_of(simplified)
    # Not retried by a job (unlike 502): asking again cannot help.
    if len(answers) != len(q.answers):
        raise HTTPException(
            status_code=422, detail="the check needs a rephrase with the same answers"
        )
    chapter = BY_KEY.get(chapter_of(story) or "")
    background = f"{COMPANY_INTRO}\n{chapter.intro}" if chapter else COMPANY_INTRO
    prompt = simplified.get("prompt") or {}
    system = _build_system_message(len(answers))
    return (
        system,
        _build_user_message(prompt.get("question") or "", answers, background),
        system,
        _build_user_message(q.question, list(q.answers), ""),
    )


def input_key(doc: dict, simplified: dict, story: dict | None = None) -> str | None:
    """The input key of the check of a rephrase (`step_key.py`)."""
    return key_or_none(lambda: step_key(CODE_VERSION, *messages(key_free(doc), simplified, story)))


async def check_item(
    ica: IcaClient,
    doc: dict,
    simplified: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    story: dict | None = None,
) -> dict:
    """Check the rephrase in `simplified` against its original `doc` and return the
    `check` field to store. `story` is what `game.domain.story_of` returned for the
    question (its chapter gives the background a player has read). The models answer
    without the key; only the verdict compares their picks with it."""
    correct = correct_indices(doc)
    # Not retried by a job (unlike 502): asking again cannot help.
    if len(correct) != 1:
        raise HTTPException(status_code=422, detail="the check needs exactly one correct answer")
    system, user, original_system, original_user = messages(key_free(doc), simplified, story)
    n_answers = len(answers_of(doc))
    picked, meta = await _pick(ica, model_id, system, user, n_answers, max_tokens)
    picked_original = None
    if picked == correct[0]:
        verdict = CHECK_SAME
    else:
        picked_original, second = await _pick(
            ica, model_id, original_system, original_user, n_answers, max_tokens
        )
        meta.total_tokens += second.total_tokens
        verdict = CHECK_CHANGED if picked_original == correct[0] else CHECK_HARD
        logger.info(
            "check of %s: picked %s on the rephrase and %s on the original (correct %s): %s",
            doc.get("_id"),
            chr(65 + picked),
            chr(65 + picked_original),
            chr(65 + correct[0]),
            verdict,
        )
    return {
        "verdict": verdict,
        "picked": picked,
        "picked_original": picked_original,
        "input_key": step_key(
            CODE_VERSION, system, user, original_system, original_user
        ),
        "ica": meta.model_dump(),
    }


async def check_and_store(
    ica: IcaClient,
    simplified: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    story: dict | None = None,
) -> SimplifiedQuestion:
    """Check the stored rephrase of one examination question and store the result on
    its record (only `check` and `updated_at` change). Errors propagate."""
    question_id = str(doc["_id"])
    existing = await simplified.find_one({"question_id": question_id})
    if existing is None:
        raise HTTPException(status_code=404, detail="question has no rephrase")

    result = await check_item(ica, doc, existing, model_id, max_tokens, story)
    # Conditional like the entity step: the verdict is about the text that was sent.
    stored = await simplified.find_one_and_update(
        {"question_id": question_id, "updated_at": existing["updated_at"]},
        {"$set": {"check": result, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(
            status_code=409,
            detail="rephrase changed while it was checked; retry",
        )
    return SimplifiedQuestion(**stored)
