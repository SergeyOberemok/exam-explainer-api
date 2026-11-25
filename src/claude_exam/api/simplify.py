"""The game text of a question (an ICA call): the same question in plain words, told
as a moment in the game's story.

The original question stays what the player answers and what is scored. This step
writes the second version the game shows on Reveal: plain language, set in the company
of `scenario_catalog.py`, with one of the chapter's people bringing the problem. It is
stored in `examination_simplified` (`prompt`), one record per question.

Because a model writes it, a reply has to pass hard checks before it is stored
(`fidelity_errors`): the exact names the exam tests, the numbers of the question and
the length of each answer. A reply that fails is sent back with what was wrong, up to
`MAX_ATTEMPTS` replies in all; after that it is rejected (502, which a job retries). A later
step (`check.py`) tests that the rewrite still has the same right answer.
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from claude_exam.api.entities import PROMPT_VERSION as entities_version
from claude_exam.api.entities import derive_entities, extract_entities
from claude_exam.api.exam_terms import lost_identifiers, lost_names, lost_numbers
from claude_exam.api.ica import IcaClient
from claude_exam.api import decompose
from claude_exam.api.ica_json import ask_validated
from claude_exam.api.icons import PROMPT_VERSION as icons_version
from claude_exam.api.icons import pick_icons
from claude_exam.api.scenario_catalog import (
    BY_KEY,
    COMPANY,
    COMPANY_INTRO,
    STAGES,
    frame_version,
    known_scenario,
    scenario_of,
    split_scenario,
    text_hash,
)
from claude_exam.api.schemas import (
    IcaMeta,
    IcaModelId,
    PromptEntities,
    QuestionDecomposition,
    SemanticPart,
    SimplifiedPrompt,
    SimplifiedQuestion,
)
from claude_exam.html_clean import clean_plain, clean_question

logger = logging.getLogger(__name__)

# Bump when the prompt or the meaning of a stored field changes: records of another
# version count as missing for a pipeline backfill.
PROMPT_VERSION = "3"
# How much longer than its original a rewritten answer may be before it is rejected.
MAX_ANSWER_GROWTH = 1.25
ANSWER_SLACK_CHARS = 15
# Attempts per call of `simplify_item`: the first reply, then repairs.
MAX_ATTEMPTS = 3

_SYSTEM_MESSAGE = """\
You are a patient exam coach for the Claude Certified Architect / Developer \
certification and the writer of a quiz game built on its questions. You have deep \
familiarity with Claude Code, the Claude Agent SDK, the Claude API, MCP, prompt \
engineering, tool use, context management, evals and agent design. In the game every \
exam question happens inside one company, and the player is that company's Claude \
architect.

For one multiple-choice question you write its GAME VERSION: the same question with \
the same answers, in plain words, told as a moment in the game's story. Your job is \
for the learner to understand and remember, not just to see the key.

THE STORY
{{COMPANY_INTRO}}
{{CHAPTER}}

HOW TO WORK - in your head, before you write:
1. Find the core problem and what is being asked.
2. For every answer option: the mechanism it relies on and its main limitation.
3. Find the single deciding fact that separates the correct answer from the rest.
Use the supplied explanation, per-option feedback and correct answer as the source of \
truth. Do not invent technical facts.

THE QUESTION
- 2 to 4 short sentences, then the ask as a question. Address the player as "you".
- Sentence one sets the moment in the company. {{PEOPLE_RULE}}
- The story adds only who is asking and where. Every fact of the original stays: \
each number and percentage as digits, each limit, tool, setting, condition and \
symptom. Add no technical fact, cause or hint the original does not state.
- Do not make the correct answer easier to find than in the original: no new clue, \
no wording borrowed from the correct answer.
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
pre-approved)". Named patterns and concepts of the official explanation (for example \
"evaluator-optimizer", "context isolation", "scoped tool access") are exact terms \
too: keep the name, explain it next to it, and name the concept in the closing rule.

FEEDBACK AND EXPLANATION
- Start each feedback with "Correct." for the correct option(s) and "Incorrect." for \
every other option, then give the reasoning in one or two short sentences. No story.
- The explanation is 2-3 sentences: the single deciding fact that makes the correct \
answer correct, why the closest wrong option fails, and last a one-line rule for \
similar questions. No story and no names here either: it is what the learner takes \
to the exam.

EXAMPLE - one question of another chapter, to show the level of story and of plain \
language. Never reuse its content.
Original question: The system prompt for this agent currently instructs it to \
"escalate to a human whenever the issue seems too complex to resolve or the customer \
appears frustrated." Production transcripts show the agent escalates simple billing \
corrections when customers use strong language, yet attempts genuinely ambiguous \
policy-exception disputes on its own, and first-contact resolution is stuck at 64%. \
Which prompt change would most effectively make the escalation behavior consistent \
and improve resolution rates?
Game version: Mara, head of support, shows you the transcripts. The agent's system \
prompt says to escalate "whenever the issue seems too complex to resolve or the \
customer appears frustrated." So it escalates simple billing corrections when \
customers use strong language, yet handles unclear policy-exception disputes alone. \
First-contact resolution is stuck at 64%. Which prompt change makes escalation \
consistent and raises the resolution rate?
Original answer: Have the agent output a self-reported confidence score with each \
resolution decision and escalate any case where the confidence score falls below a \
fixed threshold.
Game version: Have the agent report a confidence score with each decision and \
escalate when the score is below a fixed threshold.

OUTPUT FORMAT - you MUST respond with ONLY a single valid JSON object and nothing \
else - no markdown fences, no prose before or after, no comments. \
Respond with exactly this schema:

{
  {{QUESTION_SCHEMA}},
  {{ANSWERS_SCHEMA}},
  {{FEEDBACKS_SCHEMA}},
  "explanation": "<2-3 sentences ending with a one-line rule>"
}

Rules:
- Preserve the number of answers and feedbacks exactly (same count as input).
- Keep the order: answers[i] and feedbacks[i] must rewrite the input option with \
the same letter (A is index 0, B is index 1, and so on). Never reorder or merge options.
- Do not change which answer is correct - you are only rewriting text.
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


def chapter_key(item: dict, story: dict | None = None) -> str | None:
    """The chapter a question's game text is set in: the one its scenario paragraph
    names, else the one its story record names (`story`: what `game.domain.story_of`
    returned), else None."""
    scenario = scenario_of(item)
    if scenario is not None:
        return scenario.key
    return known_scenario((story or {}).get("scenario"))


def opening(item: dict, chapter: str | None) -> str:
    """Who brings the question to the player, as a line for the model. Decided here and
    not by the model, which would open every question with a person: about one question
    in three gets one of the chapter's people (always the same one for the same text),
    the others start with the situation."""
    cast = BY_KEY[chapter].cast if chapter in BY_KEY else ()
    number = int(text_hash(item), 16)
    if not cast or number % 3:
        return "OPENING: no person. Start with the situation itself."
    name, role = cast[(number // 3) % len(cast)]
    return f"OPENING: {name} ({role}) brings this to you."


_QUESTION_TEXT = '"question": "<the game version of the question - plain text, no HTML>"'
_QUESTION_PARTS = (
    '"parts": [\n'
    '    {"role": "<context|current_state|problem|consequence|goal|constraint|ask>", '
    '"text": "<one sentence of the game version of the question>", '
    '"entity_ids": ["<ids of the ENTITIES this sentence mentions>"]}\n'
    "  ]"
)
_PARTS_RULE = """

THE QUESTION AS PARTS
Write the game version of the question as "parts": its sentences in reading order, \
each one with the role it plays - context (where and what), current_state (what \
happens now), problem (what goes wrong), consequence (its impact), goal (what should \
happen), constraint (a limit that must hold), ask (the question itself). One sentence \
per part, 2 to 6 parts, exactly one "ask" and it comes last, ending with "?". The \
ENTITIES list given with the question is what the question is about: name them with \
the same words, and give each part the ids of the entities it mentions ([] for \
none). The parts follow the PROBLEM given with the question; do not add a cause or a \
fix it does not list. The ask asks exactly what the original question asks, in plain \
words: do not widen or change it (a question asking for "the fix" does not become \
"the cause and the fix")."""
MAX_PARTS = 6


def _build_system_message(
    n_answers: int, chapter: str | None = None, with_parts: bool = False
) -> str:
    # Fill the {{ANSWERS_SCHEMA}} / {{FEEDBACKS_SCHEMA}} placeholders with the
    # correct number of entries so the model knows the exact expected array length.
    letters = [chr(65 + i) for i in range(n_answers)]
    answers_entries = ",\n    ".join(
        f'"<game version of answer {l} - plain words, one or two sentences>"' for l in letters
    )
    feedbacks_entries = ",\n    ".join(
        f'"<feedback for {l} - why it is right or wrong, one or two '
        f'short sentences that keep the reasoning, not just the verdict>"'
        for l in letters
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
        .replace("{{BACKGROUND_RULE}}", background + (_PARTS_RULE if with_parts else ""))
        .replace("{{QUESTION_SCHEMA}}", _QUESTION_PARTS if with_parts else _QUESTION_TEXT)
        .replace("{{ANSWERS_SCHEMA}}", f'"answers": [\n    {answers_entries}\n  ]')
        .replace("{{FEEDBACKS_SCHEMA}}", f'"feedbacks": [\n    {feedbacks_entries}\n  ]')
    )


def _decomposition_block(decomposition: QuestionDecomposition) -> str:
    """What the parts are built on: the entities (with their ids) and the problem."""
    entities = "\n".join(
        f"  {e.id}: {e.name} ({e.type})"
        + (f" - also: {', '.join(e.aliases)}" if e.aliases else "")
        for e in decomposition.entities
    )
    p = decomposition.problem
    problem = "\n".join(
        f"  {name}: {f.text}"
        for name, facts in (
            ("symptom", p.symptoms),
            ("cause", p.root_causes),
            ("consequence", p.consequences),
        )
        for f in facts
    )
    goals = "\n".join(f"  goal: {f.text}" for f in decomposition.context.desired_state)
    return (
        f"ENTITIES:\n{entities or '  (none)'}\n\n"
        f"PROBLEM:\n{problem or '  (none stated)'}"
        + (f"\n{goals}" if goals else "")
        + f"\n\nTHE QUESTION ASKS: {decomposition.question_intent.question_task}"
    )


def _build_user_message(
    item: dict, story: dict | None = None, decomposition: QuestionDecomposition | None = None
) -> str:
    """Render one practice-exam item as the user turn: the question split into its
    scenario paragraph and the question itself, where the question stands in the story,
    and the answer key."""
    prompt = item.get("prompt") or {}
    paragraph, stem = split_scenario(clean_question(prompt.get("question", "")))
    answers = prompt.get("answers", [])
    feedbacks = prompt.get("feedbacks", [])
    correct = item.get("correct_response", [])
    explanation = clean_plain(prompt.get("explanation", ""))
    story = story or {}

    choices_block = "\n".join(
        f"  {chr(65 + i)}. {clean_plain(a)}" for i, a in enumerate(answers)
    )
    correct_labels = ", ".join(l.upper() for l in correct)
    feedbacks_block = "\n".join(
        f"  {chr(65 + i)}. {clean_plain(f)}" for i, f in enumerate(feedbacks)
    )
    stage = next((s for s in STAGES if s.key == story.get("stage")), None)
    parts = [f"Section: {item.get('section', '')}"]
    if stage is not None:
        parts.append(f"STAGE OF THE CHAPTER: {stage.title} - {stage.description}")
    if story.get("beat"):
        parts.append(f"LINE SHOWN BEFORE THE QUESTION:\n{story['beat']}")
    parts.append(opening(item, chapter_key(item, story)))
    if paragraph:
        parts.append(f"SCENARIO:\n{paragraph}")
    parts += [
        f"QUESTION:\n{stem}",
        f"ANSWER CHOICES:\n{choices_block}",
        f"CORRECT ANSWER: {correct_labels}",
        f"PER-CHOICE FEEDBACK:\n{feedbacks_block}",
        f"OFFICIAL EXPLANATION:\n{explanation}",
    ]
    if decomposition is not None:
        parts.append(_decomposition_block(decomposition))
    return "\n\n".join(parts)


_COPY_RATIO = 0.85


def fidelity_errors(item: dict, parsed: dict) -> list[str]:
    """
    What makes a shape-valid reply unusable, as sentences the model can act on (empty
    when it passes). A rewrite may change the words, never what the exam tests:
    - every identifier of an original answer (`exam_terms.lost_identifiers`) is in the
      rewrite of that same answer, and every identifier and number of the question
      itself in the rewritten question (the scenario paragraph is background and may
      be left out);
    - every named concept of the question and its answers (`exam_terms.lost_names`)
      is still named somewhere in the rewritten question and answers;
    - no answer has grown past `MAX_ANSWER_GROWTH`.
    """
    prompt = item.get("prompt") or {}
    originals = [clean_plain(a) for a in prompt.get("answers", [])]
    stem = split_scenario(clean_question(prompt.get("question", "")))[1]

    errors = []
    for i, (original, rewrite) in enumerate(zip(originals, parsed["answers"])):
        letter = chr(65 + i)
        lost = lost_identifiers(original, rewrite)
        if lost:
            errors.append(
                f"answer {letter} must keep these exact terms: {', '.join(lost)}"
            )
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
        errors.append(
            f"the question must keep these numbers as digits: {', '.join(numbers)}"
        )
    return errors


def quality_warnings(item: dict, parsed: dict) -> list[str]:
    """
    Everything a report should show about a shape-valid reply: the hard problems of
    `fidelity_errors` (a stored record has none, unless it predates them) and one soft
    one that never rejects a reply, an answer that is (nearly) a copy of the original.
    Many answers of this bank are already short and exact, so a copy is not an error.
    """
    originals = [clean_plain(a) for a in (item.get("prompt") or {}).get("answers", [])]
    warnings = fidelity_errors(item, parsed)
    for i, (orig, new) in enumerate(zip(originals, parsed["answers"])):
        if SequenceMatcher(None, orig.lower(), new.lower()).ratio() >= _COPY_RATIO:
            warnings.append(f"answer {chr(65 + i)} is not simplified (near copy)")
    return warnings


def parts_errors(parsed: dict, decomposition: QuestionDecomposition) -> list[str]:
    """What is wrong with the `parts` of a reply (see `_PARTS_RULE`); empty when they
    pass. The faults are sent back to the model."""
    raw = parsed.get("parts")
    if not isinstance(raw, list) or not raw:
        return ['"parts" must be a non-empty list of {role, text, entity_ids}']
    try:
        parts = [SemanticPart.model_validate(p) for p in raw]
    except ValueError as exc:
        return [f'"parts" has an invalid item: {str(exc).splitlines()[0]}']
    errors = []
    if len(parts) > MAX_PARTS:
        errors.append(f"at most {MAX_PARTS} parts, got {len(parts)}")
    if any(not p.text.strip() for p in parts):
        errors.append("every part needs a sentence")
    asks = [i for i, p in enumerate(parts) if p.role == "ask"]
    if asks != [len(parts) - 1]:
        errors.append('exactly one part has role "ask", and it is the last one')
    known = {e.id for e in decomposition.entities}
    unknown = sorted({i for p in parts for i in p.entity_ids if i not in known})
    if unknown:
        errors.append(f"parts refer to entity ids that are not listed: {', '.join(unknown)}")
    return errors


def joined(parsed: dict) -> dict:
    """A parts reply with `question` set to its sentences joined (a copy), so the
    checks of the plain text apply unchanged."""
    texts = [
        str(p.get("text", "")).strip() if isinstance(p, dict) else ""
        for p in parsed.get("parts") or []
    ]
    return {**parsed, "question": " ".join(t for t in texts if t)}


def _validate_shape(
    parsed: dict, n_answers: int, correct: list[str] | None = None
) -> None:
    """Require the same structure as the source prompt; raise 502 otherwise.
    With `correct` (the letters of `correct_response`), each feedback must also open
    with "Correct" exactly for the correct options."""

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
    if correct:
        right = {c.lower() for c in correct}
        for i, feedback in enumerate(parsed["feedbacks"]):
            says_correct = feedback.strip().lower().startswith("correct")
            if says_correct != (chr(97 + i) in right):
                raise bad(
                    f"feedback {chr(65 + i)} verdict does not match the correct answer"
                )


async def simplify_item(
    ica: IcaClient,
    item: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    story: dict | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> tuple[SimplifiedPrompt, IcaMeta]:
    """
    Run one exam item (needs `prompt`, `correct_response`, `section`) through the
    ICA chat model and return its validated game text plus call metadata. `story` is
    where the question stands in the story (`scenario`, `stage`, `beat`, as
    `game.domain.story_of` returns them); without it the text is set in the chapter the
    question's own scenario paragraph names, or just in the company.

    A reply that breaks `fidelity_errors` is sent back with what was wrong; when the
    last of `MAX_ATTEMPTS` replies still breaks it the call raises 502. The metadata
    counts the tokens of all attempts.

    With `decomposition` (the question's, see `decompose.py`) the question is written
    as semantic parts that name its entities (`parts_errors`); `question` is then their
    sentences joined, and every check of the plain text runs on that.
    """
    n_answers = len((item.get("prompt") or {}).get("answers", []))
    if n_answers == 0:
        raise HTTPException(
            status_code=422, detail="question has no answers to simplify"
        )
    with_parts = decomposition is not None
    system = _build_system_message(n_answers, chapter_key(item, story), with_parts)
    user = _build_user_message(item, story, decomposition)

    correct = item.get("correct_response") or []

    def validate(parsed: dict) -> list[str]:
        if not with_parts:
            _validate_shape(parsed, n_answers, correct)
            return fidelity_errors(item, parsed)
        errors = parts_errors(parsed, decomposition)
        if errors:
            return errors
        text = joined(parsed)
        _validate_shape(text, n_answers, correct)
        return fidelity_errors(item, text)

    parsed, meta = await ask_validated(
        ica,
        model_id,
        user,
        system=system,
        what="game text",
        validate=validate,
        max_attempts=MAX_ATTEMPTS,
        max_tokens=max_tokens,
        label=item.get("_id"),
    )

    if with_parts:
        parsed = joined(parsed)
    for warning in quality_warnings(item, parsed):
        logger.info("simplify %s: %s", item.get("_id"), warning)
    return (
        SimplifiedPrompt(
            question=parsed["question"],
            answers=parsed["answers"],
            feedbacks=parsed["feedbacks"],
            explanation=parsed["explanation"],
            parts=parsed.get("parts") or [],
        ),
        meta,
    )


def is_current(
    record: dict | None,
    doc: dict,
    story: dict | None = None,
    decomposition_version: str | None = None,
) -> bool:
    """Whether a stored simplified record was written with today's prompt, with the
    story of the question's chapter as it is today, and with the question's current
    decomposition (`decomposition_version`: its `decompose.fingerprint`, None when it
    has none). So a text written before the question was decomposed, or with another
    decomposition, is outdated."""
    return (
        record is not None
        and record.get("prompt_version") == PROMPT_VERSION
        and record.get("frame_version") == frame_version(chapter_key(doc, story))
        and record.get("decomposition_version") == decomposition_version
    )


async def simplify_and_store(
    ica: IcaClient,
    simplified: AsyncCollection,
    doc: dict,
    model_id: IcaModelId,
    max_tokens: int | None = None,
    with_entities: bool = True,
    story: dict | None = None,
    decomposition: QuestionDecomposition | None = None,
) -> SimplifiedQuestion:
    """
    Write the game text of one examination document and upsert the result into
    `examination_simplified`, keyed by `question_id`. An existing record is
    replaced and keeps its `created_at`. `story` is passed on to `simplify_item`.

    Unless `with_entities` is false, a second ICA call extracts the visual entities
    of the new text. That call is best effort: if it fails the record is still
    stored, with empty entities (backfill them later via
    `entities.enrich_and_store`); entities stored earlier survive only when the new
    question and answers are unchanged, and so does the result of the check step.
    After a successful extraction a third call picks the icons, best effort too
    (`icons.assign_and_store` backfills).

    With `decomposition` the text is written as semantic parts, and the entities are
    derived from the decomposition instead of being asked for.
    """
    prompt, meta = await simplify_item(ica, doc, model_id, max_tokens, story, decomposition)

    entities, entities_meta, icons_meta = PromptEntities(), None, None
    entities_source = None
    if with_entities and decomposition is not None:
        entities = derive_entities(decomposition, doc, prompt)
        entities_meta = IcaMeta(model_id=model_id, finish_reason="derived", total_tokens=0)
        entities_source = "decomposition"
    elif with_entities:
        try:
            entities, entities_meta = await extract_entities(
                ica, doc, prompt, model_id, max_tokens
            )
            entities_source = "ica"
        except Exception as exc:  # best effort: never lose the simplified text
            logger.warning(
                "entity extraction failed for %s: %s",
                doc.get("_id"),
                getattr(exc, "detail", exc),
            )
    if entities_meta is not None:
        try:
            entities, icons_meta = await pick_icons(
                ica, doc.get("section", ""), prompt, entities, model_id, max_tokens
            )
        except Exception as exc:  # best effort: the entities are stored without icons
            logger.warning(
                "icon picking failed for %s: %s", doc.get("_id"), getattr(exc, "detail", exc)
            )

    question_id = str(doc["_id"])
    existing = await simplified.find_one({"question_id": question_id})
    old = (existing or {}).get("prompt") or {}
    # What was derived from the stored text still fits only while the text is the same;
    # otherwise entities would point at the wrong words and a check at another wording.
    same_text = (
        existing is not None
        and old.get("question") == prompt.question
        and old.get("answers") == prompt.answers
    )
    # Derived entities point into their decomposition (`ref`), so they also need the
    # same one.
    fingerprint = decompose.fingerprint(decomposition) if decomposition else None
    keep_entities = (
        entities_meta is None
        and same_text
        and existing.get("entities_ica") is not None
        and (
            existing.get("entities_source") != "decomposition"
            or existing.get("decomposition_version") == fingerprint
        )
    )

    now = datetime.now(timezone.utc)
    fields = {
        "section": doc.get("section", ""),
        "assessment_type": doc.get("assessment_type", ""),
        "source": doc.get("source", ""),
        "correct_response": doc.get("correct_response", []),
        "related_lectures": doc.get("related_lectures", []),
        "prompt": prompt.model_dump(),
        "ica": meta.model_dump(),
        "prompt_version": PROMPT_VERSION,
        "frame_version": frame_version(chapter_key(doc, story)),
        "decomposition_version": fingerprint,
        "updated_at": now,
    }
    if not keep_entities:
        fields["entities"] = entities.model_dump()
        fields["entities_ica"] = entities_meta.model_dump() if entities_meta else None
        fields["icons_ica"] = icons_meta.model_dump() if icons_meta else None
        fields["entities_version"] = entities_version if entities_meta else None
        fields["icons_version"] = icons_version if icons_meta else None
        fields["entities_source"] = entities_source
    if not same_text:
        fields["check"] = None
    stored = await simplified.find_one_and_update(
        {"question_id": question_id},
        {"$set": fields, "$setOnInsert": {"created_at": now}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return SimplifiedQuestion(**stored)
