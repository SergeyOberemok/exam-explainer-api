"""One page on the state of the ingest pipeline: what every stored question has, what
is missing or outdated, and what the game cannot serve.

Read-only. Run it after an import job (or any backfill) to see whether there is
anything left to do, and what kind of thing it is:
- a step that is missing or was written with an older prompt or catalog: run
  `POST /examination/pipeline/backfill`;
- a scenario paragraph no chapter claims, or a section without objectives: extend the
  catalog (`propose_scenarios.py`, `propose_objectives.py`), then run the backfill;
- a game text the check found changed: read it (`--show`), then write it again with
  `POST /questions/{id}/simplify?force=true` and check it with
  `POST /examination/{id}/check`. Until then the game serves the original only.

Usage: uv run python scripts/pipeline_report.py [DATABASE_URL] [--source FILE] [--show] [--check]
  --source FILE  only questions imported from this file
  --show         print the question and the game text of every flagged question
  --check        exit with status 1 when something needs work
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys
from collections import Counter

from pymongo import MongoClient

from claude_exam.api import check, simplify
from claude_exam.api.batch import STAGES
from claude_exam.api.game.domain import (
    CHECK_CHANGED,
    CHECK_HARD,
    answers_of,
    correct_indices,
    is_playable,
    simplified_matches,
    story_of,
)
from claude_exam.api.objective_catalog import CATALOG_VERSION as OBJECTIVE_CATALOG
from claude_exam.api.objective_catalog import for_section, normalize_section
from claude_exam.api.objectives import PROMPT_VERSION as OBJECTIVE_PROMPT
from claude_exam.api.scenario_catalog import CHAPTER_VERSIONS, unknown_paragraph
from claude_exam.api.simplify import quality_warnings
from claude_exam.api.story import PROMPT_VERSION as STORY_PROMPT


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def stage_states(doc: dict, simplified: dict | None, tags: dict | None, story: dict | None) -> dict:
    """For each step of the full pipeline: "current", "outdated" or "missing". The same
    rules as `batch.is_current`, on records already read."""

    def state(record, current: bool) -> str:
        return "missing" if record is None else "current" if current else "outdated"

    place = story_of(doc, story)
    return {
        "objectives": state(
            tags,
            bool(tags)
            and (tags.get("catalog_version"), tags.get("prompt_version"))
            == (OBJECTIVE_CATALOG, OBJECTIVE_PROMPT),
        ),
        "story": state(
            story,
            bool(story)
            and (
                story.get("origin") == "seed"
                or (
                    story.get("catalog_version") in CHAPTER_VERSIONS
                    and story.get("prompt_version") == STORY_PROMPT
                )
            ),
        ),
        "simplify": state(simplified, simplify.is_current(simplified, doc, place)),
        "entities": state(
            simplified and simplified.get("entities_ica"),
            bool(simplified and simplified.get("entities_ica")),
        ),
        "icons": state(
            simplified and simplified.get("icons_ica"),
            bool(simplified and simplified.get("icons_ica")),
        ),
        "check": state(simplified and simplified.get("check"), check.is_current(simplified)),
    }


def show(doc: dict, simplified: dict) -> None:
    correct = correct_indices(doc)
    for title, source in (("ORIGINAL", doc), ("GAME TEXT", simplified)):
        print(f"      {title}: {' '.join((source['prompt'].get('question') or '').split())}")
        for i, answer in enumerate(answers_of(source)):
            mark = "*" if i in correct else " "
            print(f"       {mark}{chr(65 + i)}. {answer}")


def main() -> int:
    argv = sys.argv[1:]
    source = argv[argv.index("--source") + 1] if "--source" in argv else None
    skip = {argv.index("--source") + 1} if "--source" in argv else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    db = MongoClient(database_url(args)).get_default_database()

    docs = list(db["examination"].find({"source": source} if source else {}))
    simplified = {r["question_id"]: r for r in db["examination_simplified"].find()}
    tags = {r["question_id"]: r for r in db["question_objectives"].find()}
    stories = {r["question_id"]: r for r in db["question_story"].find()}

    states = {stage: Counter() for stage in STAGES["full"]}
    verdicts: Counter = Counter()
    difficulty: Counter = Counter()
    sections: Counter = Counter()
    paragraphs: Counter = Counter()
    unplayable = 0
    changed: list[str] = []
    hard: list[str] = []
    lossy: list[tuple[str, list[str]]] = []
    unusable: list[str] = []

    for doc in docs:
        question_id = str(doc["_id"])
        text = simplified.get(question_id)
        for stage, state in stage_states(
            doc, text, tags.get(question_id), stories.get(question_id)
        ).items():
            states[stage][state] += 1
        unplayable += not is_playable(doc)
        if not for_section(doc.get("section") or ""):
            sections[normalize_section(doc.get("section")) or "(no section)"] += 1
        paragraph = unknown_paragraph(doc)
        if paragraph:
            paragraphs[" ".join(paragraph.split())[:70]] += 1
        difficulty[(tags.get(question_id) or {}).get("difficulty", 0)] += 1
        if text is None:
            continue
        verdict = (text.get("check") or {}).get("verdict")
        verdicts[verdict or "not checked"] += 1
        if verdict == CHECK_CHANGED:
            changed.append(question_id)
        elif verdict == CHECK_HARD:
            hard.append(question_id)
        if is_playable(doc) and not simplified_matches(doc, text) and verdict != CHECK_CHANGED:
            unusable.append(question_id)
        if text.get("prompt_version") == simplify.PROMPT_VERSION:
            hard_problems = [w for w in quality_warnings(doc, text["prompt"]) if "near copy" not in w]
            if hard_problems:
                lossy.append((question_id, hard_problems))

    print(f"{len(docs)} questions" + (f" from {source}" if source else "") + f", {unplayable} the game cannot play\n")
    print(f"  {'step':<12}{'current':>9}{'outdated':>10}{'missing':>9}")
    for stage in STAGES["full"]:
        c = states[stage]
        print(f"  {stage:<12}{c['current']:>9}{c['outdated']:>10}{c['missing']:>9}")
    print(
        "\n  check verdicts: "
        + (", ".join(f"{name} {count}" for name, count in verdicts.most_common()) or "(none)")
    )
    print("  difficulty: " + ", ".join(f"{d or 'untagged'}: {difficulty[d]}" for d in (1, 2, 3, 0)))

    todo = sum(c["outdated"] + c["missing"] for c in states.values())
    print("\nNEEDS WORK")
    print(f"  steps to run (POST /examination/pipeline/backfill): {todo}")
    for name, count in sections.most_common():
        print(f"  section without objectives: {name!r} ({count} questions)")
    for name, count in paragraphs.most_common():
        print(f"  scenario no chapter claims: {name!r} ({count} questions)")
    print(f"  game texts the check found changed (not served): {len(changed)}")
    for question_id in changed:
        print(f"    {question_id}")
        if "--show" in argv:
            show(next(d for d in docs if str(d["_id"]) == question_id), simplified[question_id])
    print(f"  game texts with another answer count or key (not served): {len(unusable)}")
    for question_id in unusable:
        print(f"    {question_id}")
    print(f"  current game texts that break a fidelity rule: {len(lossy)}")
    for question_id, problems in lossy:
        print(f"    {question_id}: {'; '.join(problems)}")

    print("\nFOR INFORMATION")
    print(f"  questions the model missed in both wordings (hard): {len(hard)}")
    for question_id in hard:
        print(f"    {question_id}")

    needs_work = bool(todo or sections or paragraphs or changed or unusable or lossy)
    return 1 if needs_work and "--check" in argv else 0


if __name__ == "__main__":
    sys.exit(main())
