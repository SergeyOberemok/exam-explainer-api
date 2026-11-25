"""Read-only report on how the question bank is covered by the game's story.

Per chapter of the scenario catalog: how many playable questions each stage has (a
chapter's stage is one stop on the game's map). Then what needs work: questions whose
scenario no chapter claims, questions without a story record or with one written for an
older text, catalog or prompt, and stages with too few questions. Last, which learning
objectives each chapter exercises.

Usage: uv run python scripts/story_coverage.py [DATABASE_URL] [--min N] [--check]
  --min N   a chapter's stage needs at least N questions (default 2)
  --check   exit with status 1 when something needs work (for a content release gate)
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys
from collections import Counter

from pymongo import MongoClient

from claude_exam.api.game.domain import chapter_of, is_playable, objectives_of, usable_story
from claude_exam.api.scenario_catalog import (
    SCENARIOS,
    STAGES,
    unknown_paragraph,
)
from claude_exam.api import pipeline_keys
from claude_exam.api.pipeline_keys import Records


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def main() -> int:
    argv = sys.argv[1:]
    minimum = int(argv[argv.index("--min") + 1]) if "--min" in argv else 2
    skip = {argv.index("--min") + 1} if "--min" in argv else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    db = MongoClient(database_url(args)).get_default_database()

    questions = {str(d["_id"]): d for d in db["examination"].find() if is_playable(d)}
    records = {r["question_id"]: r for r in db["question_story"].find()}
    tags = {t["question_id"]: t for t in db["question_objectives"].find()}

    nodes: Counter = Counter()
    origins: Counter = Counter()
    objectives: dict[str, Counter] = {s.key: Counter() for s in SCENARIOS}
    unclaimed: Counter = Counter()
    no_story: list[str] = []
    outdated: list[str] = []
    picked = 0

    for question_id, doc in questions.items():
        paragraph = unknown_paragraph(doc)
        if paragraph:
            unclaimed[" ".join(paragraph.split())[:70]] += 1
            continue
        # The chapter the text names, or the one the model picked for a question
        # without a scenario paragraph (it has none until its story is written).
        chapter = chapter_of(doc, records.get(question_id))
        record = usable_story(doc, records.get(question_id))
        if not chapter or record is None:
            no_story.append(question_id)
            continue
        keys = objectives_of(tags.get(question_id))
        if keys:
            objectives[chapter][keys[0]] += 1
        nodes[(chapter, record["stage"])] += 1
        origins[record.get("origin")] += 1
        picked += record.get("scenario_origin") == "model"
        if not pipeline_keys.is_current("story", doc, Records(story=record)):
            outdated.append(question_id)

    print(
        f"{len(questions)} playable "
        f"questions, {sum(nodes.values())} with a story "
        f"({origins['seed']} seed, {origins['ica']} written by the model; "
        f"{picked} in a chapter the model picked)\n"
    )
    print(f"  {'chapter':<12}" + "".join(f"{s.key:>10}" for s in STAGES) + f"{'total':>8}")
    thin: list[str] = []
    for scenario in SCENARIOS:
        counts = [nodes[(scenario.key, s.key)] for s in STAGES]
        thin += [f"{scenario.key}/{s.key}" for s, c in zip(STAGES, counts) if c < minimum]
        cells = "".join(f"{c:>9}{'<' if c < minimum else ' '}" for c in counts)
        print(f"  {scenario.key:<12}{cells}{sum(counts):>8}")

    print("\nNEEDS WORK")
    for name, count in sorted(unclaimed.items()):
        print(f"  scenario no chapter claims: {name!r} ({count} questions)")
    print(f"  stages with fewer than {minimum} questions: {len(thin)}"
          + (f" ({', '.join(thin)})" if thin else ""))
    print(f"  questions without a usable story record: {len(no_story)}")
    for question_id in no_story[:20]:
        print(f"    {question_id}")
    print(f"  written with an older catalog or prompt: {len(outdated)}")

    print("\nMAIN OBJECTIVES PER CHAPTER")
    for scenario in SCENARIOS:
        listed = ", ".join(f"{k} {c}" for k, c in objectives[scenario.key].most_common())
        print(f"  {scenario.key}: {listed or '(no tagged questions)'}")

    needs_work = bool(unclaimed or thin or no_story or outdated)
    return 1 if needs_work and "--check" in argv else 0


if __name__ == "__main__":
    sys.exit(main())
