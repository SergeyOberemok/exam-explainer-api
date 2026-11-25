"""Read-only report on learning-objective coverage of the question bank.

Per objective of the catalog: how many playable questions have it as their main
objective, split by difficulty, and how many more touch it. Then what needs work:
sections without objectives, objectives with too few questions, questions that are
untagged or were tagged with an older catalog or prompt, and the objectives wrong
answers are most often confused with.

Usage: uv run python scripts/objective_coverage.py [DATABASE_URL] [--min N] [--check]
  --min N   an objective needs at least N main questions (default 2)
  --check   exit with status 1 when something needs work (for a content release gate)
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys
from collections import Counter

from pymongo import MongoClient

from claude_exam.api.game.domain import is_playable
from claude_exam.api.objective_catalog import (
    BY_KEY,
    OBJECTIVES,
    for_section,
    normalize_section,
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
    tags = {t["question_id"]: t for t in db["question_objectives"].find()}

    main_by_key: dict[str, Counter] = {o.key: Counter() for o in OBJECTIVES}
    touched: Counter = Counter()
    confused: Counter = Counter()
    untagged: list[str] = []
    outdated: list[str] = []
    no_catalog: Counter = Counter()
    elsewhere = 0

    for question_id, doc in questions.items():
        record = tags.get(question_id)
        keys = [k for k in (record or {}).get("objectives") or [] if k in BY_KEY]
        if not for_section(doc.get("section")):
            no_catalog[normalize_section(doc.get("section"))] += 1
        if not keys:
            untagged.append(question_id)
            continue
        if not pipeline_keys.is_current("objectives", doc, Records(objectives=record)):
            outdated.append(question_id)
        main_by_key[keys[0]][record.get("difficulty")] += 1
        elsewhere += BY_KEY[keys[0]].section != normalize_section(doc.get("section"))
        for key in keys[1:]:
            touched[key] += 1
        for item in record.get("distractors") or []:
            if item and item.get("confused_with") in BY_KEY:
                confused[(keys[0], item["confused_with"])] += 1

    print(f"{len(OBJECTIVES)} objectives, "
          f"{len(questions)} playable questions, {len(questions) - len(untagged)} tagged\n")
    section = None
    thin: list[str] = []
    for objective in OBJECTIVES:
        if objective.section != section:
            section = objective.section
            print(f"{section}")
            print(f"  {'objective':<28}{'main':>5}{'easy':>6}{'mid':>5}{'hard':>6}{'also':>6}")
        by_difficulty = main_by_key[objective.key]
        total = sum(by_difficulty.values())
        if total < minimum:
            thin.append(objective.key)
        print(
            f"  {objective.key:<28}{total:>5}{by_difficulty[1]:>6}{by_difficulty[2]:>5}"
            f"{by_difficulty[3]:>6}{touched[objective.key]:>6}{'   <' if total < minimum else ''}"
        )

    print("\nNEEDS WORK")
    for name, count in sorted(no_catalog.items()):
        print(f"  section without objectives: {name!r} ({count} questions)")
    print(f"  objectives with fewer than {minimum} main questions: {len(thin)}"
          + (f" ({', '.join(thin)})" if thin else ""))
    print(f"  untagged playable questions: {len(untagged)}")
    for question_id in untagged[:20]:
        print(f"    {question_id}  {normalize_section(questions[question_id].get('section'))}")
    print(f"  tagged with an older catalog or prompt: {len(outdated)}")
    print(f"  (for information) main objective outside the question's section: {elsewhere}")

    if confused:
        print("\nMOST CONFUSED (question's objective <- what a wrong answer belongs to)")
        for (key, other), count in confused.most_common(10):
            print(f"  {count:>3}  {key} <- {other}")

    needs_work = bool(no_catalog or thin or untagged or outdated)
    return 1 if needs_work and "--check" in argv else 0


if __name__ == "__main__":
    sys.exit(main())
