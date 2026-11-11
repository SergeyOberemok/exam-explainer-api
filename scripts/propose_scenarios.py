"""List the scenario paragraphs no chapter claims, with a chapter draft for each.

A helper for editing `src/claude_exam/api/scenario_catalog.py` by hand, like
`propose_objectives.py` is for the objective catalog. Questions whose `SCENARIO:`
paragraph is in no chapter's `marker` get no story (the import's dry run counts them).
This prints each such paragraph in full, how many questions carry it, a few of their
questions, and a `ScenarioDef(...)` skeleton to fill in. It calls no model and changes
nothing: a chapter is what a person approves.

After adding the chapter: `POST /examination/pipeline/backfill` writes the story and the
game text of its questions (nothing is redone for the other chapters).

Usage: uv run python scripts/propose_scenarios.py [DATABASE_URL] [--samples N]
Without DATABASE_URL the one from the environment or `.env` is used. Exits with status
1 when there is something to add.
"""

import os
import sys
from collections import defaultdict

from pymongo import MongoClient

from claude_exam.api.scenario_catalog import SCENARIOS, split_scenario, unknown_paragraph


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def unclaimed(docs: list[dict]) -> dict[str, list[dict]]:
    """Questions per scenario paragraph no chapter claims (spacing ignored), the
    paragraph with the most questions first."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        paragraph = " ".join(unknown_paragraph(doc).split())
        if paragraph:
            groups[paragraph].append(doc)
    return dict(sorted(groups.items(), key=lambda item: -len(item[1])))


def main() -> int:
    argv = sys.argv[1:]
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 3
    skip = {argv.index("--samples") + 1} if "--samples" in argv else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    db = MongoClient(database_url(args)).get_default_database()

    groups = unclaimed(list(db["examination"].find()))
    if not groups:
        print(f"Every scenario paragraph belongs to one of the {len(SCENARIOS)} chapters.")
        return 0

    for number, (paragraph, docs) in enumerate(groups.items(), 1):
        print(f"--- {number}. {len(docs)} questions ---\n{paragraph}\n")
        for doc in docs[:samples]:
            stem = " ".join(split_scenario(doc["prompt"].get("question"))[1].split())
            print(f"  e.g. {stem[:200]}")
        print(
            "\n  Add to SCENARIOS (the marker: lower-case words of this paragraph that "
            "are in no other):\n"
            "    ScenarioDef(\n"
            '        "<key>", "<Title>", "<District on the map>",\n'
            '        "<intro: which team wants what, in the company\'s words>",\n'
            '        "<outro: what is better when the chapter is done>",\n'
            '        "<marker>",\n'
            '        (("<Name>", "<role>"), ("<Name>", "<role>")),\n'
            "    ),\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
