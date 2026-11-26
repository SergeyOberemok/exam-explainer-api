"""One page on the state of the ingest pipeline: what every stored question has, what
is missing or outdated, and what the game cannot serve.

The game (`/v2/game`) serves a question's composed document (`game_question`, the last
step, `compose.py`); a question that cannot be composed is listed with what it lacks.

Read-only. Run it after an import job (or any backfill) to see whether there is
anything left to do, and what kind of thing it is:
- a step that is missing or was written with an older prompt or catalog: run
  `POST /examination/pipeline/backfill`;
- a scenario paragraph no chapter claims, or a section without objectives: extend the
  catalog (`propose_scenarios.py`, `propose_objectives.py`), then run the backfill;
- a question that cannot be composed: the steps it lacks are run by the backfill; a
  resolution "without a consequence per answer" was written before the resolve step
  wrote consequences, and the backfill writes it again (its prompt changed);
- a pre-answer flow that may point at the correct answer (`flow.leak_warning`): look at
  it with `scripts/run_pipeline.py ID --stages flow --dry-run --mermaid` and redo it
  (`POST /examination/{id}/flow`) or tighten the flow prompt.

Usage: uv run python scripts/pipeline_report.py [DATABASE_URL] [--source FILE] [--check]
  --source FILE  only questions imported from this file
  --check        exit with status 1 when something needs work
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys
from collections import Counter

from pymongo import MongoClient

from claude_exam.api import compose, decompose, hold, pipeline_keys
from claude_exam.api.batch import STAGES
from claude_exam.api.pipeline_keys import Records
from claude_exam.api.game.domain import is_playable
from claude_exam.api.objective_catalog import for_section, normalize_section
from claude_exam.api.scenario_catalog import unknown_paragraph
from claude_exam.api.flow import leak_warning
from claude_exam.api.schemas import DecisionFlow


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def stage_states(
    doc: dict,
    tags: dict | None,
    story: dict | None,
    decomposition: dict | None = None,
    game_question: dict | None = None,
) -> dict:
    """For each step of the full pipeline: "current", "outdated" or "missing", by the
    same rule as the job (`pipeline_keys.state`), on records already read."""
    records = Records(
        objectives=tags,
        story=story,
        decomposition=decomposition,
        game_question=game_question,
    )
    return {stage: pipeline_keys.state(stage, doc, records) for stage in REPORTED}


REPORTED = STAGES["full"]


def main() -> int:
    argv = sys.argv[1:]
    source = argv[argv.index("--source") + 1] if "--source" in argv else None
    skip = {argv.index("--source") + 1} if "--source" in argv else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    db = MongoClient(database_url(args)).get_default_database()

    docs = list(db["examination"].find({"source": source} if source else {}))
    tags = {r["question_id"]: r for r in db["question_objectives"].find()}
    stories = {r["question_id"]: r for r in db["question_story"].find()}
    decompositions = {r["question_id"]: r for r in db["question_decomposition"].find()}
    games = {r["question_id"]: r for r in db["game_question"].find()}

    states = {stage: Counter() for stage in REPORTED}
    difficulty: Counter = Counter()
    sections: Counter = Counter()
    paragraphs: Counter = Counter()
    unplayable = 0
    held = 0
    # Playable, but without a game document (not served by /v2), by what is missing.
    uncomposed: Counter = Counter()
    not_composed = 0
    leaks: list[tuple[str, str]] = []
    views: Counter = Counter()

    for doc in docs:
        question_id = str(doc["_id"])
        record = decompositions.get(question_id)
        question_states = stage_states(
            doc, tags.get(question_id), stories.get(question_id), record, games.get(question_id)
        )
        for stage, state in question_states.items():
            states[stage][state] += 1
        if decompose.matches(record, doc) and decompose.has_flow(record):
            try:
                flow = DecisionFlow.model_validate(record["flow"])
            except ValueError:
                flow = None
            if flow is not None:
                views[flow.visualization.primary] += 1
                warning = leak_warning(doc, flow)
                if warning:
                    leaks.append((question_id, warning))
        held += hold.is_held(doc)
        unplayable += not is_playable(doc) and not hold.is_held(doc)
        if is_playable(doc) and question_id not in games:
            not_composed += 1
            for part in compose.missing(doc, tags.get(question_id), stories.get(question_id), record):
                uncomposed[part] += 1
        if not for_section(doc.get("section") or ""):
            sections[normalize_section(doc.get("section")) or "(no section)"] += 1
        paragraph = unknown_paragraph(doc)
        if paragraph:
            paragraphs[" ".join(paragraph.split())[:70]] += 1
        difficulty[(tags.get(question_id) or {}).get("difficulty", 0)] += 1

    print(f"{len(docs)} questions" + (f" from {source}" if source else "") + f", {unplayable} the game cannot play"
          + (f", {held} held for manual processing (not played yet)" if held else "")
          + (f", {not_composed} without a game document (not served)" if not_composed else "") + "\n")
    print(f"  {'step':<12}{'current':>9}{'outdated':>10}{'missing':>9}")
    for stage in REPORTED:
        c = states[stage]
        print(f"  {stage:<12}{c['current']:>9}{c['outdated']:>10}{c['missing']:>9}")
    print("\n  difficulty: " + ", ".join(f"{d or 'untagged'}: {difficulty[d]}" for d in (1, 2, 3, 0)))
    print("  flow views: " + (", ".join(f"{v} {c}" for v, c in views.most_common()) or "(none)"))

    todo = sum(states[s]["outdated"] + states[s]["missing"] for s in STAGES["full"])
    print("\nNEEDS WORK")
    print(f"  game steps to run (POST /examination/pipeline/backfill): {todo}")
    for part, count in uncomposed.most_common():
        print(f"  not composed for the game, missing {part}: {count}")
    for name, count in sections.most_common():
        print(f"  section without objectives: {name!r} ({count} questions)")
    for name, count in paragraphs.most_common():
        print(f"  scenario no chapter claims: {name!r} ({count} questions)")
    print(f"  pre-answer flows that may point at the correct answer: {len(leaks)}")
    for question_id, warning in leaks:
        print(f"    {question_id}: {warning}")

    needs_work = bool(todo or sections or paragraphs or leaks)
    return 1 if needs_work and "--check" in argv else 0


if __name__ == "__main__":
    sys.exit(main())
