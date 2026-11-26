"""Load the hand-written story records of `data/story_seed.json` into `question_story`.

Each seed item names a question by `hash` (`scenario_catalog.text_hash`, so it survives
a new import under new ids) and gives its stage and story lines; the order of the items
inside a chapter's stage is the play order. Items are checked with the same parser as
the model's replies. A question's record is replaced, whoever wrote it; questions
without a seed item are not touched (the story backfill writes those).

Usage: uv run python scripts/load_story_seed.py [DATABASE_URL] [--dry-run]
Without DATABASE_URL the one from the environment or `.env` is used. Exits with
status 1 when an item is invalid or matches no question; nothing is written then.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from pymongo import MongoClient

from claude_exam.api.decompose import read_decomposition
from claude_exam.api.game.domain import answers_of
from claude_exam.api.leaks import answer_only, leak
from claude_exam.api.scenario_catalog import scenario_of, text_hash
from claude_exam.api.story import SEED_LINE_WORDS, parse_story, record_fields, upsert_update

SEED = Path(__file__).resolve().parents[1] / "data" / "story_seed.json"


def seed_updates(
    items: list[dict], docs: list[dict], decompositions: dict[str, dict] | None = None
) -> tuple[list[tuple[str, dict]], list[str]]:
    """`(question_id, record fields)` for every seed item, and what is wrong with the
    seed (empty when it can be loaded). A beat is also tested as the beat step tests its
    own (`leaks.leak`, with the question's decomposition from `decompositions`, by
    question id, when it has a current one)."""
    decompositions = decompositions or {}
    by_hash = {text_hash(d): d for d in docs}
    updates: list[tuple[str, dict]] = []
    problems: list[str] = []
    seen: set[str] = set()
    position: dict[tuple, int] = {}
    for item in items:
        name = item.get("hash")
        doc = by_hash.get(name)
        if doc is None:
            problems.append(f"{name}: no question has this hash")
            continue
        if name in seen:
            problems.append(f"{name}: listed twice")
            continue
        seen.add(name)
        scenario = scenario_of(doc)
        if scenario is None or scenario.key != item.get("scenario"):
            problems.append(
                f"{name}: the question's chapter is {scenario.key if scenario else None!r}, "
                f"the seed says {item.get('scenario')!r}"
            )
            continue
        try:
            fields = parse_story(item, answers_of(doc), max_words=SEED_LINE_WORDS, with_beat=True)
        except HTTPException as exc:
            problems.append(f"{name}: {exc.detail}")
            continue
        decomposition = read_decomposition(decompositions.get(str(doc["_id"])), doc)
        reason = leak(fields["beat"], answer_only(doc, decomposition).narrow())
        if reason:
            problems.append(f"{name}: the beat {reason}")
            continue
        # A beat the model wrote earlier leaves its metadata behind otherwise.
        fields = {**fields, "beat_ica": None, "beat_input_key": None}
        node = (scenario.key, fields["stage"])
        position[node] = position.get(node, 0) + 1
        updates.append((str(doc["_id"]), record_fields(doc, fields, "seed", position[node])))
    return updates, problems


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def main() -> int:
    argv = sys.argv[1:]
    db = MongoClient(database_url([a for a in argv if not a.startswith("--")])).get_default_database()
    items = json.loads(SEED.read_text(encoding="utf-8"))["items"]
    docs = list(db["examination"].find())
    decompositions = {d["question_id"]: d for d in db["question_decomposition"].find()}
    updates, problems = seed_updates(items, docs, decompositions)
    for problem in problems:
        print(f"  ! {problem}")
    if problems:
        print(f"{len(problems)} problem(s); nothing written")
        return 1
    if "--dry-run" not in argv:
        now = datetime.now(timezone.utc)
        for question_id, fields in updates:
            db["question_story"].update_one(
                {"question_id": question_id}, upsert_update(fields, None, now), upsert=True
            )
    print(
        f"{len(updates)} of {len(docs)} questions have a seed record"
        + (" (dry run, nothing written)" if "--dry-run" in argv else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
