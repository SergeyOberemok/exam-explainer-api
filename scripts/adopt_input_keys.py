"""Stamp input keys (`api/step_key.py`) on the step results the old version fields call
current, so the first backfill after the keys came redoes only what is really outdated
instead of the whole bank. Run it once after deploying the keys; running it again
stamps nothing new.

What the version fields call current but whose inputs had drifted (a story written
before its chapter's intro changed, which the old `chapter_version` did not cover) is
adopted as current too, as it was before.

Usage: uv run python scripts/adopt_input_keys.py [DATABASE_URL] [--dry-run]
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import asyncio
import os
import sys
from collections import Counter

from pymongo import AsyncMongoClient

from claude_exam.api.batch import STAGES, Stores, adopt_input_keys


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


async def main() -> int:
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    args = [a for a in argv if not a.startswith("--")]
    client: AsyncMongoClient = AsyncMongoClient(database_url(args))
    try:
        db = client.get_default_database()
        stores = Stores(
            db["examination"],
            db["examination_simplified"],
            db["question_objectives"],
            db["question_story"],
            db["question_decomposition"],
        )
        counts: Counter = Counter()
        total = 0
        async for doc in db["examination"].find():
            total += 1
            counts.update(await adopt_input_keys(stores, doc, dry_run=dry_run))
    finally:
        await client.close()

    verb = "would stamp" if dry_run else "stamped"
    print(f"{total} questions; {verb} input keys on:")
    for stage in STAGES["full"]:
        print(f"  {stage:<12}{counts[stage]:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
