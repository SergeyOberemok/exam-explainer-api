"""Make the database fresh: empty every collection except the original questions
(`examination`), so the pipeline starts again from the imported questions alone.

Emptied (documents deleted, indexes kept, so a running API needs no restart): every
collection of the database but `examination` and those named with `--keep`; today that is
the pipeline's records (`question_objectives`, `question_story`, `question_decomposition`,
`game_question`), the jobs (`simplification_jobs`) and the game rounds (`game_rounds`).

The questions themselves are not touched (their `text_hash` stays, and so does a
manual-mode hold unless `--unhold`). The story seed lives in `question_story`, so load it
again afterwards: `uv run python scripts/load_story_seed.py`. Then rebuild everything with
`POST /examination/pipeline/backfill`.

Without `--yes` it only prints what it would delete. It refuses while a job is running
(`--force` to go on anyway, e.g. for a job left "running" by a killed server).

Usage: uv run python scripts/reset_db.py [DATABASE_URL] [--yes] [--force] [--unhold] [--keep NAME ...]
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys

from pymongo import MongoClient

from claude_exam.api import hold

KEPT = "examination"


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def parse(argv: list[str]) -> tuple[list[str], set[str]]:
    """Positional arguments, and the collections named after `--keep`."""
    args: list[str] = []
    keep: set[str] = set()
    taking = False
    for arg in argv:
        if arg == "--keep":
            taking = True
        elif arg.startswith("--"):
            taking = False
        elif taking:
            keep.add(arg)
        else:
            args.append(arg)
    return args, keep


def main() -> int:
    argv = sys.argv[1:]
    args, keep = parse(argv)
    db = MongoClient(database_url(args)).get_default_database()
    kept = {KEPT, *keep}

    names = sorted(n for n in db.list_collection_names() if not n.startswith("system.") and n not in kept)
    print(f"database {db.name!r}: {db[KEPT].count_documents({})} questions in {KEPT!r} are kept")
    for name in sorted(keep):
        print(f"  kept (--keep): {name}")
    for name in names:
        print(f"  {'empty' if '--yes' in argv else 'would empty'}: {name} ({db[name].count_documents({})} documents)")
    held = db[KEPT].count_documents({hold.FIELD: {"$exists": True}})
    if "--unhold" in argv:
        print(f"  {'release' if '--yes' in argv else 'would release'}: {held} questions held for manual processing")
    elif held:
        print(f"  {held} questions stay held for manual processing (--unhold releases them)")

    if "simplification_jobs" in names and "--force" not in argv:
        running = db["simplification_jobs"].count_documents({"status": "running"})
        if running:
            print(f"\n{running} job(s) marked running: stop them first, or pass --force")
            return 1

    if "--yes" not in argv:
        print("\nnothing changed (pass --yes to do it)")
        return 0
    for name in names:
        db[name].delete_many({})
    if "--unhold" in argv:
        db[KEPT].update_many({hold.FIELD: {"$exists": True}}, {"$unset": {hold.FIELD: ""}})
    print("\ndone. Next: scripts/load_story_seed.py, then POST /examination/pipeline/backfill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
