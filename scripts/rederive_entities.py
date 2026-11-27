"""Rebuild the stored visual entities of decomposed questions from their decomposition.

The entities are built in code when a decomposition is stored (`entities.derive_entities`)
and have no input key, so a change to that derivation reaches stored questions only
through this script (or by decomposing them again, which asks the model). Nothing is
asked of ICA: the decomposition (`semantic`) is read as it is, and the icons already
picked are kept: by id from `icons.entities`, else from the old entity with the same
`ref`, else the kind's fallback (the icons step can pick again for those). The question's
and the answers' own icons stay. The input keys and `updated_at` are left alone, so the
flow, resolution and icons count as current; the game document does not read these
entities (it takes the cast from the decomposition), so nothing needs composing again.

A record whose decomposition was written for an older text of the question, or is
malformed, is skipped. The write is conditional on `updated_at` (a concurrent rewrite of
the decomposition wins).

Usage: uv run python scripts/rederive_entities.py [DATABASE_URL] [--dry-run]
Without DATABASE_URL the one from the environment or `.env` is used.
"""

import os
import sys

from bson import ObjectId
from pymongo import MongoClient

from claude_exam.api.decompose import read_decomposition
from claude_exam.api.entities import derive_entities
from claude_exam.api.icon_catalog import KIND_FALLBACK, known
from claude_exam.api.question_view import key_free
from claude_exam.api.schemas import PromptEntities, VisualEntity


def database_url(args: list[str]) -> str:
    if args:
        return args[0]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from claude_exam.api.config import get_settings

    return get_settings().database_url


def rederived(record: dict, doc: dict) -> dict | None:
    """The record's `entities` rebuilt from its decomposition, with the icons it had; None
    when the decomposition is not current for the question."""
    decomposition = read_decomposition(record, doc)
    if decomposition is None:
        return None
    new = derive_entities(decomposition, key_free(doc))

    old = record.get("entities") or {}
    old_by_ref: dict[str, str] = {}
    for entity in [*(old.get("question") or []), *(a for x in old.get("answers") or [] for a in x.get("entities") or [])]:
        if entity.get("ref") and entity.get("icon"):
            old_by_ref.setdefault(entity["ref"], entity["icon"])
    by_id = (record.get("icons") or {}).get("entities") or {}

    def icon(entity: VisualEntity) -> str:
        return known(by_id.get(entity.ref)) or old_by_ref.get(entity.ref) or KIND_FALLBACK[entity.kind]

    for entity in new.question:
        entity.icon = icon(entity)
    for answer in new.answers:
        for entity in answer.entities:
            entity.icon = icon(entity)

    new.question_icon = old.get("question_icon")
    old_answers = old.get("answers") or []
    for i, answer in enumerate(new.answers):
        answer.icon = old_answers[i].get("icon") if i < len(old_answers) else None
    return new.model_dump()


def main() -> int:
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    args = [a for a in argv if not a.startswith("--")]
    db = MongoClient(database_url(args)).get_default_database()

    changed = same = skipped = lost = 0
    for record in db["question_decomposition"].find():
        doc = db["examination"].find_one({"_id": ObjectId(record["question_id"])})
        entities = rederived(record, doc) if doc else None
        if entities is None:
            skipped += 1
            continue
        if entities == (record.get("entities") or {}):
            same += 1
            continue
        changed += 1
        if dry_run:
            continue
        done = db["question_decomposition"].update_one(
            {"_id": record["_id"], "updated_at": record.get("updated_at")},
            {"$set": {"entities": entities}},
        )
        lost += 1 - done.modified_count
    print(
        f"{'would change' if dry_run else 'changed'} {changed}, unchanged {same}, "
        f"skipped (no current decomposition) {skipped}" + (f", lost to a concurrent write {lost}" if lost else "")
    )
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
