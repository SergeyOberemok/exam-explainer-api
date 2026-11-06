"""Read-only report on simplification + entity quality.

Joins examination_simplified to examination and prints, per question: soft quality
warnings (identifiers lost, answers not simplified), feedback verdicts that disagree
with correct_response, and entity references or aliases not found in the text.
Usage: uv run python scripts/compare_simplified.py [DATABASE_URL] [--limit N]
"""

import os
import sys

from bson import ObjectId
from pymongo import MongoClient

from claude_exam.api.entities import _mentions, _source_texts
from claude_exam.api.schemas import SimplifiedPrompt, VisualEntity
from claude_exam.api.simplify import quality_warnings


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    url = args[0] if args else os.environ.get(
        "DATABASE_URL", "mongodb://localhost:27017/exam-explainer"
    )
    db = MongoClient(url).get_default_database()
    totals = {"questions": 0, "warnings": 0, "verdicts": 0, "ungrounded": 0, "aliases": 0}

    for rec in db["examination_simplified"].find().limit(limit):
        orig = db["examination"].find_one({"_id": ObjectId(rec["question_id"])})
        if not orig:
            continue
        totals["questions"] += 1
        prompt, issues = rec["prompt"], []
        for w in quality_warnings(orig, prompt):
            issues.append(w)
            totals["warnings"] += 1
        right = {c.lower() for c in rec.get("correct_response", [])}
        for i, fb in enumerate(prompt["feedbacks"]):
            if fb.strip().lower().startswith("correct") != (chr(97 + i) in right):
                issues.append(f"feedback {chr(65 + i)} verdict mismatch")
                totals["verdicts"] += 1

        ents = rec.get("entities") or {}
        q_text, a_texts = _source_texts(orig, SimplifiedPrompt(**prompt))
        places = [("question", ents.get("question", []), q_text)] + [
            (f"answer {chr(65 + i)}", a.get("entities", []), a_texts[i])
            for i, a in enumerate(ents.get("answers", []))
            if i < len(a_texts)
        ]
        everything = "\n".join([q_text, *a_texts]).lower()
        for where, items, text in places:
            for e in items:
                ent = VisualEntity(**e)
                if not _mentions(ent, text):
                    issues.append(f"{where}: ungrounded entity {ent.name!r}")
                    totals["ungrounded"] += 1
                for alias in ent.aliases:
                    if alias.lower() not in everything:
                        issues.append(f"{where}: alias {alias!r} of {ent.name!r} not in text")
                        totals["aliases"] += 1
        print(f"{rec['question_id']}  {'OK' if not issues else ''}")
        for issue in issues:
            print(f"    - {issue}")

    print("\nTOTALS", totals)


if __name__ == "__main__":
    main()
