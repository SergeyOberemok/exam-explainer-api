"""Suggest learning objectives for one section from its questions (one ICA call).

A helper for editing `src/claude_exam/api/objective_catalog.py` by hand: it reads the
section's questions and explanations, sends them to the model with the objectives the
catalog already has for that section, and prints `ObjectiveDef(...)` lines to review.
It changes nothing: the catalog is what a person approves.

Usage: uv run python scripts/propose_objectives.py "<section>" [--model MODEL] [--limit N]
  <section>  as stored or without its "Domain N: " prefix
  --limit N  send at most N questions (default 60)
Needs DATABASE_URL, ICA_URL and ICA_API_KEY (environment or `.env`).
"""

import asyncio
import sys

from pymongo import MongoClient

from claude_exam.api.config import get_settings
from claude_exam.api.game.domain import answers_of, correct_indices
from claude_exam.api.ica import IcaClient
from claude_exam.api.json_reply import extract_json
from claude_exam.api.objective_catalog import KEY_PATTERN, for_section, normalize_section

SYSTEM = """\
You design the learning objectives of one section of a certification exam about Claude \
Code, the Claude Agent SDK, the Claude API, MCP, prompt engineering, tool use and agent \
design. You get the section's practice questions with their correct answers and \
explanations, and the objectives the section already has.

Propose the set of objectives this section should have: 4 to 8 in total, each one \
thing a learner can do, each tested by at least two of the questions, together covering \
every question. Keep an existing objective (same key) when it still fits; reword, merge, \
split or add where the questions show a gap.

Respond with ONLY a JSON object:
{"objectives": [{"key": "<kebab-case, starts with the section's prefix>",
                 "title": "<2-4 words>",
                 "statement": "<what the learner can do, one sentence, starts with a verb>",
                 "questions": <how many of the questions it is the main objective of>,
                 "change": "kept" | "reworded" | "new"}],
 "notes": "<one or two sentences on what changed and why>"}\
"""


def option(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


async def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        return 2
    section = normalize_section(sys.argv[1])
    settings = get_settings()
    db = MongoClient(settings.database_url).get_default_database()
    docs = [
        d for d in db["examination"].find() if normalize_section(d.get("section")) == section
    ][: int(option("--limit", "60"))]
    if not docs:
        print(f"no questions in section {section!r}")
        return 1

    existing = for_section(section)
    parts = [
        f"SECTION: {section}",
        "EXISTING OBJECTIVES:\n"
        + ("\n".join(f"- {o.key}: {o.statement}" for o in existing) or "(none)"),
    ]
    for number, doc in enumerate(docs, 1):
        prompt = doc.get("prompt") or {}
        answers = answers_of(doc)
        right = "; ".join(str(answers[i]) for i in correct_indices(doc))
        parts.append(
            f"QUESTION {number}:\n{prompt.get('question') or ''}\n"
            f"CORRECT: {right}\nWHY: {prompt.get('explanation') or ''}"
        )

    ica = IcaClient(settings.ica_url, settings.ica_api_key)
    try:
        data = await ica.chat_model(
            option("--model", "claude-sonnet-5-5"), "\n\n".join(parts), system=SYSTEM
        )
    finally:
        await ica.aclose()
    reply = extract_json(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

    print(f"# {section}: {len(docs)} questions, {len(existing)} objectives today\n")
    for item in reply.get("objectives") or []:
        key = str(item.get("key"))
        flag = "" if KEY_PATTERN.match(key) else "  # FIX THE KEY"
        print(f"    # {item.get('change')}, main objective of about {item.get('questions')} questions{flag}")
        print(
            f"    ObjectiveDef(\n        {key!r}, {section!r}, {str(item.get('title'))!r},\n"
            f"        {str(item.get('statement'))!r},\n    ),"
        )
    print(f"\n# {reply.get('notes')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
