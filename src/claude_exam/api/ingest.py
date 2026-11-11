"""What an import of a practice-exam file would do, before anything is stored.

`plan_import` sorts the file's questions into new ones and duplicates and says what the
pipeline will not be able to do for the new ones without a change to a catalog. The
import route stores the new ones; with `dry_run` it only returns the report, which is
the check to run before every upload.
"""

from collections import Counter

from claude_exam.api.game.domain import is_playable
from claude_exam.api.objective_catalog import for_section, normalize_section
from claude_exam.api.scenario_catalog import scenario_of, unknown_paragraph
from claude_exam.api.schemas import ImportCount

# How much of an unknown scenario paragraph the report shows.
PARAGRAPH_PREVIEW_CHARS = 160


def _counts(counter: Counter) -> list[ImportCount]:
    return [ImportCount(name=name, count=count) for name, count in counter.most_common()]


def plan_import(documents: list[dict], stored_hashes: set[str]) -> tuple[list[dict], dict]:
    """
    `documents` are normalized questions (each with a `text_hash`); `stored_hashes` the
    hashes among them the bank already holds. Returns the documents to insert (the first
    of each hash that is not stored) and the fields of `ImportResult` that describe them.
    """
    new: list[dict] = []
    seen = set(stored_hashes)
    for doc in documents:
        if doc["text_hash"] in seen:
            continue
        seen.add(doc["text_hash"])
        new.append(doc)

    chapters: Counter = Counter()
    paragraphs: Counter = Counter()
    sections: Counter = Counter()
    without_scenario = 0
    for doc in new:
        scenario = scenario_of(doc)
        paragraph = unknown_paragraph(doc)
        if scenario is not None:
            chapters[scenario.key] += 1
        elif paragraph:
            preview = " ".join(paragraph.split())
            if len(preview) > PARAGRAPH_PREVIEW_CHARS:
                preview = preview[:PARAGRAPH_PREVIEW_CHARS].rstrip() + "..."
            paragraphs[preview] += 1
        else:
            without_scenario += 1
        if not for_section(doc.get("section") or ""):
            sections[normalize_section(doc.get("section")) or "(no section)"] += 1

    return new, {
        "new": len(new),
        "duplicates": len(documents) - len(new),
        "unplayable": sum(not is_playable(doc) for doc in new),
        "chapters": _counts(chapters),
        "without_scenario": without_scenario,
        "unknown_scenarios": _counts(paragraphs),
        "unknown_sections": _counts(sections),
    }
