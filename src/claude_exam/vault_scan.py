import html
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = REPO_ROOT / "Questions"
VAULT_NAME = REPO_ROOT.name

SNIPPET_LEN = 50
QUESTION_MARKER = "QUESTION:"

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TAG_ITEM = re.compile(r"^\s*-\s*(\S.*)$", re.MULTILINE)
_RATING = re.compile(r"^rating:\s*(\d+)\s*$", re.MULTILINE)
_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_QUESTION_SECTION = re.compile(r"^### Question\s*\n(.*?)\n### Answer", re.DOTALL | re.MULTILINE)
_ANSWER_SECTION = re.compile(r"^### Answer\s*\n(.*?)\n##### Explanation", re.DOTALL | re.MULTILINE)
_EXPLANATION_SECTION = re.compile(r"^##### Explanation\s*\n(.*?)\n##### References", re.DOTALL | re.MULTILINE)
_REFERENCES_SECTION = re.compile(
    r"^##### References\s*\n(.*?)(?=\n##### Notes\s*\n|\Z)", re.DOTALL | re.MULTILINE
)
_NOTES_SECTION = re.compile(r"^##### Notes\s*\n(.*)\Z", re.DOTALL | re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")


def load_vault(questions_dir: Path = QUESTIONS_DIR) -> list[tuple[Path, str]]:
    files = []
    for path in questions_dir.rglob("*.md"):
        files.append((path, path.read_text(encoding="utf-8")))
    return files


def obsidian_url(relpath: str) -> str:
    """Build an obsidian:// URI that opens this vault straight to the given note."""
    file_param = relpath[:-3] if relpath.endswith(".md") else relpath
    return f"obsidian://open?vault={quote(VAULT_NAME)}&file={quote(file_param, safe='/')}"


def snippet_for(question_plain: str) -> str:
    question_plain = html.unescape(question_plain)
    idx = question_plain.find(QUESTION_MARKER)
    tail = question_plain[idx + len(QUESTION_MARKER):] if idx != -1 else question_plain
    return tail.strip()[:SNIPPET_LEN]


def find_existing(question_plain: str, files: list[tuple[Path, str]]) -> list[Path]:
    snippet = snippet_for(question_plain)
    return [path for path, content in files if snippet in content]


@dataclass
class Card:
    path: Path
    title: str
    tags: list[str] = field(default_factory=list)
    question: str = ""
    answer: str = ""
    explanation: str = ""
    references: str = ""
    rating: int | None = None
    notes: str = ""


def parse_card(path: Path, content: str) -> Card:
    body = content
    tags: list[str] = []
    rating: int | None = None
    fm_match = _FRONTMATTER.match(content)
    if fm_match:
        fm_text = fm_match.group(1)
        tags = [t.strip() for t in _TAG_ITEM.findall(fm_text)]
        rating_match = _RATING.search(fm_text)
        rating = int(rating_match.group(1)) if rating_match else None
        body = content[fm_match.end():]

    title_match = _TITLE.search(body)
    title = title_match.group(1).strip() if title_match else path.stem

    question_match = _QUESTION_SECTION.search(body)
    question = question_match.group(1).strip() if question_match else ""

    answer_match = _ANSWER_SECTION.search(body)
    answer = answer_match.group(1).strip() if answer_match else ""

    explanation_match = _EXPLANATION_SECTION.search(body)
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    references_match = _REFERENCES_SECTION.search(body)
    references = references_match.group(1).strip() if references_match else ""

    notes_match = _NOTES_SECTION.search(body)
    notes = notes_match.group(1).strip() if notes_match else ""

    return Card(
        path=path,
        title=title,
        tags=tags,
        question=question,
        answer=answer,
        explanation=explanation,
        references=references,
        rating=rating,
        notes=notes,
    )


def load_cards(questions_dir: Path = QUESTIONS_DIR) -> list[Card]:
    return [parse_card(path, content) for path, content in load_vault(questions_dir)]


def set_card_rating(path: Path, rating: int) -> None:
    """Add or overwrite the `rating` frontmatter property on a card, in place.

    Only the `rating` line is touched — the rest of the frontmatter block
    (notably the hand-formatted `tags:` list) and the whole body are left
    byte-identical.
    """
    content = path.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER.match(content)
    if not fm_match:
        raise ValueError(f"card has no frontmatter block: {path}")

    fm_text = fm_match.group(1)
    rating_match = _RATING.search(fm_text)
    if rating_match:
        new_fm_text = f"{fm_text[:rating_match.start()]}rating: {rating}{fm_text[rating_match.end():]}"
    else:
        new_fm_text = f"{fm_text}\nrating: {rating}"

    new_content = content[:fm_match.start(1)] + new_fm_text + content[fm_match.end(1):]
    path.write_text(new_content, encoding="utf-8")


def append_card_note(path: Path, note: str, *, today: str | None = None) -> None:
    """Append a new dated bullet to a card's `##### Notes` section, creating it if absent.

    Existing notes are never rewritten or removed — this only ever adds a line.
    """
    when = today or date.today().isoformat()
    entry = f"- **{when}**: {note.strip()}"

    content = path.read_text(encoding="utf-8")
    notes_match = _NOTES_SECTION.search(content)
    if notes_match:
        existing = notes_match.group(1).rstrip("\n")
        new_notes_block = f"{existing}\n{entry}\n"
        new_content = content[:notes_match.start(1)] + new_notes_block
    else:
        stripped = content.rstrip("\n")
        new_content = f"{stripped}\n\n##### Notes\n{entry}\n"

    path.write_text(new_content, encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("-", " ")))


def search_cards(query: str, cards: list[Card], limit: int = 5) -> list[tuple[float, Card]]:
    """Score cards against a query by overlap with the title, tags, and Question body.

    Title and tag matches are weighted highest since they're the most
    deliberate signal of what a card is about; Question-body overlap is the
    weakest signal since it just reflects shared vocabulary.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    query_lower = query.lower()

    scored = []
    for card in cards:
        score = 0.0
        score += 3.0 * len(query_tokens & _tokens(card.title))
        score += 2.5 * len(query_tokens & _tokens(" ".join(card.tags)))
        score += 1.0 * len(query_tokens & _tokens(card.question))

        title_lower = card.title.lower()
        if title_lower in query_lower or query_lower in title_lower:
            score += 5.0
        for tag in card.tags:
            if tag.lower() in query_lower:
                score += 4.0

        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:limit]
