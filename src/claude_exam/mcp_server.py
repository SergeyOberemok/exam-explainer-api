from pathlib import Path

from mcp.server.mcpserver import MCPServer

from claude_exam.vault_scan import (
    QUESTIONS_DIR,
    REPO_ROOT,
    append_card_note,
    load_cards,
    obsidian_url,
    parse_card,
    search_cards,
    set_card_rating,
)

mcp = MCPServer("claude-exam-vault")


@mcp.tool()
def search_vault(query: str, limit: int = 5) -> list[dict]:
    """Search Questions/ cards by title/tags/question overlap.

    Returns scored candidates (relpath, title, tags, score, question,
    answer) — no explanation/references; call get_card for full detail
    on a chosen match.
    """
    cards = load_cards()
    results = search_cards(query, cards, limit=limit)
    out = []
    for score, card in results:
        relpath = str(card.path.relative_to(REPO_ROOT)).replace("\\", "/")
        out.append(
            {
                "relpath": relpath,
                "title": card.title,
                "tags": card.tags,
                "score": score,
                "question": card.question,
                "answer": card.answer,
                "obsidian_url": obsidian_url(relpath),
            }
        )
    return out


def _resolve_in_questions(relpath: str) -> Path:
    target = (REPO_ROOT / relpath).resolve()
    if QUESTIONS_DIR.resolve() not in target.parents:
        raise ValueError(f"relpath must be inside Questions/: {relpath}")
    if not target.is_file():
        raise FileNotFoundError(relpath)
    return target


@mcp.tool()
def get_card(relpath: str) -> dict:
    """Return the full content of one vault card given a relpath from search_vault.

    Includes title, tags, question, answer, explanation, and references.
    """
    target = _resolve_in_questions(relpath)

    card = parse_card(target, target.read_text(encoding="utf-8"))
    relpath = str(target.relative_to(REPO_ROOT)).replace("\\", "/")
    return {
        "relpath": relpath,
        "title": card.title,
        "tags": card.tags,
        "question": card.question,
        "answer": card.answer,
        "explanation": card.explanation,
        "references": card.references,
        "rating": card.rating,
        "notes": card.notes,
        "obsidian_url": obsidian_url(relpath),
    }


@mcp.tool()
def list_topics() -> list[dict]:
    """List topic folders under Questions/ with their card counts, for browsing."""
    topics = {
        p.name: sum(1 for _ in p.glob("*.md"))
        for p in QUESTIONS_DIR.iterdir()
        if p.is_dir()
    }
    return [{"topic": name, "card_count": count} for name, count in sorted(topics.items())]


@mcp.tool()
def update_card(relpath: str, rating: int | None = None, notes: str | None = None) -> dict:
    """Record self-assessed understanding for a card already fetched via get_card.

    rating: 0-100 integer; overwrites the card's `rating` frontmatter property.
    notes: free text; appended as a new dated bullet under a `##### Notes`
    section — existing notes are preserved, never overwritten.
    At least one of rating/notes must be given. Returns the updated card
    (same shape as get_card) so the caller can confirm what was stored.
    """
    if rating is None and notes is None:
        raise ValueError("update_card requires at least one of rating or notes")
    if rating is not None and not (0 <= rating <= 100):
        raise ValueError(f"rating must be between 0 and 100: {rating}")

    target = _resolve_in_questions(relpath)

    if rating is not None:
        set_card_rating(target, rating)
    if notes:
        append_card_note(target, notes)

    card = parse_card(target, target.read_text(encoding="utf-8"))
    relpath = str(target.relative_to(REPO_ROOT)).replace("\\", "/")
    return {
        "relpath": relpath,
        "title": card.title,
        "tags": card.tags,
        "question": card.question,
        "answer": card.answer,
        "explanation": card.explanation,
        "references": card.references,
        "rating": card.rating,
        "notes": card.notes,
        "obsidian_url": obsidian_url(relpath),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
