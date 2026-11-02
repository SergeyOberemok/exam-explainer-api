import json
import sys

from claude_exam.vault_scan import REPO_ROOT, load_cards, search_cards

DEFAULT_LIMIT = 5


def main(argv: list[str]) -> int:
    args = argv[1:]
    limit = DEFAULT_LIMIT
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]

    if len(args) != 1:
        print(
            'usage: python -m claude_exam.search_cards "<query>" [--limit N]',
            file=sys.stderr,
        )
        return 2

    query = args[0]
    cards = load_cards()
    results = search_cards(query, cards, limit=limit)

    payload = [
        {
            "path": str(card.path),
            "relpath": str(card.path.relative_to(REPO_ROOT)),
            "title": card.title,
            "tags": card.tags,
            "score": score,
            "question": card.question,
            "answer": card.answer,
        }
        for score, card in results
    ]

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
