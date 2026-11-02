---
name: rate-card
description: Use when the user states a self-assessed understanding rating (0-100) and/or a study note about a Questions/ card that has already been discussed in this conversation ("I'd rate my understanding of that 70", "note that I keep confusing X", "save that as an 80"). Persists the rating into the card's `rating` frontmatter property and/or the note as a new dated bullet under a `##### Notes` section, via the claude-exam-vault MCP `update_card` tool. Do NOT use this for a plain lookup request, or to fetch/explain a card — use short-question-answer, extended-question-answer, or vault-tutor for that.
---

# Rate Card

The other three vault skills only read cards. This skill is the one write path: it takes a rating and/or note the user just stated about a card already surfaced in this conversation, and persists it into that card's Markdown file via the `claude-exam-vault` MCP server's `update_card` tool.

## What to do

1. **Identify the target card.** Reuse the `relpath` of the card most recently fetched (via `get_card`/`search_vault`) earlier in this conversation by any of the four vault skills. If no card has been discussed yet, or the user's referent is ambiguous, call `search_vault` to locate it — or ask the user which card they mean — before writing anything.
2. **Extract exactly what the user stated.** Pull the rating (a 0-100 integer) and/or the note text directly from the user's message. Never invent, round, or infer a value they didn't give — if they only gave a note, pass only `notes`; if they only gave a rating, pass only `rating`.
3. **Call `update_card(relpath, rating=..., notes=...)`**, passing only the fields that were actually stated (omit/leave `None` the other).
4. **Confirm tersely.** State what was saved and on which card, e.g. "Saved rating 70 and a note on *Fixing CI Not Triggering on GITHUB_TOKEN-Authored Commits*." Do not dump the full card back — that's what the other three skills are for.

## Constraints

- Never fabricate a rating or note the user didn't state.
- Notes always **append** — never replace or delete a prior note. `update_card` handles this; don't pre-edit the file yourself.
- Never touch Question/Answer/Explanation/References content, tags, or the Roadmap line.
- Only act on the card clearly being discussed — if it's unclear which card, ask rather than guessing.
