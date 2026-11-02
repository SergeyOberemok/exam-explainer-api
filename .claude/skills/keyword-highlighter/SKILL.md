---
name: keyword-highlighter
description: Use whenever a Questions/**/*.md card is created or substantively edited, or when the user asks to highlight/re-highlight a card's correlating keywords. Wraps the decision-relevant phrase(s) in a card's Question in `==highlight==`, and the correlating phrase(s) in its Answer/Explanation that resolve them, so a reader can see at a glance which words in the answer answer which words in the question. Never rewords, reorders, adds, or removes any word — markup-only. Never touches frontmatter/tags, References, the Roadmap line, rating, Notes, code blocks, or existing `**bold**` text.
---

# Keyword Highlighter

Every card already states its Question, Answer, and (usually) Explanation in full — the
justification is already there, but nothing visually ties the specific words in the
Question to the words in the Answer/Explanation that resolve them. This skill adds that
visual link with Obsidian's `==highlight==` syntax. This is a targeted, mechanical
retrofit: **markup only, nothing else changes.**

## When to run this

- Right after drafting or substantially rewriting a card's Question/Answer/Explanation
  (new prose has no highlighting yet, or old highlighting no longer tracks the words).
- When the user names specific card(s) or a topic to (re)highlight.
- As a batch sweep across `Questions/**/*.md`, e.g. after a bulk content edit.

## Step 1 — read the whole card

Read the full file before editing. Note which parts are fenced code blocks (never
touched) and whether any `**bold**` already exists in the Explanation — this vault uses
`**bold**` for a separate, pre-existing convention (bolding the one sentence that
restates the answer). Leave every existing `**bold**` span exactly as it is; don't wrap
any of it in `==...==` and don't unwrap it.

## Step 2 — pick the correlating phrases

In `### Question`, identify the phrase(s) that state the decision-relevant fact — the
specific defect, constraint, or scenario detail the answer actually turns on (not
generic scene-setting nouns like "agent" or "tool call" just because they're topical).

In `### Answer` and `##### Explanation`, find the phrase(s) that directly resolve or
justify that fact, and highlight those too, so the two ends of the correlation are both
visible.

**Be sparing.** One highlighted span per section is correct — often preferable — when a
single phrase is the decisive keyword; don't force multiple highlights per section just
to fill a range. A typical card ends up with 1-3 highlighted spans per section, never
near-every-word. A multi-paragraph Explanation doesn't need a highlight in every
paragraph, only where the correlating fact/fix actually appears.

## Step 3 — apply with Edit only

Use `Edit` with an exact `old_string` → `new_string` pair that only wraps an existing
exact substring in `==...==` — never reword, reorder, add, or remove any word,
character, or punctuation outside the two `==` marks inserted. If a phrase is already
wrapped in `==...==` from a prior pass, leave it as-is rather than re-wrapping or
double-wrapping. Never touch:

- YAML frontmatter (`tags:`, `rating:`)
- `##### References` and its trailing `Roadmap: [[...]]` line
- an existing `##### Notes` section
- any existing `**bold**` text
- anything inside a fenced code block

If a card already has messy or inconsistent highlighting (highlighting filler words,
mixing with bold, wrapping too much), clean it up to match this sparing style rather
than leaving it as-is — but still only via markup add/remove, never a wording change.

## Verification

For every touched file, confirm the change is markup-only: stripping `==` (and `\r`, to
ignore line-ending noise) from the new content must reproduce the pre-edit content with
the same strip applied.

```
git diff --name-only -- Questions/ | while IFS= read -r f; do
  old=$(git show HEAD:"$f" | tr -d '=\r')
  new=$(cat "$f" | tr -d '=\r')
  [ "$old" = "$new" ] || echo "WORDING CHANGED: $f"
done
```

Any file this flags must be fixed before considering the pass complete. Also spot-check
with `git diff` that no frontmatter/References/Notes lines were touched.
