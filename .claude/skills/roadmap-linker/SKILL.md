---
name: roadmap-linker
description: Use whenever a Questions/**/*.md card is created or substantively edited, or when the user asks to (re)link card(s) to the roadmap, add roadmap tags, or sweep for cards missing a roadmap link. Adds a `Roadmap: [[<Leaf>]]` line to a card's References section pointing at the single most specific matching note under roadmap/, and adds that leaf's kebab-case slug as a tag. Never touches roadmap/ or roadmap.yaml (generator-owned) and never edits Question/Answer/Explanation.
---

# Roadmap Linker

Keeps every card under `Questions/**` pointed at its place on the certification study
path (`roadmap/`, generated from `public/roadmap.yaml` — see `CLAUDE.md`). This is a
targeted, mechanical retrofit: **exactly two additions per card, nothing else changes.**

## When to run this

- Right after drafting or substantially rewriting a card (its topic changed enough that
  an existing roadmap link/tag might now be wrong).
- When the user names specific card(s) or a topic to (re)link.
- As a batch sweep: "every card missing a roadmap link" — find candidates with
  `grep -rL "^Roadmap: \[\[" Questions --include=*.md`.

## Step 1 — get the current leaf list

`roadmap/` is a generated tree: each branch folder has one index/MOC note sharing the
folder's name (skip these), and each real topic is a leaf note (a `.md` file whose name
differs from its parent folder). List it fresh each run — it can grow between sessions:

```
find roadmap -name "*.md" | grep -v '/Roadmap.md$'
```

A leaf note's **title** is its `# <Title>` heading (or, for slash-command notes like
`/help`, the `aliases:` frontmatter value — the filename itself is sanitized, e.g.
`help.md`, but the title/alias is the thing you link and tag). Do not invent a title —
read the note if its filename doesn't obviously match a heading (e.g. anything under
`Using Claude Code/Commands/`).

## Step 2 — pick the best leaf, per card

Read the card's Question/Answer/Explanation and choose the **single most specific**
matching leaf. Prefer a leaf over its parent branch's index note every time — only fall
back to a branch-level link if truly nothing under it fits (this should be rare; the
tree has ~100 leaves for exactly this reason). A card's `Questions/<Topic>/` folder is
not a reliable predictor of which roadmap branch it lands in — folders and roadmap
branches don't map 1:1 (e.g. a card filed under `Questions/Claude/` about `claude -p`
in CI maps to `Using Claude Code` or `Scaling Claude`, not to any "Claude" branch).

## Step 3 — apply the two edits

1. **Reference link** — if the card's `##### References` section doesn't already end
   with a `Roadmap: [[...]]` line, append one (after the existing reference links, each
   of which already ends with a trailing double-space line break):
   ```
   Roadmap: [[<Leaf Title>]]
   ```
   Use the leaf's exact title as it appears in Obsidian (its `# <Title>` heading, or its
   alias for slash-commands) — e.g. `[[MCP Tools & Schema Design]]`, `[[/help]]`,
   `[[Human-in-the-Loop Approvals]]`. If a card already has a `Roadmap:` line but the
   card's topic changed since, update it in place rather than adding a second one.

2. **Roadmap-derived tag** — add the leaf title's kebab-case slug to the frontmatter
   `tags:` list:
   - lowercase; drop `&` and other punctuation; spaces → hyphens; drop a leading `/`.
   - Examples: `MCP Tools & Schema Design` → `mcp-tools-schema-design`; `/help` →
     `help`; `State-Graph Workflows & Durable Checkpoints` →
     `state-graph-workflows-durable-checkpoints`; `Registration Scope (User vs Project)`
     → `registration-scope-user-vs-project`.
   - Skip adding it if an existing tag on the card already covers the same concept —
     per `CLAUDE.md`'s tag philosophy, don't invent near-duplicate tags (e.g. a card
     already tagged `mcp` + `json-schema` for a `MCP Tools & Schema Design` leaf doesn't
     also need `mcp-tools-schema-design`; use judgment, this is meant to be additive
     signal, not a second, redundant taxonomy on every card).
   - Never remove or rename any existing tag, and never touch the first tag (the
     `Questions/` topic-folder slug).

Nothing else in the file changes — not the title, not Question/Answer/Explanation, not
other tags, not other reference links.

## Verification

After editing, `grep -rL "^Roadmap: \[\[" Questions --include=*.md` on the touched
files (or the whole vault, for a sweep) should return nothing left unlinked, and
`git diff -- Questions/` should show only frontmatter `tags:` and one new/updated line
at the end of `##### References` per touched file.
