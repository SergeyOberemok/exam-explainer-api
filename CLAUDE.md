# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An Obsidian vault used to build and study a Q&A bank for a Claude Code / Claude API certification exam. The content itself has no build system, linter, or test suite — it's a vault of Markdown notes. A small uv-managed Python package, `src/claude_exam`, backs the vault-lookup tooling (card search/parsing, an MCP server) that the `.claude/skills/` skills use to serve the notes.

## Structure

- `Questions/<Topic>/<Question Title>.md` — the Q&A bank. Topics currently: Agents and Workflows, Applications & Integration & Orchestration, Claude, Context Management & Reliability, Eval Testing Debug, Model selection and Optimization, Prompt and Context Engineering, Security and Safety, Tools and MCPs. No question files sit directly under `Questions/` uncategorized anymore — every card lives in a topic folder.
- `Templates/Question_Template.md` — the Obsidian template for new question cards (configured as the templates folder in `.obsidian/templates.json`). It only contains the section skeleton (`### Question` / `### Answer` / `##### Explanation` / `##### References`) — the YAML tags frontmatter and `# <Title>` heading are added by hand per file, not by the template.
- `Docs/` — free-standing Anthropic SDK reference notes. `Docs/ICA/` holds the IBM Consulting Advantage API reference (`README.md` summary plus the full `openapi.yaml`) used by the `/ica` routes in `src/claude_exam/api/`.
- `.obsidian/` — Obsidian app config; not relevant to editing content.

## Question card format

Every file in `Questions/` follows this exact fixed structure — preserve it when adding or editing cards:

```
---
tags:
  - topic-slug
  - keyword-slug
  - ...
---

# <Title matching the filename>

### Question
<scenario/question text, may include a code block>
### Answer
<the correct answer, stated as a decision plus its direct justification>
##### Explanation
<2-3 short paragraphs justifying the answer>
##### References
<one or more docs links, one per line with a trailing double-space/line-break>
```

Tags are topic/keyword slugs (kebab-case) used for Obsidian's tag pane/search, not a controlled vocabulary — reuse existing tags across files where the topic overlaps rather than inventing near-duplicates.

A card's `##### References` section ends with an extra `Roadmap: [[Step Title]]` line, wikilinking the matching note under `roadmap/` (see below) — this gives study context for where the question's problem sits on the certification learning path. A card may also carry a roadmap-leaf-derived tag (kebab-case slug of that same `Step Title`) alongside its topic/keyword tags, added only when no existing tag already covers the same concept. The `.claude/skills/roadmap-linker/` skill applies both additions to new or edited cards.

A card may optionally also carry a `rating` frontmatter property (integer 0-100, a self-assessed "how well do I understand this" score, overwritten on each save) and a trailing `##### Notes` section (a dated bullet log, e.g. `- **2026-08-29**: kept mixing up X and Y`, appended to — never overwritten — on each save). Both are absent on most cards; they're written by the `.claude/skills/rate-card/` skill via the MCP server's `update_card` tool, never added by hand.

Within Question/Answer/Explanation prose, `==highlight==` marks the phrase(s) that correlate a Question's decisive fact to the phrase in the Answer/Explanation that resolves it (sparingly — often just one span per section); `**bold**` is a separate, older convention that bolds the single Explanation sentence restating the answer. The two never overlap. The `.claude/skills/keyword-highlighter/` skill applies the former to new or edited cards.

## `roadmap/` and `public/roadmap.yaml`

`public/roadmap.yaml` is the certification study roadmap: a nested list of `title` (+ optional `children`) nodes, in learning-path order. `src/claude_exam/roadmap_gen.py` (re)generates the `roadmap/` folder from it — one folder + index/MOC note (`## Steps` linking to its children) per branch, one plain note (`Part of: [[Parent]]`) per leaf. Regenerate after editing the YAML with:

```
uv run python -m claude_exam.roadmap_gen
```

This wipes and rewrites `roadmap/` from scratch, so make structural edits in `roadmap.yaml`, not directly in the generated notes.

## `src/claude_exam` and the `claude-exam-vault` MCP server

`src/claude_exam/vault_scan.py` parses every `Questions/**/*.md` file into a `Card` (title, tags, question, answer, explanation, references, rating, notes) and scores cards against a query by title/tag/question overlap (`search_cards()`). It also holds the vault's only write helpers, `set_card_rating()` and `append_card_note()`, which do targeted in-place edits (never a full frontmatter/body re-serialization) so untouched lines stay byte-identical. Two things sit on top of it:

- `src/claude_exam/search_cards.py` — a CLI (`uv run python -m claude_exam.search_cards "<query>"`) used by the `short-question-answer` skill via Bash.
- `src/claude_exam/mcp_server.py` — the MCP server (registered in `.mcp.json` as `claude-exam-vault`), mostly a thin retrieval layer exposing `search_vault`, `get_card`, and `list_topics`, plus one narrow write tool, `update_card(relpath, rating=None, notes=None)`, used by the `rate-card` skill. No LLM calls happen inside the server; any rephrasing/explaining/reasoning about what to write is done by the calling skill itself.

`extract.py`, `section_map.py`, `explanation_parse.py`, and `html_clean.py` are a separate, one-off pipeline that converts raw practice-exam JSON (`public/*.json`) into staged candidate cards (`.staging/*.pending.json`) — unrelated to the lookup tooling above. Turning a staged record into an actual `Questions/**/*.md` file is a manual/agent-driven step, not a script — see the `promote-staged-cards` skill below.

## Card-authoring skills

These three skills retrofit or create cards under `Questions/` — distinct from the four read-oriented skills below, which only ever look cards up:

- **`roadmap-linker`** — adds the `Roadmap: [[Step Title]]` reference line and, where warranted, a roadmap-derived tag, linking a card to its place on the `roadmap/` study path.
- **`keyword-highlighter`** — adds sparing `==highlight==` markup correlating a card's Question to its Answer/Explanation (see "Question card format" above).
- **`promote-staged-cards`** — turns a `.staging/*.pending.json` record into a real card (title, folder, tags authored by hand from the record's already-cleaned question/answer/explanation), then explicitly invokes `roadmap-linker` and `keyword-highlighter` on it as its last step. This is the intended entry point whenever new practice-exam questions get added to the vault — the other two skills otherwise only run when a card is edited later or the user asks for them by name.

## The four vault skills

Each skill under `.claude/skills/` has a distinct, deliberate output contract — preserve these when editing:

- **`short-question-answer`** — Bash + the `search_cards` CLI. Returns **only** the matched card's `### Answer` text verbatim plus its tags as hashtags — never Question/Explanation/References/frontmatter, never added commentary. For plain "what's the answer to X" lookups.
- **`extended-question-answer`** — the `claude-exam-vault` MCP tools (`search_vault` + `get_card`). Returns the **full card verbatim** (Question, Answer, Explanation, References, tags) — still no paraphrasing or commentary, just more of the card. For lookups where the user also wants the explanation/references.
- **`vault-tutor`** — the same MCP tools, but reasons in Claude's own words: rephrases a confusing question/answer in simpler language, or expands on a card's explanation when the user says it didn't help. The only one of the first three that produces freeform prose rather than verbatim card text.
- **`rate-card`** — the only skill that writes. Takes a rating (0-100) and/or note the user just stated about a card already discussed in the conversation, and persists it via the MCP `update_card` tool. Never fabricates a value, never replaces prior notes (always appends), never touches Question/Answer/Explanation/References/tags.
