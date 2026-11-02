---
name: short-question-answer
description: Use whenever the user asks a question about Claude Code, the Claude Agent SDK, or the Claude/Anthropic API that matches or resembles one of the practice questions stored as .md files under the Questions/ folder of this repo (topics include Agents and Workflows, Applications and Integration, Eval Testing Debug, Model selection and Optimization, Prompt and Context Engineering, Security and Safety, Tools and MCPs). Also trigger when the user references "@Questions" or asks to look up/quiz/answer from the Questions folder. Returns only the matching card's Answer text — never the Question, frontmatter, title, Explanation, or References, and never freeform commentary.
---

# Short Question Answer Lookup

Each file under `Questions/` (recursively, including subfolders) is a single Q&A card with this fixed structure:

```
---
tags: [...]
---

# <Title>

### Question
<question text, may include a code block>
### Answer
<answer text>
##### Explanation
<one or more paragraphs>
##### References
<links>
```

## What to do

1. Take the user's question (their own phrasing, a pasted question, or a topic/title they mention).
2. Run the vault's search tool from the repo root to find candidate cards, scored by overlap against each card's `# <Title>` heading, `### Question` body, and YAML `tags`:
   ```
   uv run python -m claude_exam.search_cards "<user's question or topic, verbatim>"
   ```
   This returns a JSON array (highest score first), each entry already carrying `path`, `title`, `tags`, `question`, and `answer` — no separate Read is normally needed. Don't use Grep/Glob or rely on memory of file contents for this step; the tool is the source of truth for matching.
3. If exactly one result is clearly the best match (a large score gap over the next entry, or only one result at all), use its `answer` and `tags` fields directly.
4. If multiple results are close in score, prefer the one whose `question` field most directly corresponds to what the user asked. Only ask the user to disambiguate if genuinely tied between distinct topics. If the returned `question`/`answer` text looks truncated or you need the full file for any reason, read `path` with the Read tool.
5. If the tool returns an empty array, or nothing returned is actually about what the user asked, say so plainly instead of inventing a card.

## Output format

Use **only** the matched result's `answer` field — never its `question`, `title`, or any frontmatter/Explanation/References content. After the answer text, append its `tags` as Obsidian-style hashtags (e.g. `#tag-one #tag-two`) on their own line, so the source card can be found in Obsidian's tag search. Reply with exactly this and nothing else (no title, no preamble, no source-file mention, no explanation, no "### Answer" label):

```
<verbatim answer text>

#tag-one #tag-two ...
```

Preserve the original wording verbatim — do not paraphrase, summarize, or add the Question/Explanation/References content.
