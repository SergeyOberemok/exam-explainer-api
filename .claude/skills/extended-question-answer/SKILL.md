---
name: extended-question-answer
description: Use when the user asks a question about Claude Code, the Claude Agent SDK, or the Claude/Anthropic API that matches or resembles a practice question in Questions/, AND their phrasing asks for more than a bare answer — e.g. "explain", "why", "with explanation", "with references", "full card", "give me the details". For a plain "what's the answer to X" with no request for detail, use short-question-answer instead. Returns the matching card's full verbatim content (Question, Answer, Explanation, References, tags) — never paraphrased, never with added commentary of Claude's own.
---

# Extended Question Answer Lookup

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

This skill surfaces the **whole card** verbatim via the `claude-exam-vault` MCP server, in contrast to `short-question-answer` (answer only) and `vault-tutor` (freeform rephrasing/elaboration).

## What to do

1. Take the user's question (their own phrasing, a pasted question, or a topic/title they mention).
2. Call the `search_vault` MCP tool with the user's question or topic, verbatim, to get scored candidates (`relpath`, `title`, `tags`, `score`, `question`, `answer`).
3. If exactly one result is clearly the best match (a large score gap over the next entry, or only one result at all), use it. If multiple results are close in score, prefer the one whose `question` field most directly corresponds to what the user asked — only ask the user to disambiguate if genuinely tied between distinct topics.
4. Call the `get_card` MCP tool with the chosen result's `relpath` to fetch the full card, including `explanation`, `references`, `obsidian_url`, and — when present — `rating` and `notes`.
5. If `search_vault` returns an empty array, or nothing returned is actually about what the user asked, say so plainly instead of inventing a card.

## Output format

Present the full card content **verbatim** — Question, Answer, Explanation, References, and (if present) Rating/Notes — with no paraphrasing, no summarizing, and no commentary or elaboration added on top. "Verbatim" means the wording, not the raw Obsidian markup: before inserting Question/Answer/Explanation/References text into the reply, translate every `==highlighted text==` span into an inline code span — `==text==` becomes `` `text` `` — since Obsidian's highlight syntax has no CommonMark equivalent this CLI renders with color. Do this translation on every card, every time; do not pass `==...==` through unchanged. This is a syntax swap only, never a wording change: don't alter, drop, or add to the highlighted text itself.

Structure the reply like this, then append the card's `tags` as Obsidian-style hashtags on their own line, followed by a link built from `obsidian_url`:

```
**Question:** <verbatim question text>

**Answer:** <verbatim answer text>

**Explanation:** <verbatim explanation text>

**References:** <verbatim references text>

**Rating:** <verbatim rating value>

**Notes:** <verbatim notes text>

#tag-one #tag-two ...

[Open in Obsidian](<obsidian_url>)
```

Omit the Rating and/or Notes lines entirely when the card has no `rating` or no `notes` — most cards don't, since those fields are only written by the `rate-card` skill. Never fabricate a rating or notes when the card lacks them.

Always include the `[Open in Obsidian](<obsidian_url>)` link exactly as returned by `get_card` — never rebuild or guess the URI yourself.

If the user only wants a specific piece (e.g. "just the explanation"), you may omit the other sections, but never alter the wording of what you do include.
