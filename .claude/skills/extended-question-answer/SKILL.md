---
name: extended-question-answer
description: Use when the user asks a question about Claude Code, the Claude Agent SDK, or the Claude/Anthropic API that matches or resembles a practice question in Questions/, AND their phrasing asks for more than a bare answer — e.g. "explain", "why", "with explanation", "with references", "full card", "give me the details". For a plain "what's the answer to X" with no request for detail, use short-question-answer instead. Returns the matching card's full verbatim content (Question, Answer, Explanation, References, tags) — never paraphrased, never with added commentary of Claude's own.
---

# Extended Question Answer Lookup

Questions live in the MongoDB `examination` collection and are searched through the API's `GET /search/questions` endpoint (the old Obsidian `Questions/` cards are no longer used).

This skill surfaces the **whole question record** verbatim, in contrast to `short-question-answer` (answer only) and `vault-tutor` (freeform rephrasing/elaboration).

## What to do

1. Take the user's question (their own phrasing, a pasted question, or a topic they mention).
2. Query the search endpoint (the API must be running, `fastapi dev`; default `http://localhost:8000`), passing the user's wording verbatim:
   ```
   curl -sG "http://localhost:8000/search/questions" --data-urlencode "q=<user's question or topic, verbatim>" --data-urlencode "limit=5"
   ```
   Each item in `items` (highest `score` first) already carries `id`, `score`, `section`, `assessment_type`, `question`, `answers`, `correct_answers`, and `explanation`, so no second call is needed. Optional filters: `section`, `assessment_type`.
3. If exactly one item is clearly the best match (a large score gap over the next item, or only one item at all), use it. If several are close in score, prefer the one whose `question` most directly corresponds to what the user asked — only ask the user to disambiguate if genuinely tied between distinct topics.
4. If `items` is empty, or nothing returned is actually about what the user asked, say so plainly instead of inventing a record. If the request fails to connect, tell the user the API isn't running.

## Output format

Present the record **verbatim** — Question, Answer options, Correct answer, Explanation — with no paraphrasing, no summarizing, and no commentary added on top. Structure the reply like this, then append the `section` as a kebab-case hashtag on its own line:

```
**Question:** <verbatim question text>

**Options:**
- <answers[0]>
- <answers[1]>
...

**Answer:** <verbatim correct_answers, one per line if several>

**Explanation:** <verbatim explanation text>

#section-slug
```

The API has no references, rating, notes, or Obsidian link, so never include or fabricate them.

If the user only wants a specific piece (e.g. "just the explanation"), you may omit the other sections, but never alter the wording of what you do include.
