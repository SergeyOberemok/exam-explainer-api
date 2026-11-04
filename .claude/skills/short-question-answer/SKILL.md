---
name: short-question-answer
description: Use whenever the user asks a question about Claude Code, the Claude Agent SDK, or the Claude/Anthropic API that matches or resembles one of the practice questions stored as .md files under the Questions/ folder of this repo (topics include Agents and Workflows, Applications and Integration, Eval Testing Debug, Model selection and Optimization, Prompt and Context Engineering, Security and Safety, Tools and MCPs). Also trigger when the user references "@Questions" or asks to look up/quiz/answer from the Questions folder. Returns only the matching card's Answer text — never the Question, frontmatter, title, Explanation, or References, and never freeform commentary.
---

# Short Question Answer Lookup

Questions live in the MongoDB `examination` collection and are searched through the API's `GET /v1/search/questions` endpoint (the old Obsidian `Questions/` cards are no longer used).

## What to do

1. Take the user's question (their own phrasing, a pasted question, or a topic/title they mention).
2. Query the exam API's search endpoint (MongoDB `$text` search, ranked by `textScore`). The API must be running (`fastapi dev`; default `http://localhost:8000`):
   ```
   curl -sG "http://localhost:8000/v1/search/questions" --data-urlencode "q=<user's question or topic, verbatim>" --data-urlencode "limit=5"
   ```
   This returns `{"items": [...], "total": N}` (highest `score` first), each item carrying `id`, `score`, `section`, `assessment_type`, `question`, `answers`, `correct_answers`, and `explanation`. Optional filters: `section`, `assessment_type`. Don't rely on memory for matching; the endpoint is the source of truth.
3. If exactly one item is clearly the best match (a large score gap over the next item, or only one item at all), use its `correct_answers` directly.
4. If multiple items are close in score, prefer the one whose `question` most directly corresponds to what the user asked. Only ask the user to disambiguate if genuinely tied between distinct topics.
5. If `items` is empty, or nothing returned is actually about what the user asked, say so plainly instead of inventing an answer. If the request fails to connect, tell the user the API isn't running.

## Output format

Use **only** the matched item's `correct_answers` — never its `question`, `answers` list, `explanation`, or ids. Output each correct answer's text verbatim, one per line when there are several, followed by the item's `section` as a hashtag (kebab-case, e.g. `#tools-and-mcps`) on its own line. Reply with exactly this and nothing else (no preamble, no explanation, no "Answer" label):

```
<verbatim correct answer text>

#section-slug
```

Preserve the original wording verbatim — do not paraphrase, summarize, or add question/explanation content.
