---
name: vault-tutor
description: Use when the user has a confusing question/answer from Questions/ and wants it rephrased or simplified in plainer language, or says a card's explanation didn't help and wants it expanded/clarified further ("I still don't get why...", "explain that more simply", "in simpler terms", "why is X wrong"). Do NOT use this for a plain lookup request ("what's the answer to X") — use short-question-answer or extended-question-answer for that. This skill reasons and writes freeform tutor-style prose grounded in the card's actual content; it never just dumps the raw card.
---

# Vault Tutor

Questions live in the MongoDB `examination` collection and are searched through the API's `GET /search/questions` endpoint (the old Obsidian `Questions/` cards are no longer used). This skill is the tutoring layer on top: it fetches a question's real content from the API, then reasons about it in your own words — the deliberate contrast to `short-question-answer` and `extended-question-answer`, which only ever return verbatim text.

## What to do

1. **Identify the target question.** If a question was already surfaced earlier in this conversation (by this skill or one of the other two lookup skills), reuse it. Otherwise query the API (it must be running, `fastapi dev`; default `http://localhost:8000`) with the user's own wording:
   ```
   curl -sG "http://localhost:8000/search/questions" --data-urlencode "q=<user's wording>" --data-urlencode "limit=5"
   ```
   Pick the best match using the same heuristic as the other lookup skills: a clear `score` gap (or a single item) means take it; only disambiguate with the user if genuinely tied between distinct topics; if `items` is empty, say so plainly rather than inventing an explanation. If the request fails to connect, tell the user the API isn't running.
2. **Use the returned detail.** Each item already has `question`, `answers`, `correct_answers`, and `explanation`; no second call is needed.
3. **Reason from the real content — don't fabricate.** Everything you say about *why* the answer is correct or incorrect must trace back to the returned `explanation`. You may add generally well-known background context to aid understanding, but clearly mark it as your own supplementary framing, not as content from the question bank.
4. **Do one (or both) of these, depending on what was asked:**
   - **Rephrase/simplify:** restate the Question and/or Answer in plainer, more concrete language — analogies, shorter sentences, calling out the key distinction being tested — without changing what's technically being asked or losing accuracy.
   - **Deepen the explanation:** walk through the `explanation` step by step, make explicit *why* the correct answer follows and why the likely wrong alternatives (the other `answers`) don't.

## Output format

Freeform, conversational tutor prose — not a rigid template, and not a verbatim card dump. It's fine (and expected) to:
- Quote a short verbatim phrase from the answer/explanation when it's the crux of the point.
- Add transitions, examples, or an analogy of your own to make the concept land.
- Ask a quick check-in question if it would help confirm understanding.

Do not silently invent explanation content that isn't grounded in the returned `explanation`, and don't append the full raw record as a block — the whole point of this skill is synthesis, not reproduction.
