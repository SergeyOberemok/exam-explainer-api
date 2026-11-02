---
name: vault-tutor
description: Use when the user has a confusing question/answer from Questions/ and wants it rephrased or simplified in plainer language, or says a card's explanation didn't help and wants it expanded/clarified further ("I still don't get why...", "explain that more simply", "in simpler terms", "why is X wrong"). Do NOT use this for a plain lookup request ("what's the answer to X") — use short-question-answer or extended-question-answer for that. This skill reasons and writes freeform tutor-style prose grounded in the card's actual content; it never just dumps the raw card.
---

# Vault Tutor

Each file under `Questions/` (recursively) is a Q&A card with Question, Answer, Explanation, and References sections. This skill is the tutoring layer on top of the vault: it fetches a card's real content via the `claude-exam-vault` MCP server, then reasons about it in your own words — the deliberate contrast to `short-question-answer` and `extended-question-answer`, which only ever return verbatim card text.

## What to do

1. **Identify the target card.** If a card was already surfaced earlier in this conversation (by this skill or one of the other two vault skills), reuse it. Otherwise call the `search_vault` MCP tool with the user's own wording and pick the best match using the same heuristic as the other vault skills: a clear score gap (or a single result) means take it; only disambiguate with the user if genuinely tied between distinct topics; if nothing relevant comes back, say so plainly rather than inventing an explanation.
2. **Fetch full detail.** Call `get_card` with the chosen `relpath` to get `question`, `answer`, `explanation`, and `references`.
3. **Reason from the real content — don't fabricate.** Everything you say about *why* the answer is correct or incorrect must trace back to the fetched `explanation`. You may add generally well-known background context to aid understanding, but clearly mark it as your own supplementary framing, not as vault content.
4. **Do one (or both) of these, depending on what was asked:**
   - **Rephrase/simplify:** restate the Question and/or Answer in plainer, more concrete language — analogies, shorter sentences, calling out the key distinction being tested — without changing what's technically being asked or losing accuracy.
   - **Deepen the explanation:** walk through the card's `explanation` step by step, make explicit *why* the correct answer follows and why the likely wrong alternatives don't, and point to `references` for further reading.

## Output format

Freeform, conversational tutor prose — not a rigid template, and not a verbatim card dump. It's fine (and expected) to:
- Quote a short verbatim phrase from the card's answer/explanation when it's the crux of the point.
- Add transitions, examples, or an analogy of your own to make the concept land.
- Ask a quick check-in question if it would help confirm understanding.

Do not silently invent explanation content that isn't grounded in the card's `explanation`/`references` fields, and don't append the card's full raw text as a block — the whole point of this skill is synthesis, not reproduction.
