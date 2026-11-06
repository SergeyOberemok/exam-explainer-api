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

## Game API (`/v1/game`) for the Unity client

Layers: `routers/game.py` (thin HTTP handlers) → `api/game/service.py` (`GameService`, the round rules; raises the `ApiError`s in `api/game/errors.py`) → `api/game/repository.py` (`QuestionRepository`, `RoundRepository`: the only MongoDB access, each write one conditional update) and `api/game/domain.py` (pure rules: `is_playable`, `correct_indices`, `round_entry`, `PLAYABLE_QUERY`, constants). `get_game_service` in `dependencies.py` builds the service from the collection dependencies, so tests still override the collections. `api/search.py` imports the rules from `game/domain.py`, never from a router.

The game and search routes exist only under `/v1` (`main.py`); there are no unversioned `/game` or `/search` paths. A breaking change to the game contract goes to a new prefix. No auth (the game routes must stay open), no CORS (native builds don't need it; WebGL would). Responses are flat, null-free objects with lists wrapped in `items`, because `JsonUtility` can't parse a top-level array or null.

**Errors** from every route are `{"detail": "<string>", "code": "<code>"}` (`api/errors.py`); validation errors also carry FastAPI's list in `errors`. Game codes: `round_not_found`, `question_not_found`, `simplified_not_found`, `question_not_in_round`, `question_not_answered`, `no_questions` (404); `already_answered`, `no_fifty_fifty_charges`, `too_few_answers`, `round_full`, `batch_out_of_order`, `round_changed` (409; `round_changed` means "retry"); `answer_out_of_range`, `answer_eliminated`, `question_not_playable`, `validation_error` (422); `rate_limited` (429). Other `HTTPException`s get `http_<status>`.

**No `/game` response reveals the answer key before the question is answered.** The game plays rounds; the round document in `game_rounds` (layout documented in `game/repository.py`) keeps the correct index, answer count and both versions' review per question, every question id in `question_ids`, the batches handed out, the filters, and what has been answered and eliminated (TTL index on `created_at`: rounds expire after `ROUND_TTL_SECONDS`, a day). A round holds at most `MAX_ROUND_QUESTIONS` (500).

- `POST /v1/game/rounds {count=10, section="", assessment_type=""}` → 201 `{round_id, items: GameQuestion[], simplified: GameQuestion[], total, fifty_fifty_charges}`. Samples `count * OVERSAMPLE_FACTOR` and keeps the first `count` playable questions (`is_playable`: 2..4 answers, exactly one valid correct letter); includes a simplified version only if it keeps the answer count and the correct answer (`simplified_matches`). `total` counts the pre-filter matches, so `total > len(items)` means batches can follow. 404 `no_questions` (and nothing stored) when none match. Empty strings mean "any". Rate limited (30/min per client, in process memory).
- `POST /v1/game/rounds/{round_id}/batches {batch_index, count=10}` → `{batch_index, items, simplified, remaining}`: the round's next questions, none already in it (started, searched or earlier batches), with the round's filters; empty `items` once the pool is used up (not stored). Batch 0 is the start, so the first call sends 1; asking for an existing batch returns it again (a retry), a later one is 409 `batch_out_of_order`; a concurrent add makes the write miss (409 `round_changed`).
- `POST /v1/game/rounds/{round_id}/answers {question_id, answer_index}` → `{question_id, answer_index, is_correct, correct_index, original: {explanation, feedbacks[]}, has_simplified, simplified}` (`simplified` is an empty review when there is none). One atomic `find_one_and_update`, so a question counts once. The same answer again repeats the result; another answer is 409 `already_answered`, and the client then reads the recorded one with:
- `GET /v1/game/rounds/{round_id}/answers/{question_id}` → the same `GameAnswerResult` for the answer the round recorded (404 `question_not_answered` if none). This is how a lost answer response is recovered.
- `POST /v1/game/rounds/{round_id}/fifty-fifty {question_id}` → `{question_id, eliminated_indices, fifty_fifty_remaining}`: two random wrong answers, once per unanswered question (asking again repeats the result), at most `FIFTY_FIFTY_CHARGES` (2) per round. The game takes `fifty_fifty_remaining` as the count.
- `POST /v1/game/rounds/{round_id}/questions {question_id}` → `{item: GameQuestion, simplified: GameQuestion[]}` (0 or 1 simplified): adds a question (the game's search result) to the round, conditionally, so adding one it already has changes nothing and returns it again (offering the simplified text only if the stored entry has its review). 409 `round_full`, 422 `question_not_playable`.
- `question_id` in request bodies and paths must be a 24-char ObjectId hex string (`OBJECT_ID_PATTERN` / `GameQuestionId`), which also keeps it safe as a key in the round document's field paths.
- Browsing, without answer keys, playable questions only: `GET /v1/game/questions?count=10&section=&assessment_type=&shuffle=true` → `{items: GameQuestion[], total}` (`count` 1–50), `GET /v1/game/questions/{id}`, `GET /v1/game/simplified-questions?…`, `GET /v1/game/simplified-questions/{question_id}` (404 `simplified_not_found`), `GET /v1/game/sections?simplified=false` → `{items: string[]}`.

`GameQuestion` = `id, simplified, section, assessment_type, question, answers[{index,text}]`. Simplified questions keep the examination `id` and set `simplified: true`. The stored `correct_response` letters (`"a"`, `"b"`…) are converted to indices on the server only. `examination_simplified.question_id` is the examination `_id` as a *string* (not an ObjectId); the code always joins through `str(_id)`.

The answer key is otherwise served by the study routes (`/questions`, `/examination`, `/ica`, `/v1/search/questions`). Setting `API_KEY` makes them require an `X-API-Key` header (`security.require_api_key`, applied at router level; the key is read into `app.state.api_key` at startup). Lives and score live in the client, so a determined player can still learn an answer by answering it in a throwaway round; move them into the round document if a leaderboard ever depends on them.

Indexes are created at startup by `api/indexes.py` (`ensure_indexes`); one whose options changed (e.g. new text weights) is logged, not fatal, and has to be dropped by hand to apply. The app uses a single `AsyncMongoClient`.

Tests (`uv run --with pytest pytest`): `tests/test_game.py` (routes over `tests/fakes.py`, in-memory collections implementing the operators the repository sends, with MongoDB's array semantics), `tests/test_game_mongo.py` (the service and repository against a real MongoDB in a throwaway database; opt-in with `TEST_MONGO_URL=mongodb://127.0.0.1:27017`), and `tests/test_contract.py`: real responses pinned in `tests/contract/*.json` plus the game's OpenAPI schema (`openapi_game.json`), and the same JSON written to `../exam-cards-game/Assets/Tests/EditMode/Contract/` for the Unity contract tests. After an intended contract change run `UPDATE_CONTRACT=1 uv run --with pytest pytest tests/test_contract.py` and commit both repos. `tests/conftest.py` resets rate limits, the API key and dependency overrides per test.

## Visual entities of simplified questions

After simplifying, a second ICA call (`api/entities.py`, prompt reference in `docs/extracting_actor_entities.md`, which is gitignored) extracts actors/objects/actions from the ORIGINAL (primary: exact names) and SIMPLIFIED text. `examination_simplified` stores `entities` = `{question: [{name,kind,aliases}], answers: [{entities: [{name,kind,aliases}]}]}` (answers index-aligned with `prompt.answers`; the model's id-catalog reply is resolved in `parse_entities`, and the first stored shape is converted on read) plus `entities_ica` (set only after a successful extraction). The pipeline is import → simplify → extract entities; entities are always added to an existing simplified record (`entities.enrich_and_store` `$set`s only `entities`/`entities_ica`/`updated_at`, conditional on `updated_at`, else 409), so the last step can run on its own. `simplify_and_store` treats the call as best effort; `POST /examination/{id}/entities` (shares the job lock: 409 while a job runs) and `POST /examination/entities/backfill` (job `mode="entities"`; `only_missing` is re-checked per question at run time, so repeated or overlapping backfills count `skipped` instead of calling ICA) are strict. `?entities=false` on `POST /examination/import` (job `mode="simplify_only"`) and `POST /questions/{id}/simplify` skips extraction. Jobs (`batch.new_job` / `SimplificationJob`) record `mode`, and each error its `stage` (`simplify`/`entities`); `entities_failed` counts questions whose text was stored but whose extraction failed, and a job is `failed` only if nothing was stored. A malformed stored `entities` value reads as empty. `/v1/game` sends them on simplified items as `GameQuestion.entities` (`game/domain.py` `entities_of`: answers padded/trimmed to the answer count, a malformed record gives empty lists instead of failing the round; originals and search results carry empty lists). Tests: `tests/test_entities.py`, and `tests/test_jobs.py` (job modes and routes over in-memory collections, plus the pipeline against a real MongoDB when `TEST_MONGO_URL` is set).

## Question search API (`/v1/search`)

`GET /v1/search/questions?q=&section=&assessment_type=&limit=5` (`src/claude_exam/api/routers/search.py` → service `src/claude_exam/api/search.py` → MongoDB `examination`) replaces the Obsidian card search for the lookup skills. It runs a `$text` query against a weighted text index (`examination_text`, created in the app `lifespan`: `prompt.question` 10, `section` 3, `prompt.explanation` 2, `prompt.answers` 1) and ranks by `{$meta: "textScore"}`. `q` is 2–200 chars, `limit` 1–50, empty `section`/`assessment_type` mean "any".

Response is `{items: SearchHit[], total}`; `SearchHit` = `id, score, section, assessment_type, question, answers[], correct_answers[], explanation`. **It does return the answer key** (it backs the study skills), so keep it separate from `/game`, whose responses never reveal answers before answering. Like `/questions`, it is open unless `API_KEY` is set (then it needs `X-API-Key`).

Differences from the old card search: Mongo questions have no title or tags (so no title/tag bonuses), `$text` stems and drops stopwords but does not match partial words, and scores are not comparable to the old overlap scores. `short-question-answer`, `extended-question-answer` and `vault-tutor` now call this endpoint; `vault_scan.py`, `search_cards.py` and the `claude-exam-vault` MCP server remain in place but are unused by them until removed.

`GET /v1/search/game-questions` (same query parameters) is the game's search: only playable questions (`PLAYABLE_QUERY` + `is_playable`), returned as `{items: GameQuestion[], total}` **without** the answer key or score. To play a hit, the game adds it to its round with `POST /v1/game/rounds/{round_id}/questions`. The `/search` routes are rate limited (120/min per client), and `/search/questions` needs the API key when `API_KEY` is set.

Tests: `tests/test_search.py` (fake collection with chainable `find().sort().limit()`).

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
