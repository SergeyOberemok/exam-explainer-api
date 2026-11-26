# Exam Explainer API

A FastAPI backend for a Claude certification exam question bank, plus the Obsidian vault (study cards and roadmap) it grew out of. It:

- stores practice-exam questions in MongoDB and serves them through study routes (with answer key) and a game API (`/v1/game`, no answer key until answered) for a Unity client;
- turns an uploaded exam file into playable questions through an IBM Consulting Advantage (ICA) pipeline: learning objectives, story, simplified game text, visual entities, icons and a fidelity check;
- keeps a Markdown vault (`Questions/`, `roadmap/`) with a small MCP server and Claude Code skills for studying.

See [CLAUDE.md](CLAUDE.md) for the detailed design of each part.

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- A MongoDB instance (local or Atlas)
- ICA credentials (only needed for the import pipeline)

## Getting started

```bash
uv sync
cp .env.example .env     # then fill in the values below
uv run fastapi dev       # entrypoint is set in pyproject.toml (claude_exam.api.main:app)
```

The API is then at <http://127.0.0.1:8000>; interactive docs at `/docs`.

> Don't edit `.py` files while a pipeline job runs under `fastapi dev`: the reload kills the job. Resume it with the backfill route.

### Configuration (`.env`)

| Variable | Required | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | yes | MongoDB connection string |
| `ICA_API_KEY` | yes | ICA API key |
| `ICA_URL` | yes | ICA base URL |
| `ICA_ASSISTANT` | no | ICA assistant to use |
| `API_KEY` | no | When set, the study routes (`/questions`, `/examination`, `/ica`, `/v1/search/questions`) require an `X-API-Key` header. The game routes stay open. |
| `PIPELINE_CONCURRENCY` | no | Questions sent to ICA at once (default 3); lower it on rate limits |

Indexes are created at startup. An index whose options changed is logged, not fatal, and must be dropped by hand.

## API overview

| Prefix | Purpose |
| --- | --- |
| `/v1/game` | Game for the Unity client: rounds, batches, answers, 50/50, browsing, story, objectives. Never reveals the answer before it is given. |
| `/v1/search` | Text search over questions (`/questions` with answer key, `/game-questions` without). |
| `/questions`, `/examination`, `/ica` | Study routes, import and pipeline jobs, ICA reference. |
| `/evaluate` | Prompt evaluation: UI at `/evaluate/ui/`, dry-run API behind the API key. |

Errors are `{"detail": "...", "code": "..."}`. Game responses are flat and null-free so Unity's `JsonUtility` can parse them.

## Testing the extraction pipeline (UI)

With the API running, open <http://127.0.0.1:8000/evaluate/ui/> to try a prompt or gate change on one question. Find a question by id or by its text, pick the steps to run, and see each result (steps, parts, entities, icons, decomposition, beat, flow, resolution, rephrase, check, raw JSON). Runs are dry: nothing is written to the database. If `API_KEY` is set, enter it in the page's key field.

## Adding questions

Upload a practice-exam JSON file to `POST /examination/import` (use `?dry_run=true` first to see the report: new, duplicates, unplayable, unknown scenarios/sections). The import starts a `full` job: `objectives → story → simplify → entities → icons → check`. Re-uploading the same file is safe (duplicates are skipped).

To resume a broken import or apply a changed prompt or catalog, bump the module's `PROMPT_VERSION` if needed and call `POST /examination/pipeline/backfill`. The `ingest-exam` Claude Code skill runs this end to end.

Progress report:

```bash
uv run python scripts/pipeline_report.py [DATABASE_URL] [--source FILE] [--show] [--check]
```

Other scripts in `scripts/`: `objective_coverage.py`, `story_coverage.py`, `propose_objectives.py`, `propose_scenarios.py`, `load_story_seed.py`, `export_game_icons.py`, `compare_simplified.py`.

## Tests

```bash
uv run --with pytest pytest
```

- Most tests use in-memory fakes and need no database.
- `tests/test_game_mongo.py` and the pipeline tests against a real MongoDB are opt-in: `TEST_MONGO_URL=mongodb://127.0.0.1:27017 uv run --with pytest pytest`.
- After an intended contract change: `UPDATE_CONTRACT=1 uv run --with pytest pytest tests/test_contract.py`, then commit both this repo and `../exam-cards-game`.

## The vault

- `Questions/<Topic>/<Title>.md`: Q&A cards (format in CLAUDE.md); `Templates/` holds the card template.
- `public/roadmap.yaml` is the study roadmap. Regenerate the `roadmap/` notes with `uv run python -m claude_exam.roadmap_gen` (this wipes and rewrites `roadmap/`).
- `uv run python -m claude_exam.search_cards "<query>"` searches cards from the CLI.
- `.mcp.json` registers the `claude-exam-vault` MCP server (`search_vault`, `get_card`, `list_topics`, `update_card`) and a read-only `mongodb-local` server that reads `DATABASE_URL` from `.env` (needs Node/`npx`).
- Claude Code skills in `.claude/skills/`: `short-question-answer`, `extended-question-answer`, `vault-tutor`, `rate-card`, `roadmap-linker`, `keyword-highlighter`, `promote-staged-cards`, `ingest-exam`.

## Project layout

```
src/claude_exam/
  api/          FastAPI app: routers, game/ (service, repository, domain), pipeline steps, catalogs
  vault_scan.py, search_cards.py, mcp_server.py   vault lookup tooling
  roadmap_gen.py, extract.py, ...                  roadmap and one-off import helpers
scripts/        reports and loaders
tests/          pytest suite, contract/ holds pinned responses
data/           story seed
Questions/, roadmap/, Docs/, Templates/            Obsidian vault
```
