---
name: ingest-exam
description: Use when the user wants to add practice-exam questions to the game's question bank, asks to import or upload a practice-exam JSON file, to process, classify, simplify or "adapt to the game" new questions, to resume or check an import job, or to apply a changed prompt or catalog to stored questions. Runs the ingest end to end through the local API (dry run, import, job, report) and says what to do for each kind of finding. Not for looking up or explaining a question (use the question-answer skills).
---

# Ingest practice-exam questions

One upload does everything for a new question: learning objectives, its place in the
game's story, its decomposition (with semantic parts and visual entities built from it),
icons, the flow shown before answering and the resolution shown after, a rephrase (the
question in plain words, set in the company, written without the key) and a check that
the rephrase still has the same right answer. Your
job is to run it, read the report, and fix what the report names. Do not hand-write
records for questions: if the pipeline cannot do something, the fix is a catalog or a
prompt, so the next upload needs no hand work either.

The API must be running (`uv run fastapi dev`, `http://127.0.0.1:8000`). If `API_KEY`
is set in `.env`, add `-H "X-API-Key: <key>"` to every call below.

Do not edit any `.py` file of this repository while a job runs: `fastapi dev` reloads
on a change and the job dies with the process. (Nothing is lost: step 5 resumes it.)

## 1. Dry run

```
curl -s -X POST "http://127.0.0.1:8000/examination/import?dry_run=true" -F "file=@<path to file>.json"
```

Nothing is stored. Read the result before going on:

| Field | Meaning | What to do |
|---|---|---|
| `new` / `duplicates` | Questions that would be stored / that the bank already holds (same text and answers) | Nothing. Uploading a file twice is safe. |
| `unplayable` | Not pick-one with 2-4 answers | Nothing. They are stored for the study routes; the game leaves them out. Tell the user the count. |
| `chapters`, `without_scenario` | New questions per chapter, and how many have no `SCENARIO:` paragraph | Nothing. The story step picks a chapter for those without a paragraph. |
| `unknown_scenarios` | Scenario paragraphs no chapter claims | Step 2 before importing, or import now and do step 2 afterwards. |
| `unknown_sections` | Sections without learning objectives | Step 3, same choice. |

## 2. A new scenario (only when `unknown_scenarios` is not empty)

After the import (the script reads stored questions): `uv run python scripts/propose_scenarios.py`
prints each unclaimed paragraph, its question count, sample questions and a
`ScenarioDef(...)` skeleton. Add a chapter to `src/claude_exam/api/scenario_catalog.py`:
a team of the same company wanting that system, an intro in the company's words, two
named people in `cast`, and a `marker` (lower-case words of the paragraph found in no
other). Show the user the new chapter; it is theirs to approve. Other chapters are not
redone.

## 3. A new section (only when `unknown_sections` is not empty)

`uv run python scripts/propose_objectives.py "<section>"` asks the model for objectives
from the section's questions and prints `ObjectiveDef(...)` lines. Add the ones that
hold up to `src/claude_exam/api/objective_catalog.py` and show the user. Changing the
catalog makes the next backfill tag every question again (one call per question).

## 4. Import

```
curl -s -X POST "http://127.0.0.1:8000/examination/import" -F "file=@<path to file>.json"
```

It returns `job_id`. Poll `GET /examination/jobs/<job_id>` until `status` is `completed`
or `failed` (about 30 seconds per question with the default three at a time; eight
model calls each, more when a gate asks for a repair). In the job: `succeeded`, `failed`, `stages_failed` (failures per step) and
`errors` (question, step, status, detail).

- 422 errors are the catalog gaps of steps 2 and 3. Fix the catalog, then step 5.
- 502 errors are replies the gates rejected three times (the detail says which rule).
  A few are normal; step 5 tries them again. The same question failing twice is worth
  reading: the rule may be wrong for it, which is a change to `exam_terms.py` or the
  prompt, not a hand-written text.
- 429 errors: lower `PIPELINE_CONCURRENCY` in `.env`, restart the API, then step 5.

## 5. Resume or apply a change

```
curl -s -X POST "http://127.0.0.1:8000/examination/pipeline/backfill"
```

Starts a job for every stored question that lacks a step or has one written from other
inputs (another prompt, catalog, chapter or upstream result: each step stores a key of
the messages it sent, so there is nothing to bump), and runs only those steps. Use it
after a crash, after fixing a catalog, and after changing a prompt.
`{"job_id": null, "total": 0}` means nothing is left. `?source=<file name>` limits it
to one upload; `?limit=N` to a few questions for a trial.

## 6. Report

```
uv run python scripts/pipeline_report.py --source <file name> --check
```

Exit status 0 means done. Otherwise it names what is left:

- **steps to run**: step 5.
- **section without objectives / scenario no chapter claims**: steps 3 and 2.
- **rephrases the check found changed**: a model answering the rephrase without the
  key picked a wrong answer, but the right one on the original. The game does not offer
  these rephrases (only the original). Read them with `--show`, write each again
  with `POST /questions/<id>/rephrase?force=true` and check it with
  `POST /examination/<id>/check`. If the same kind of mistake repeats, fix the prompt in
  `rephrase.py` instead.
- **current rephrases that break a fidelity rule**: should be 0; anything listed means
  a gate changed since the text was written. Bump `rephrase.CODE_VERSION` and run step 5.
- **pre-answer flows that may point at the correct answer**: the flow shown before
  answering shares clearly more words with the correct answer than with the others.
  Look at one with `uv run python scripts/run_pipeline.py <id> --stages flow --dry-run --mermaid`;
  redo it with `POST /examination/<id>/flow` (then `/resolve`), or, when it repeats,
  tighten the flow prompt in `flow.py` (the next backfill redoes the flows).
- **hard** (for information): the model missed the question in both wordings. Nothing
  is wrong with the rephrase; these are the bank's hardest questions.

To look at or redo one question step by step, use
`uv run python scripts/run_pipeline.py <id> --show --mermaid` (add `--dry-run` to write
nothing, `--stages a,b` for some steps, `--force` to redo current ones).

Then `uv run python scripts/story_coverage.py` and `uv run python scripts/objective_coverage.py`
show how the new questions spread over chapters, stages and objectives.

## What to tell the user at the end

How many questions were stored and how many were duplicates; how many have everything;
what is left and whose decision it is (a new chapter or new objectives are theirs to
approve); and how many rephrases the check held back.
