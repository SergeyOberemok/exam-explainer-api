"""Run the ingest pipeline on ONE question and see what each step made of it.

The way to try a prompt or gate change on a real question before a backfill, or to redo
one question by hand.

- Default (a real run): starts a job for that one question exactly as
  `POST /examination/pipeline/backfill` would (`batch.run_simplification_job`, same code
  path, visible under `GET /examination/jobs/{job_id}`), so it UPDATES the question's
  records in the database: its objectives, story, decomposition (with its parts,
  entities and icons), flow, resolution and game document (compose), as far as the
  chosen steps go (`--mode rephrase` for the rephrase and its check). Without `--force` a step whose
  result is current is skipped (`only_missing`).
- `--dry-run`: calls the steps' compute functions only, feeding each step's result to the
  next in memory, and writes NOTHING. A step that is not run uses what is stored
  (`--stages flow` draws the flow of the stored decomposition). The objectives and story
  steps have no dry form: their stored records are used (a dry beat is written on the
  stored story's stage).

With a decomposition, `--show` and `--mermaid` also print the pre-answer flow built in
code (`flow.flow_from`, an experiment that is never stored) with the flow gate's verdict.

Usage:
  uv run python scripts/run_pipeline.py QUESTION_ID [DATABASE_URL]
      [--stages decompose,icons,flow] [--mode full] [--model claude-haiku-5-5]
      [--force] [--dry-run] [--show] [--mermaid] [--out FILE]

  --stages    comma-separated steps, run in the given order (default: those of --mode)
  --mode      a job mode (`batch.STAGES`), default "full"
  --force     a real run redoes every chosen step, current or not
  --show      print the question, its beat, decomposition, parts (with their roles),
              entities with their icons, the resolution, the game card and the rephrase
  --mermaid   print the flows as Mermaid `flowchart TD`, to paste into any Mermaid viewer
  --out FILE  write everything as JSON

Limits: the API's job lock lives in the API process and does not cover this one; if a
job of the API touches the same question meanwhile, the conditional writes answer 409
and the step fails (run it again). Do not run it while `fastapi dev` reloads.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import get_args

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import AsyncMongoClient

from claude_exam.api import batch
from claude_exam.api.config import get_settings
from claude_exam.api.flow import mermaid
from claude_exam.api.game.domain import answers_of, correct_indices
from claude_exam.api.ica import IcaClient
from claude_exam.api.pipeline_preview import Result, run_dry
from claude_exam.api.schemas import IcaModelId, JobMode, JobStage, SimplificationJob
from claude_exam.html_clean import clean_plain, clean_question


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question_id")
    parser.add_argument("database_url", nargs="?")
    parser.add_argument("--stages", default="")
    parser.add_argument("--mode", default="full", choices=list(batch.STAGES))
    parser.add_argument("--model", default="claude-haiku-5-5", choices=list(get_args(IcaModelId)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--mermaid", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    known = set(get_args(JobStage))
    args.stage_list = tuple(s.strip() for s in args.stages.split(",") if s.strip()) or batch.STAGES[args.mode]
    unknown = [s for s in args.stage_list if s not in known]
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    try:
        args.object_id = ObjectId(args.question_id)
    except InvalidId:
        parser.error(f"not a question id: {args.question_id!r}")
    return args


# --- a dry run (claude_exam.api.pipeline_preview) or a real job ----------------------


async def dry_run(args, ica: IcaClient, db, doc: dict) -> Result:
    question_id = str(doc["_id"])
    result = Result.from_records(
        doc,
        await db["examination_simplified"].find_one({"question_id": question_id}),
        await db["question_decomposition"].find_one({"question_id": question_id}),
    )
    story = await db["question_story"].find_one({"question_id": question_id})
    tags = await db["question_objectives"].find_one({"question_id": question_id})
    async for step in run_dry(
        ica, doc, result, story, args.stage_list, args.model, objectives_record=tags
    ):
        print_step(step)
    if result.beat is None:
        result.beat = (story or {}).get("beat") or ""
    return result


async def real_run(args, ica: IcaClient, mongo: AsyncMongoClient, db, doc: dict) -> Result:
    jobs = db["simplification_jobs"]
    mode: JobMode = args.mode
    inserted = await jobs.insert_one(
        batch.new_job("run_pipeline.py", args.model, 1, mode, only_missing=not args.force)
    )
    print(f"  job {inserted.inserted_id} ({'all chosen steps' if args.force else 'only what is missing'})")
    await batch.run_simplification_job(
        mongo,
        ica,
        asyncio.Semaphore(1),
        inserted.inserted_id,
        [doc["_id"]],
        args.model,
        mode,
        only_missing=not args.force,
        concurrency=1,
        stages=args.stage_list,
    )
    raw = await jobs.find_one({"_id": inserted.inserted_id})
    job = SimplificationJob(**{**raw, "id": str(raw.pop("_id"))})
    question_id = str(doc["_id"])
    result = Result.from_records(
        doc,
        await db["examination_simplified"].find_one({"question_id": question_id}),
        await db["question_decomposition"].find_one({"question_id": question_id}),
        await db["game_question"].find_one({"question_id": question_id}),
    )
    story = await db["question_story"].find_one({"question_id": question_id})
    result.beat = (story or {}).get("beat") or ""
    failed = {e.stage: e for e in job.errors}
    for stage in args.stage_list:
        error = failed.get(stage)
        step = (
            {"stage": stage, "status": f"failed {error.status_code}", "detail": error.detail}
            if error
            else {"stage": stage, "status": "ok or current"}
        )
        result.steps.append(step)
        print_step(step)
    print(f"  job {job.status}: succeeded {job.succeeded}, skipped {job.skipped}, failed {job.failed}")
    return result


# --- printing -------------------------------------------------------------------------


def print_step(step: dict) -> None:
    extra = ""
    if "tokens" in step:
        extra = f"  {step['tokens']} tokens  {step['seconds']}s"
    print(f"  {step['stage']:<11}{step['status']}{extra}")
    for warning in step.get("warnings") or ():
        print(f"      warning: {warning}")
    if step.get("detail"):
        print(f"      {step['detail'][:600]}")


def show(doc: dict, result: Result) -> None:
    correct = correct_indices(doc)
    prompt = doc.get("prompt") or {}
    print("\nORIGINAL")
    print("  " + " ".join(clean_question(prompt.get("question", "")).split()))
    for i, answer in enumerate(answers_of(doc)):
        print(f"  {'*' if i in correct else ' '}{chr(65 + i)}. {clean_plain(answer)}")
    if result.beat:
        print(f"\nBEAT (shown before the question)\n  {result.beat}")

    d = result.decomposition
    if d is not None:
        print("\nDECOMPOSITION")
        print(f"  intent: {d.question_intent.primary} - {d.question_intent.question_task}")
        print(f"  summary: {d.question_summary}")
        for e in d.entities:
            print(f"  {e.id:<4}{e.name} ({e.type}, {e.source_status})")
        for name, facts in (
            ("symptom", d.problem.symptoms),
            ("cause", d.problem.root_causes),
            ("consequence", d.problem.consequences),
            ("wanted", d.context.desired_state),
        ):
            for f in facts:
                print(f"  {name}: {f.text} [{f.source_status}]")
        for c in d.conditions:
            print(f"  condition {c.id}: {c.description} [{c.source_status}]")
        for o in d.answer_options:
            print(f"  option {chr(65 + o.index)}: {o.mechanism}")
        for a in d.ambiguities_and_unknowns:
            print(f"  open: {a}")

    if result.parts:
        print("\nPARTS")
        for part in result.parts:
            ids = f"  [{', '.join(part.entity_ids)}]" if part.entity_ids else ""
            print(f"  {part.role:<14}{part.text}{ids}")

    if result.rephrase is not None:
        print("\nREPHRASE")
        print(f"  {result.rephrase.question}")
        for i, answer in enumerate(result.rephrase.answers):
            print(f"  {'*' if i in correct else ' '}{chr(65 + i)}. {answer}")

    entities = result.entities
    if entities is not None:
        print(f"\nENTITIES (question icon: {entities.question_icon})")
        name = lambda e: f"{e.name}<{e.icon}>" + (f"#{e.ref}" if e.ref else "")  # noqa: E731
        print("  question: " + ", ".join(name(e) for e in entities.question))
        for i, answer in enumerate(entities.answers):
            print(f"  {chr(65 + i)} <{answer.icon}>: " + ", ".join(name(e) for e in answer.entities))

    if result.flow is not None:
        v = result.flow.visualization
        print(f"\nFLOW: {v.primary} (+{', '.join(v.secondary) or 'none'}) - {v.reason}")
    built, errors = result.code_flow(doc)
    if built is not None:
        print(f"\nFLOW BUILT IN CODE (experiment, not stored): {len(built.nodes)} nodes")
        for node in built.nodes:
            print(f"  {node.id:<4}{node.type:<9}{node.label}")
        for error in errors or ["passes the flow gate"]:
            print(f"  gate: {error}")
    if result.resolution is not None:
        r = result.resolution
        print("\nRESOLUTION")
        for requirement in r.answer_requirements:
            print(f"  needs: {requirement}")
        for o in r.options:
            gaps = f" gaps: {'; '.join(o.gaps)}" if o.gaps else ""
            print(f"  {chr(65 + o.index)} {o.verdict}: {o.reason}{gaps}")
            print(f"      {o.impact}: {o.consequence or '(no consequence)'}")
        print(f"  rule: {r.rule}")
    if result.game:
        card = result.game["public"]
        print("\nGAME CARD (compose)")
        for part in card["briefing"]:
            print(f"  {part['role']:<14}{part['text']}")
        print(f"  {'ask':<14}{card['ask']['text']}")
        names = {c["ref"]: f"{c['name']}<{c['icon']}>" for c in card["cast"]}
        for tool in card["tools"]:
            parts = ", ".join(names.get(r, r) for r in tool["components"])
            print(f"  tool {chr(65 + tool['index'])} <{tool['icon']}> {tool['label']}  [{parts}]")
        for outcome in result.game["key"]["outcomes"]:
            print(f"  outcome {chr(65 + outcome['index'])} {outcome['impact']}: {outcome['consequence']}")
    elif result.game_missing:
        print(f"\nGAME CARD: not composed, missing {', '.join(result.game_missing)}")
    if result.check:
        print(f"\nCHECK: {result.check.get('verdict')}")


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="      %(name)s: %(message)s")
    # The repair notes of the steps ("... needs repair: ...") are what to read first.
    logging.getLogger("claude_exam").setLevel(logging.INFO)
    settings = get_settings()
    mongo = AsyncMongoClient(args.database_url or settings.database_url)
    ica = IcaClient(settings.ica_url, settings.ica_api_key)
    try:
        db = mongo.get_default_database()
        doc = await db["examination"].find_one({"_id": args.object_id})
        if doc is None:
            print(f"no question {args.question_id}")
            return 2
        how = "dry run, nothing is written" if args.dry_run else "real run, records are updated"
        print(f"{args.question_id} ({doc.get('section', '')}) - {how}; steps: {', '.join(args.stage_list)}")
        if args.dry_run:
            result = await dry_run(args, ica, db, doc)
        else:
            result = await real_run(args, ica, mongo, db, doc)

        if args.show:
            show(doc, result)
        if args.mermaid:
            answers = [clean_plain(a) for a in answers_of(doc)]
            if result.flow is not None:
                print("\nFLOW BEFORE ANSWERING (Mermaid)\n" + mermaid(result.flow, answers))
            if result.resolution is not None:
                print("\nRESOLVED FLOW (Mermaid)\n" + mermaid(result.resolution.flow))
            built, _errors = result.code_flow(doc)
            if built is not None:
                print("\nFLOW BUILT IN CODE, experiment (Mermaid)\n" + mermaid(built, answers))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(result.as_json(doc), handle, ensure_ascii=False, indent=1, default=str)
            print(f"\nwritten to {args.out}")
        return 1 if any(s["status"].startswith("failed") for s in result.steps) else 0
    finally:
        await ica.aclose()
        await mongo.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
