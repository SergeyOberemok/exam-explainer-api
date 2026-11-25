"""Run the ingest pipeline on ONE question and see what each step made of it.

The way to try a prompt or gate change on a real question before a backfill, or to redo
one question by hand.

- Default (a real run): starts a job for that one question exactly as
  `POST /examination/pipeline/backfill` would (`batch.run_simplification_job`, same code
  path, visible under `GET /examination/jobs/{job_id}`), so it UPDATES the question's
  records in the database: its simplified record, decomposition, flow, resolution,
  objectives and story, as far as the chosen steps go. Without `--force` a step whose
  result is current is skipped (`only_missing`).
- `--dry-run`: calls the steps' compute functions only, feeding each step's result to the
  next in memory, and writes NOTHING. A step that is not run uses what is stored
  (`--stages flow` draws the flow of the stored decomposition). The objectives and story
  steps have no dry form: their stored records are used.

Usage:
  uv run python scripts/run_pipeline.py QUESTION_ID [DATABASE_URL]
      [--stages decompose,simplify,flow] [--mode full] [--model claude-haiku-5-5]
      [--force] [--dry-run] [--show] [--mermaid] [--out FILE]

  --stages    comma-separated steps, run in the given order (default: those of --mode)
  --mode      a job mode (`batch.STAGES`), default "full"
  --force     a real run redoes every chosen step, current or not
  --show      print the question next to its game text (parts with their roles), the
              entities with their icons, the decomposition and the resolution
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
import time
from typing import get_args

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException
from pymongo import AsyncMongoClient

from claude_exam.api import batch
from claude_exam.api.check import check_item
from claude_exam.api.config import get_settings
from claude_exam.api.decompose import decompose_item, read_decomposition
from claude_exam.api.entities import derive_entities, extract_entities
from claude_exam.api.flow import (
    flow_is_current,
    flow_item,
    leak_warning,
    mermaid,
    resolution_is_current,
    resolve_item,
)
from claude_exam.api.game.domain import answers_of, correct_indices, story_of
from claude_exam.api.ica import IcaClient
from claude_exam.api.icons import pick_icons
from claude_exam.api.schemas import (
    DecisionFlow,
    IcaModelId,
    JobMode,
    JobStage,
    PromptEntities,
    Resolution,
    SimplificationJob,
    SimplifiedPrompt,
)
from claude_exam.api.simplify import quality_warnings, simplify_item
from claude_exam.html_clean import clean_plain, clean_question

DRY_STAGES = ("decompose", "simplify", "entities", "icons", "flow", "resolve", "check")


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


# --- what a question has, read from the database or built in a dry run -----------------


class Result:
    """The pieces the steps make, whichever way they were made."""

    def __init__(self) -> None:
        self.decomposition = None
        self.prompt: SimplifiedPrompt | None = None
        self.entities: PromptEntities | None = None
        self.flow: DecisionFlow | None = None
        self.resolution: Resolution | None = None
        self.check: dict | None = None
        self.steps: list[dict] = []

    @classmethod
    def from_records(cls, doc: dict, simplified: dict | None, decomposition: dict | None) -> "Result":
        result = cls()
        result.decomposition = read_decomposition(decomposition, doc)
        if simplified:
            result.prompt = SimplifiedPrompt(**simplified["prompt"])
            try:
                result.entities = PromptEntities.model_validate(simplified.get("entities") or {})
            except ValueError:
                result.entities = None
            result.check = simplified.get("check")
        if result.decomposition is not None and flow_is_current(decomposition):
            result.flow = DecisionFlow.model_validate(decomposition["flow"])
        if result.decomposition is not None and resolution_is_current(decomposition):
            result.resolution = Resolution.model_validate(decomposition["resolution"])
        return result

    def as_json(self) -> dict:
        dump = lambda m: m.model_dump() if m is not None else None  # noqa: E731
        return {
            "steps": self.steps,
            "decomposition": dump(self.decomposition),
            "prompt": dump(self.prompt),
            "entities": dump(self.entities),
            "flow": dump(self.flow),
            "resolution": dump(self.resolution),
            "check": self.check,
        }


async def dry_run(args, ica: IcaClient, db, doc: dict) -> Result:
    question_id = str(doc["_id"])
    result = Result.from_records(
        doc,
        await db["examination_simplified"].find_one({"question_id": question_id}),
        await db["question_decomposition"].find_one({"question_id": question_id}),
    )
    story = story_of(doc, await db["question_story"].find_one({"question_id": question_id}))
    model = args.model

    async def run(stage: str) -> None:
        if stage == "decompose":
            result.decomposition, meta = await decompose_item(ica, doc, model)
            result.flow = result.resolution = None
            return meta
        if stage == "simplify":
            result.prompt, meta = await simplify_item(
                ica, doc, model, story=story, decomposition=result.decomposition
            )
            for warning in quality_warnings(doc, result.prompt.model_dump()):
                print(f"      warning: {warning}")
            return meta
        if stage == "entities":
            need(result.prompt, "a game text")
            if result.decomposition is not None:
                result.entities = derive_entities(result.decomposition, doc, result.prompt)
                return None
            result.entities, meta = await extract_entities(ica, doc, result.prompt, model)
            return meta
        if stage == "icons":
            need(result.prompt, "a game text")
            result.entities, meta = await pick_icons(
                ica, doc.get("section", ""), result.prompt, result.entities or PromptEntities(), model
            )
            return meta
        if stage == "flow":
            need(result.decomposition, "a decomposition")
            result.flow, meta = await flow_item(ica, doc, result.decomposition, model)
            result.resolution = None
            warning = leak_warning(doc, result.flow)
            if warning:
                print(f"      warning: {warning}")
            return meta
        if stage == "resolve":
            need(result.flow, "a flow")
            result.resolution, meta = await resolve_item(ica, doc, result.decomposition, result.flow, model)
            return meta
        if stage == "check":
            need(result.prompt, "a game text")
            result.check = await check_item(ica, doc, {"prompt": result.prompt.model_dump()}, model, story=story)
            return None
        raise HTTPException(status_code=400, detail="no dry run for this step: its stored record is used")

    for stage in args.stage_list:
        if stage not in DRY_STAGES:
            step = {"stage": stage, "status": "skipped (no dry form; the stored record is used)"}
            result.steps.append(step)
            print_step(step)
            continue
        started = time.monotonic()
        try:
            meta = await run(stage)
            step = {"stage": stage, "status": "ok", "tokens": meta.total_tokens if meta else 0}
        except HTTPException as exc:
            step = {"stage": stage, "status": f"failed {exc.status_code}", "detail": str(exc.detail)}
        step["seconds"] = round(time.monotonic() - started, 1)
        result.steps.append(step)
        print_step(step)
    return result


def need(value, what: str) -> None:
    if value is None:
        raise HTTPException(status_code=409, detail=f"needs {what}: run that step first (or store one)")


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
    )
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
    if step.get("detail"):
        print(f"      {step['detail'][:600]}")


def show(doc: dict, result: Result) -> None:
    correct = correct_indices(doc)
    prompt = doc.get("prompt") or {}
    print("\nORIGINAL")
    print("  " + " ".join(clean_question(prompt.get("question", "")).split()))
    for i, answer in enumerate(answers_of(doc)):
        print(f"  {'*' if i in correct else ' '}{chr(65 + i)}. {clean_plain(answer)}")

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

    if result.prompt is not None:
        print("\nGAME TEXT")
        if result.prompt.parts:
            for part in result.prompt.parts:
                ids = f"  [{', '.join(part.entity_ids)}]" if part.entity_ids else ""
                print(f"  {part.role:<14}{part.text}{ids}")
        else:
            print(f"  {result.prompt.question}  (no parts)")
        for i, answer in enumerate(result.prompt.answers):
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
    if result.resolution is not None:
        r = result.resolution
        print("\nRESOLUTION")
        for requirement in r.answer_requirements:
            print(f"  needs: {requirement}")
        for o in r.options:
            gaps = f" gaps: {'; '.join(o.gaps)}" if o.gaps else ""
            print(f"  {chr(65 + o.index)} {o.verdict}: {o.reason}{gaps}")
        print(f"  rule: {r.rule}")
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
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(result.as_json(), handle, ensure_ascii=False, indent=1, default=str)
            print(f"\nwritten to {args.out}")
        return 1 if any(s["status"].startswith("failed") for s in result.steps) else 0
    finally:
        await ica.aclose()
        await mongo.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
