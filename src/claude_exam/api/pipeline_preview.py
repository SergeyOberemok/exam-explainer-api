"""A dry run of the ingest pipeline on ONE question: the steps' compute functions are
chained in memory, each step's result feeding the next, and NOTHING is written.

Used by `scripts/run_pipeline.py --dry-run` and the prompt evaluation page
(`routers/evaluate.py`). A step that is not run uses what is stored (`stages=("flow",)`
draws the flow of the stored decomposition). The objectives and story steps have no dry
form: their stored records are used (the beat is written on the stored story's stage).
A dry decomposition also builds its parts and entities, as storing it would.

With a decomposition the result also carries `code_flow`, an experiment: the pre-answer
flow built in code (`flow.flow_from`) with what the flow gate says about it, shown next
to the model's flow and never stored.
"""

import time
from collections.abc import AsyncIterator, Sequence

from fastapi import HTTPException

from claude_exam.api.beat import beat_item
from claude_exam.api.check import check_item
from claude_exam.api.decompose import (
    decompose_item,
    has_flow,
    has_resolution,
    parts_from,
    read_decomposition,
)
from claude_exam.api.entities import derive_entities
from claude_exam.api.flow import (
    flow_errors,
    flow_from,
    flow_item,
    leak_warning,
    mermaid,
    resolve_item,
)
from claude_exam.api.game.domain import answers_of, correct_indices, story_of
from claude_exam.api.ica import IcaClient
from claude_exam.api.icons import pick_icons, stored_entities
from claude_exam.api.question_view import key_free
from claude_exam.api.rephrase import quality_warnings, rephrase_item
from claude_exam.api.scenario_catalog import scenario_of, split_scenario, stem_of
from claude_exam.api.schemas import (
    DecisionFlow,
    PromptEntities,
    Rephrase,
    Resolution,
    SemanticPart,
)
from claude_exam.html_clean import clean_plain, clean_question

DRY_STAGES = ("decompose", "beat", "icons", "flow", "resolve", "rephrase", "check")


class Result:
    """The pieces the steps make, whichever way they were made."""

    def __init__(self) -> None:
        self.decomposition = None
        self.parts: list[SemanticPart] = []
        self.entities: PromptEntities | None = None
        self.flow: DecisionFlow | None = None
        self.resolution: Resolution | None = None
        self.rephrase: Rephrase | None = None
        self.check: dict | None = None
        # The beat written by a dry run (None: the stored story's beat is used).
        self.beat: str | None = None
        self.steps: list[dict] = []

    def code_flow(self, doc: dict) -> tuple[DecisionFlow | None, list[str]]:
        """The experiment: the flow built in code and the flow gate's verdict on it."""
        if self.decomposition is None:
            return None, []
        built = flow_from(self.decomposition, doc)
        return built, flow_errors(doc, self.decomposition, built.model_dump())

    @classmethod
    def from_records(cls, doc: dict, simplified: dict | None, decomposition: dict | None) -> "Result":
        result = cls()
        result.decomposition = read_decomposition(decomposition, doc)
        if result.decomposition is not None:
            try:
                result.parts = [SemanticPart.model_validate(p) for p in decomposition.get("parts") or []]
            except ValueError:
                result.parts = []
            result.entities = stored_entities(decomposition)
            if has_flow(decomposition):
                result.flow = DecisionFlow.model_validate(decomposition["flow"])
            if has_resolution(decomposition):
                result.resolution = Resolution.model_validate(decomposition["resolution"])
        if simplified:
            try:
                result.rephrase = Rephrase(**simplified["prompt"])
            except (TypeError, ValueError):
                result.rephrase = None
            result.check = simplified.get("check")
        return result

    def as_json(self, doc: dict | None = None) -> dict:
        """Everything as plain JSON; with the question `doc`, also the flows as Mermaid."""
        dump = lambda m: m.model_dump(mode="json") if m is not None else None  # noqa: E731
        out = {
            "steps": self.steps,
            "decomposition": dump(self.decomposition),
            "parts": [p.model_dump(mode="json") for p in self.parts],
            "entities": dump(self.entities),
            "flow": dump(self.flow),
            "resolution": dump(self.resolution),
            "rephrase": dump(self.rephrase),
            "check": _plain(self.check),
            "beat": self.beat,
        }
        if doc is not None:
            answers = [clean_plain(a) for a in answers_of(doc)]
            built, errors = self.code_flow(doc)
            out["code_flow"] = dump(built)
            out["code_flow_errors"] = errors
            out["mermaid"] = {
                "flow": mermaid(self.flow, answers) if self.flow is not None else "",
                "resolution": mermaid(self.resolution.flow) if self.resolution is not None else "",
                "code_flow": mermaid(built, answers) if built is not None else "",
            }
        return out


def need(value, what: str) -> None:
    if value is None:
        raise HTTPException(status_code=409, detail=f"needs {what}: run that step first (or store one)")


async def run_dry(
    ica: IcaClient,
    doc: dict,
    result: Result,
    story_record: dict | None,
    stages: Sequence[str],
    model: str,
) -> AsyncIterator[dict]:
    """Runs `stages` in order on `result` (the stored records, `Result.from_records`) and
    yields one step dict per stage: `stage`, `status` ("ok", "failed <code>" or
    "skipped ..."), `tokens`, `seconds`, `warnings` and, on a failure, `detail`. A failed
    step does not stop the run; the steps that need its result fail with 409."""
    story = story_of(doc, story_record)
    q = key_free(doc)

    async def run(stage: str, warnings: list[str]):
        if stage == "decompose":
            result.decomposition, meta = await decompose_item(ica, doc, model)
            result.parts = parts_from(result.decomposition, q)
            result.entities = derive_entities(result.decomposition, q)
            result.flow = result.resolution = None
            return meta
        if stage == "beat":
            result.beat, meta = await beat_item(
                ica, doc, story, model, decomposition=result.decomposition
            )
            # The rephrase and the check are set after it, as in a job.
            story["beat"] = result.beat
            return meta
        if stage == "icons":
            need(result.decomposition, "a decomposition")
            result.entities, meta = await pick_icons(
                ica, q, result.entities or PromptEntities(), model
            )
            return meta
        if stage == "flow":
            need(result.decomposition, "a decomposition")
            result.flow, meta = await flow_item(ica, doc, result.decomposition, model)
            result.resolution = None
            warning = leak_warning(doc, result.flow)
            if warning:
                warnings.append(warning)
            return meta
        if stage == "resolve":
            need(result.flow, "a flow")
            result.resolution, meta = await resolve_item(ica, doc, result.decomposition, result.flow, model)
            return meta
        if stage == "rephrase":
            result.rephrase, meta = await rephrase_item(
                ica, doc, model, place=story, decomposition=result.decomposition
            )
            result.check = None
            warnings.extend(quality_warnings(q, result.rephrase.model_dump()))
            return meta
        if stage == "check":
            need(result.rephrase, "a rephrase")
            result.check = await check_item(
                ica, doc, {"prompt": result.rephrase.model_dump()}, model, story=story
            )
            return None
        raise HTTPException(status_code=400, detail="no dry run for this step: its stored record is used")

    for stage in stages:
        if stage not in DRY_STAGES:
            step = {"stage": stage, "status": "skipped (no dry form; the stored record is used)", "warnings": []}
            result.steps.append(step)
            yield step
            continue
        started = time.monotonic()
        warnings: list[str] = []
        try:
            meta = await run(stage, warnings)
            step = {"stage": stage, "status": "ok", "tokens": meta.total_tokens if meta else 0}
        except HTTPException as exc:
            step = {"stage": stage, "status": f"failed {exc.status_code}", "detail": str(exc.detail)}
        step["seconds"] = round(time.monotonic() - started, 1)
        step["warnings"] = warnings
        result.steps.append(step)
        yield step


def original_view(doc: dict, story_record: dict | None, objectives_record: dict | None) -> dict:
    """The original question as the evaluation page shows it, with its answer key."""
    prompt = doc.get("prompt") or {}
    correct = set(correct_indices(doc))
    chapter = scenario_of(doc)
    feedbacks = prompt.get("feedbacks") or []
    return {
        "id": str(doc["_id"]),
        "section": doc.get("section", ""),
        "assessment_type": doc.get("assessment_type", ""),
        "source": doc.get("source", ""),
        "chapter": chapter.title if chapter else "",
        "scenario": " ".join(clean_question(split_scenario(prompt.get("question"))[0]).split()),
        "stem": " ".join(clean_question(stem_of(doc)).split()),
        "question": " ".join(clean_question(prompt.get("question", "")).split()),
        "answers": [
            {
                "index": i,
                "text": clean_plain(answer),
                "correct": i in correct,
                "feedback": clean_plain(feedbacks[i]) if i < len(feedbacks) else "",
            }
            for i, answer in enumerate(answers_of(doc))
        ],
        "explanation": clean_plain(prompt.get("explanation", "")),
        "story": _record(story_record),
        "objectives": _record(objectives_record),
    }


def _record(record: dict | None) -> dict | None:
    if not record:
        return None
    return _plain({k: v for k, v in record.items() if k not in ("_id", "ica")})


def _plain(value):
    """A Mongo record as JSON-safe values (ObjectIds and datetimes as strings)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
