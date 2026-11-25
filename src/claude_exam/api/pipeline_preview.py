"""A dry run of the ingest pipeline on ONE question: the steps' compute functions are
chained in memory, each step's result feeding the next, and NOTHING is written.

Used by `scripts/run_pipeline.py --dry-run` and the prompt evaluation page
(`routers/evaluate.py`). A step that is not run uses what is stored (`stages=("flow",)`
draws the flow of the stored decomposition). The objectives and story steps have no dry
form: their stored records are used.
"""

import time
from collections.abc import AsyncIterator, Sequence

from fastapi import HTTPException

from claude_exam.api.check import check_item
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
from claude_exam.api.scenario_catalog import scenario_of, split_scenario, stem_of
from claude_exam.api.schemas import (
    DecisionFlow,
    PromptEntities,
    Resolution,
    SimplifiedPrompt,
)
from claude_exam.api.simplify import quality_warnings, simplify_item
from claude_exam.html_clean import clean_plain, clean_question

DRY_STAGES = ("decompose", "simplify", "entities", "icons", "flow", "resolve", "check")


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

    def as_json(self, doc: dict | None = None) -> dict:
        """Everything as plain JSON; with the question `doc`, also the flows as Mermaid."""
        dump = lambda m: m.model_dump(mode="json") if m is not None else None  # noqa: E731
        out = {
            "steps": self.steps,
            "decomposition": dump(self.decomposition),
            "prompt": dump(self.prompt),
            "entities": dump(self.entities),
            "flow": dump(self.flow),
            "resolution": dump(self.resolution),
            "check": _plain(self.check),
        }
        if doc is not None:
            answers = [clean_plain(a) for a in answers_of(doc)]
            out["mermaid"] = {
                "flow": mermaid(self.flow, answers) if self.flow is not None else "",
                "resolution": mermaid(self.resolution.flow) if self.resolution is not None else "",
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

    async def run(stage: str, warnings: list[str]):
        if stage == "decompose":
            result.decomposition, meta = await decompose_item(ica, doc, model)
            result.flow = result.resolution = None
            return meta
        if stage == "simplify":
            result.prompt, meta = await simplify_item(
                ica, doc, model, story=story, decomposition=result.decomposition
            )
            warnings.extend(quality_warnings(doc, result.prompt.model_dump()))
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
                warnings.append(warning)
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
