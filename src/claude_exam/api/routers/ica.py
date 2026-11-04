import json

from fastapi import APIRouter, Depends, HTTPException

from claude_exam.api.config import get_settings
from claude_exam.api.dependencies import IcaClientDep
from claude_exam.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    IcaAgentList,
    IcaAssistant,
    IcaChatRequest,
    IcaChatResponse,
)
from claude_exam.api.security import require_api_key
from claude_exam.api.simplify import simplify_item
from claude_exam.paths import REPO_ROOT

# Study/admin routes: they need the API key when one is configured.
router = APIRouter(
    prefix="/ica", tags=["ica"], dependencies=[Depends(require_api_key)]
)

_PRACTICE_EXAM_FILE = REPO_ROOT / "public" / "claude-certified-architect-practice-1.json"

# Cached item lookup: question_id → item dict
_exam_items: dict[int, dict] | None = None


def _get_exam_items() -> dict[int, dict]:
    global _exam_items
    if _exam_items is None:
        try:
            data = json.loads(_PRACTICE_EXAM_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Practice exam file not found: {_PRACTICE_EXAM_FILE}",
            )
        _exam_items = {item["id"]: item for item in data.get("results", [])}
    return _exam_items


@router.get("/assistants")
async def list_assistants(ica: IcaClientDep) -> list[IcaAssistant]:
    return [
        IcaAssistant(
            id=a["id"], name=a.get("name", ""), model=a.get("model")
        )
        for a in await ica.list_assistants()
    ]


@router.get("/agents")
async def list_agents(ica: IcaClientDep) -> IcaAgentList:
    return IcaAgentList(items=await ica.list_agents())


@router.post("/chat")
async def chat(body: IcaChatRequest, ica: IcaClientDep) -> IcaChatResponse:
    assistant = body.assistant or get_settings().ica_assistant
    if not assistant:
        raise HTTPException(
            status_code=400,
            detail="no assistant given and ICA_ASSISTANT is not set",
        )
    assistant_id = await ica.resolve_assistant_id(assistant)
    data = await ica.chat(assistant_id, body.prompt, body.max_tokens)
    choice = (data.get("choices") or [{}])[0]
    return IcaChatResponse(
        reply=(choice.get("message") or {}).get("content") or "",
        finish_reason=choice.get("finish_reason"),
        total_tokens=(data.get("usage") or {}).get("total_tokens", 0),
        assistant_id=assistant_id,
    )


@router.post("/explain", response_model_by_alias=True)
async def explain(body: ExplainRequest, ica: IcaClientDep) -> ExplainResponse:
    """
    Single-pass simplification pipeline:
    Calls /chat-models/chat/completions directly on claude-haiku-5-5 with the
    full Exam Explainer Agent persona baked into the system message, plus a JSON
    output schema appended — one call, one structured JSON response.
    """
    items = _get_exam_items()
    item = items.get(body.question_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question id {body.question_id!r} not found in practice exam file",
        )

    simplified_prompt, meta = await simplify_item(
        ica, item, body.model_id, body.max_tokens
    )

    return ExplainResponse(
        class_=item.get("_class", "assessment"),
        id=item["id"],
        assessment_type=item.get("assessment_type", "multiple-choice"),
        prompt=simplified_prompt,
        correct_response=item.get("correct_response", []),
        section=item.get("section", ""),
        question_plain=simplified_prompt.question,
        related_lectures=item.get("related_lectures", []),
        ica=meta,
    )
