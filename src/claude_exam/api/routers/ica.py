from fastapi import APIRouter, Depends, HTTPException

from claude_exam.api.config import get_settings
from claude_exam.api.dependencies import IcaClientDep
from claude_exam.api.schemas import (
    IcaAgentList,
    IcaAssistant,
    IcaChatRequest,
    IcaChatResponse,
)
from claude_exam.api.security import require_api_key

# Study/admin routes: they need the API key when one is configured.
router = APIRouter(
    prefix="/ica", tags=["ica"], dependencies=[Depends(require_api_key)]
)


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
