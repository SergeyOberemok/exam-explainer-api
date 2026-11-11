"""One ICA chat call that must answer with a JSON object.

Every pipeline step (game text, entities, icons, objectives, story, check) sends a system
message and one user message and reads one JSON object back. This is that call: it
rejects a reply the model did not finish and parses the rest, so a step only has to
validate what the object says.
"""

from fastapi import HTTPException

from claude_exam.api.ica import IcaClient
from claude_exam.api.json_reply import extract_json
from claude_exam.api.schemas import IcaMeta, IcaModelId


async def ask_json(
    ica: IcaClient,
    model_id: IcaModelId,
    user: str,
    *,
    system: str,
    what: str,
    max_tokens: int | None = None,
) -> tuple[dict, IcaMeta]:
    """
    Send `system` and `user` to the ICA chat model and return the JSON object of its
    reply with the call's metadata. `what` names the result in the error ("the story").
    Raises 502 (which a job retries) when the model stopped early or the reply holds no
    JSON object.
    """
    # `json_object` keeps fences and prose out of the reply; `extract_json` still copes
    # with them, for a gateway or model that ignores the flag.
    data = await ica.chat_model(
        model_id, user, system=system, max_tokens=max_tokens, json_object=True
    )
    choice = (data.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        raise HTTPException(
            status_code=502,
            detail=f"ICA model stopped with finish_reason={finish_reason!r} — cannot parse {what}",
        )
    parsed = extract_json((choice.get("message") or {}).get("content") or "")
    return parsed, IcaMeta(
        model_id=model_id,
        finish_reason=finish_reason,
        total_tokens=(data.get("usage") or {}).get("total_tokens", 0),
    )
