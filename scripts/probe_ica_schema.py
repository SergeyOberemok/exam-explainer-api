"""Does ICA honour `response_format: json_schema` (structured outputs)?

Sends one tiny request per model with a strict JSON schema whose answer the model would
not produce on its own (a fixed enum value and an integer), and prints per model whether
the request was accepted and the reply matches the schema. `ica_json.ask_json` falls back
to `json_object` for a model that rejects the format; this tells which case applies.

Found on 2026-11-25: every model accepted the format and followed this flat schema, but
with a nested schema ($defs) Sonnet answered in prose, and Haiku once left out a
required field of a large one. So ICA's structured outputs are best effort: the prompts
still spell out the object and the code gates still validate it.

Usage: uv run python scripts/probe_ica_schema.py [MODEL ...]
Without models every `IcaModelId` is probed. Costs one small ICA call per model.
"""

import asyncio
import json
import sys
from typing import get_args

from fastapi import HTTPException

from claude_exam.api.config import get_settings
from claude_exam.api.ica import IcaClient
from claude_exam.api.schemas import IcaModelId

SCHEMA = {
    "type": "object",
    "properties": {
        "colour": {"type": "string", "enum": ["teal", "ochre"]},
        "count": {"type": "integer"},
    },
    "required": ["colour", "count"],
    "additionalProperties": False,
}
FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "probe", "strict": True, "schema": SCHEMA},
}
PROMPT = "Name a colour of the sea and how many legs a spider has."


def matches(reply: dict) -> bool:
    return (
        set(reply) == {"colour", "count"}
        and reply["colour"] in ("teal", "ochre")
        and isinstance(reply["count"], int)
    )


async def probe(ica: IcaClient, model: str) -> str:
    try:
        data = await ica.chat_model(model, PROMPT, response_format=FORMAT)
    except HTTPException as exc:
        return f"REJECTED ({exc.status_code}): {str(exc.detail)[:200]}"
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        reply = json.loads(content)
    except json.JSONDecodeError:
        return f"accepted, but the reply is not bare JSON: {content[:160]!r}"
    verdict = "HONOURED" if isinstance(reply, dict) and matches(reply) else "IGNORED"
    return f"accepted, schema {verdict}: {content[:160]}"


async def main() -> None:
    settings = get_settings()
    models = sys.argv[1:] or list(get_args(IcaModelId))
    ica = IcaClient(settings.ica_url, settings.ica_api_key)
    try:
        for model in models:
            print(f"{model}: {await probe(ica, model)}")
    finally:
        await ica.aclose()


if __name__ == "__main__":
    asyncio.run(main())
