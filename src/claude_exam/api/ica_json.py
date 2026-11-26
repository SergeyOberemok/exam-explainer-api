"""One ICA chat call that must answer with a JSON object.

Every pipeline step (objectives, story, decomposition, icons, flow, resolution, beat)
sends a system message and one user message and reads one JSON object back. This is that call: it rejects a reply the model did not finish and parses the rest,
so a step only has to validate what the object says.

Two levels:
- `ask_json`: one call. With `schema` (a Pydantic model) the gateway is asked for a
  strict `json_schema` reply (structured outputs). ICA accepts the format, but honours
  it only as best effort (`scripts/probe_ica_schema.py`: one probe was followed by every
  model, a nested one was ignored by Sonnet, and Haiku once left out a required field),
  so a prompt still describes the object (`schema_skeleton`) and the caller still
  validates it. A model or gateway that rejects the format is asked again with
  `json_object`, and remembered.
- `ask_validated`: the call plus the repair loop. A reply that `validate` finds fault
  with is sent back with those faults (`repair_note`), up to `max_attempts` replies;
  after that the call raises 502, which a job retries.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from claude_exam.api.ica import IcaClient
from claude_exam.api.json_reply import extract_json
from claude_exam.api.schemas import IcaMeta, IcaModelId

logger = logging.getLogger(__name__)

# Models whose gateway refused a `json_schema` response format in this process.
_NO_SCHEMA: set[str] = set()
# JSON-Schema keywords the structured-outputs grammar does not take; the code gates
# check those limits instead.
_UNSUPPORTED = frozenset(
    {
        "default",
        "title",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "examples",
    }
)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema of `model` in the strict form structured outputs need: every
    object closed (`additionalProperties: false`) with all its properties required
    (an optional field stays nullable, so the model writes null), and no keyword the
    grammar does not support."""

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if out.get("minItems", 0) > 1:
            out["minItems"] = 1
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(model.model_json_schema())


def schema_skeleton(model: type[BaseModel]) -> str:
    """The shape of `model` as a JSON skeleton for a prompt: every field with a
    placeholder ("<string>", "a|b|c" for a choice, [item] for a list), nested objects
    written out. ICA's handling of `json_schema` is best effort (a probe found one model
    ignoring it), so the prompt has to describe the object too; generated from the
    model, it cannot drift from what the code validates."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def render(node: dict, indent: int) -> str:
        if "$ref" in node:
            return render(defs[node["$ref"].split("/")[-1]], indent)
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            text = render(options[0], indent) if options else "null"
            return f"{text} or null" if len(options) < len(node["anyOf"]) else text
        if "enum" in node:
            return '"' + "|".join(str(v) for v in node["enum"]) + '"'
        kind = node.get("type")
        pad = "  " * (indent + 1)
        if kind == "object" and "properties" in node:
            fields = ",\n".join(
                f'{pad}"{name}": {render(value, indent + 1)}'
                for name, value in node["properties"].items()
            )
            return "{\n" + fields + "\n" + "  " * indent + "}"
        if kind == "array":
            return "[" + render(node.get("items", {}), indent) + ", ...]"
        return {"string": '"<string>"', "integer": "<integer>", "boolean": "true|false"}.get(
            kind, f"<{kind}>"
        )

    return render(schema, 0)


def _response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": strict_schema(schema),
        },
    }


def _format_refused(exc: HTTPException) -> bool:
    detail = str(exc.detail).lower()
    return exc.status_code == 502 and any(
        word in detail for word in ("response_format", "json_schema", "output_format")
    )


async def ask_json(
    ica: IcaClient,
    model_id: IcaModelId,
    user: str,
    *,
    system: str,
    what: str,
    max_tokens: int | None = None,
    schema: type[BaseModel] | None = None,
) -> tuple[dict, IcaMeta]:
    """
    Send `system` and `user` to the ICA chat model and return the JSON object of its
    reply with the call's metadata. `what` names the result in the error ("the story").
    `schema` asks for a reply of that model's shape (see the module doc); the object is
    returned as parsed JSON either way, so validating it stays with the caller.
    Raises 502 (which a job retries) when the model stopped early or the reply holds no
    JSON object.
    """
    data = None
    if schema is not None and model_id not in _NO_SCHEMA:
        try:
            data = await ica.chat_model(
                model_id,
                user,
                system=system,
                max_tokens=max_tokens,
                response_format=_response_format(schema),
            )
        except HTTPException as exc:
            if not _format_refused(exc):
                raise
            logger.warning("%s refused a json_schema reply; using json_object", model_id)
            _NO_SCHEMA.add(model_id)
    if data is None:
        # `json_object` keeps fences and prose out of the reply; `extract_json` still
        # copes with them, for a gateway or model that ignores the flag.
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


def repair_note(previous: dict, errors: list[str]) -> str:
    """What the model gets after a rejected reply: that reply and what is wrong with
    it. Without its own text it writes a new one and loses something else."""
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        "\n\nYOUR PREVIOUS REPLY:\n"
        f"{json.dumps(previous, ensure_ascii=False, indent=1)}\n\n"
        "IT WAS REJECTED FOR THESE REASONS ONLY:\n"
        f"{listed}\n"
        "Reply with the same JSON object, changing only the places named above and "
        "leaving every other sentence exactly as it is. Copy an exact term letter "
        "for letter into the place it belongs."
    )


async def ask_validated(
    ica: IcaClient,
    model_id: IcaModelId,
    user: str,
    *,
    system: str,
    what: str,
    validate: Callable[[dict], list[str]],
    max_attempts: int = 3,
    max_tokens: int | None = None,
    schema: type[BaseModel] | None = None,
    label: object = None,
    transform: Callable[[dict], dict] | None = None,
) -> tuple[dict, IcaMeta]:
    """
    `ask_json`, then `transform(parsed)` (a repair made in code, such as cutting what
    may not be there), then `validate` on the result: an empty list accepts the reply; a non-empty
    one (sentences the model can act on) sends it back with them, up to `max_attempts`
    replies in all, after which this raises 502 with the last faults. `validate` may
    itself raise (a reply of the wrong shape is not worth repairing). The returned
    metadata counts the tokens of all attempts. `label` names the question in the log.
    The transformed reply is what is returned and what a repair note shows, so the
    model is never asked to fix what the code already removed.
    """
    errors: list[str] = []
    parsed: dict = {}
    tokens = 0
    for attempt in range(1, max_attempts + 1):
        parsed, meta = await ask_json(
            ica,
            model_id,
            user + (repair_note(parsed, errors) if errors else ""),
            system=system,
            what=what,
            max_tokens=max_tokens,
            schema=schema,
        )
        tokens += meta.total_tokens
        if transform is not None:
            parsed = transform(parsed)
        errors = validate(parsed)
        if not errors:
            meta.total_tokens = tokens
            meta.attempts = attempt
            return parsed, meta
        logger.info("%s of %s needs repair: %s", what, label, "; ".join(errors))
    raise HTTPException(
        status_code=502,
        detail=f"Exam Explainer output rejected: {'; '.join(errors)}",
    )
