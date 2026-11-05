import json
import re

from fastapi import HTTPException


def extract_json(raw: str) -> dict:
    """
    Parse an ICA model reply as a JSON object.
    Strips an accidental markdown fence, then falls back to the span from the
    first "{" to the last "}" if prose surrounds the object.
    Raises HTTPException 502 if no JSON object can be parsed.
    """
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        try:
            if start == -1 or end <= start:
                raise exc
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail=f"ICA model returned non-JSON output: {exc}. Raw: {raw[:300]}",
            ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail=f"ICA model returned {type(parsed).__name__}, expected a JSON object",
        )
    return parsed
