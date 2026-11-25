"""The input key of a pipeline step: what a stored result was written from.

A step's key is a hash of the exact messages it sends to the model (system and user, as
passed to `ica_json.ask_json` / `ask_validated`) plus its module's `CODE_VERSION`, for
what the messages do not show (the gates, parsing, pruning, derived entities). A stored
result is current while its key equals the key of the messages the step would send
today (`pipeline_keys.is_current`), so a changed prompt, catalog, chapter or upstream
result makes it outdated without a version to bump. Not in the key: the model id, the
reply format (`ica_json` may fall back from a strict schema) and repair notes (they
depend on a reply).
"""

import hashlib
import json

from fastapi import HTTPException


def step_key(code_version: str, *parts: str) -> str:
    payload = json.dumps([code_version, *parts], ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def key_or_none(build) -> str | None:
    """`build()`'s key, or None when the step cannot run on these inputs (its message
    builder refuses them with an HTTPException, as the step itself would)."""
    try:
        return build()
    except HTTPException:
        return None
