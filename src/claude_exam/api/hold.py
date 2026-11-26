"""Questions held out of the game until they are processed by hand.

An import with `manual=true` stores its questions with `pipeline_mode: "manual"` and starts
no job: the questions are processed one by one from the evaluation page. While held, a
question is not playable (`game.domain.PLAYABLE_QUERY` / `is_playable`) and the backfills
leave it alone unless asked (`include_manual`). It is released (the field removed) as soon
as a real run leaves none of its steps missing (`batch.release_if_ready`), or by
hand (`POST /evaluate/questions/{id}/release`).
"""

FIELD = "pipeline_mode"
MANUAL = "manual"
# Matches every question that is not held (also those without the field).
NOT_HELD: dict = {FIELD: {"$ne": MANUAL}}
HELD: dict = {FIELD: MANUAL}
# Steps of the "full" job a held question may still lack when it is released: none. Its
# last step composes the game document (`compose.py`), which needs every other one.
OPTIONAL_STAGES: frozenset[str] = frozenset()


def is_held(doc: dict) -> bool:
    return doc.get(FIELD) == MANUAL


def with_hold(query: dict, include_manual: bool) -> dict:
    """A backfill's question filter, without held questions unless `include_manual`."""
    return query if include_manual else {**query, **NOT_HELD}
