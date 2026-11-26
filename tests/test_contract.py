"""The wire contract with the Unity game, pinned in files.

- `tests/contract/*.json`: one real response per type the game parses, built from fixed
  data. The Unity EditMode tests parse a copy of the same files
  (`../exam-cards-game/Assets/Tests/EditMode/Contract/`), so a change here that the game
  can't read fails there too.
- `tests/contract/icon_catalog.json`: every icon key the API can send; the game's tests
  check that it ships a picture for each.
- `tests/contract/openapi_game.json`: the OpenAPI schemas and paths of the game routes;
  `openapi_game_v2.json` the same for `/v2/game` (composed questions, `compose.py`),
  whose responses are pinned as `v2_*.json`.

After an intended change, regenerate both copies and commit them in both repos:

    UPDATE_CONTRACT=1 uv run --with pytest pytest tests/test_contract.py
"""

import json
import os
from pathlib import Path

import pytest
from bson import ObjectId
from fakes import FakeCollection, FakeRounds
from fastapi.testclient import TestClient

from test_jobs import decomposition_reply, flow_reply, resolution_reply

from claude_exam.api import compose
from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_decomposition_collection,
    get_async_game_question_collection,
    get_async_objectives_collection,
    get_async_rounds_collection,
    get_async_simplified_collection,
    get_async_story_collection,
)
from claude_exam.api.icon_catalog import ICONS
from claude_exam.api.main import app
from claude_exam.api.scenario_catalog import text_hash

CONTRACT_DIR = Path(__file__).parent / "contract"
UNITY_CONTRACT_DIR = (
    Path(__file__).resolve().parents[2] / "exam-cards-game" / "Assets" / "Tests" / "EditMode" / "Contract"
)
UPDATE = os.environ.get("UPDATE_CONTRACT") == "1"
ROUND_ID = "65f000000000000000000001"

Q1 = ObjectId("65f0000000000000000000a1")
Q2 = ObjectId("65f0000000000000000000a2")
Q3 = ObjectId("65f0000000000000000000a3")


def question(oid, text, answers, correct, **kw):
    return {
        "_id": oid,
        "section": "Tools and MCPs",
        "assessment_type": "multiple-choice",
        "correct_response": [correct],
        "prompt": {
            "question": text,
            "answers": answers,
            "feedbacks": [f"Feedback for {a}." for a in answers],
            "explanation": f"Why: {text}",
        },
        **kw,
    }


EXAM = [
    question(
        Q1,
        "SCENARIO: You are building a customer support resolution agent.\n\n"
        "QUESTION: Which header carries the API key?",
        ["Authorization", "x-api-key", "X-Key"],
        "b",
    ),
    question(Q2, "What caches prompts?", ["cache_control", "retries", "batches", "tools"], "a"),
    question(Q3, "Which <tag> is literal?", ["<example>", "<b>", "none"], "a"),
]
# Q1 has a rephrase (question and answers only; the review stays the original's). The
# game serves only questions with a rephrase record, so Q2 and Q3 have one too, but the
# check found it `changed`: the game plays their original and offers no rephrase.
SIMPLE = [
    {
        **EXAM[0],
        "_id": ObjectId("65f0000000000000000000b1"),
        "question_id": str(Q1),
        "prompt": {"question": "Where does the key go?", "answers": EXAM[0]["prompt"]["answers"]},
    },
    *(
        {**q, "_id": ObjectId(), "question_id": str(q["_id"]), "check": {"verdict": "changed"}}
        for q in EXAM[1:]
    ),
]

# Q1 is decomposed: its parts and entities (with icons) are served on the question and,
# the entities, on its rephrase. The second answer and the last entity have no icon yet,
# and "retired-key" is no longer in the catalog: all three are served as null. It has no
# flow, so the flow is empty.
DECOMPOSITION = [
    {
        "_id": ObjectId("65f0000000000000000000e1"),
        "question_id": str(Q1),
        "text_hash": text_hash(EXAM[0]),
        "parts": [
            {"role": "context", "text": "A support agent calls the API.", "entity_ids": ["e1"]},
            {"role": "ask", "text": "Which header carries the API key?", "entity_ids": ["e1"]},
        ],
        "entities": {
            "question_icon": "credential",
            "question": [
                {"name": "API Key", "kind": "object", "aliases": ["key"], "ref": "e1", "icon": "credential"},
                {"name": "Send Request", "kind": "action", "aliases": [], "icon": "send"},
            ],
            "answers": [
                {
                    "icon": "permission",
                    "entities": [
                        {"name": "Authorization", "kind": "object", "aliases": ["auth header"], "icon": "retired-key"}
                    ],
                },
                {"entities": [{"name": "x-api-key", "kind": "object", "aliases": []}]},
                {"icon": "tags", "entities": []},
            ],
        },
    }
]


# Q1 is tagged with learning objectives; the others are not ([] and 0, and no
# misconception). "retired-objective" is no longer in the catalog and is not served.
TAGS = [
    {
        "_id": ObjectId("65f0000000000000000000c1"),
        "question_id": str(Q1),
        "objectives": ["tools-mcp-config", "retired-objective", "tools-errors"],
        "difficulty": 2,
        "distractors": [
            {"confused_with": None, "misconception": "Any HTTP auth header carries the key."},
            None,
            {"confused_with": "tools-descriptions", "misconception": "Header names are free to choose."},
        ],
    }
]


# Q1 belongs to the "support" chapter (its scenario paragraph) and has a story record;
# the others have no chapter and no story.
STORY = [
    {
        "_id": ObjectId("65f0000000000000000000d1"),
        "question_id": str(Q1),
        "text_hash": text_hash(EXAM[0]),
        "scenario": "support",
        "stage": "build",
        "order": 1,
        "beat": "The first API call of the support agent is refused.",
        "win": "The agent reaches the API.",
        "fail": "Every call still comes back unauthorized.",
        "origin": "seed",
    }
]


@pytest.fixture
def client():
    rounds = FakeRounds()
    app.dependency_overrides[get_async_story_collection] = lambda: FakeCollection(STORY)
    app.dependency_overrides[get_async_objectives_collection] = lambda: FakeCollection(TAGS)
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(EXAM)
    app.dependency_overrides[get_async_simplified_collection] = lambda: FakeCollection(SIMPLE)
    app.dependency_overrides[get_async_decomposition_collection] = lambda: FakeCollection(DECOMPOSITION)
    app.dependency_overrides[get_async_rounds_collection] = lambda: rounds
    return TestClient(app)


def check(name: str, body: dict) -> None:
    text = json.dumps(body, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    targets = [CONTRACT_DIR / name]
    if UNITY_CONTRACT_DIR.parent.is_dir():
        targets.append(UNITY_CONTRACT_DIR / name)
    for path in targets:
        if UPDATE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
        else:
            assert path.is_file(), f"{path} is missing; run with UPDATE_CONTRACT=1"
            assert path.read_text(encoding="utf-8") == text, (
                f"{path} differs from the API's {name}; if intended, run with UPDATE_CONTRACT=1"
            )


def test_game_flow_responses(client):
    game_round = client.post("/v1/game/rounds", json={"count": 2}).json()
    rid = game_round["round_id"]
    check("round.json", {**game_round, "round_id": ROUND_ID})

    base = f"/v1/game/rounds/{rid}"
    check("fifty_fifty.json", client.post(f"{base}/fifty-fifty", json={"question_id": str(Q1)}).json())
    answer = client.post(f"{base}/answers", json={"question_id": str(Q1), "answer_index": 1})
    check("answer.json", answer.json())
    # A wrong answer to a tagged question says what it suggests (in its own round,
    # since a question is answered once).
    other = client.post("/v1/game/rounds", json={"count": 1}).json()["round_id"]
    check("answer_wrong.json", client.post(
        f"/v1/game/rounds/{other}/answers", json={"question_id": str(Q1), "answer_index": 2}
    ).json())
    conflict = client.post(f"{base}/answers", json={"question_id": str(Q1), "answer_index": 2})
    assert conflict.status_code == 409
    check("error_already_answered.json", conflict.json())
    check("answer_no_simplified.json", client.post(
        f"{base}/answers", json={"question_id": str(Q2), "answer_index": 0}
    ).json())
    check("batch.json", client.post(f"{base}/batches", json={"batch_index": 1, "count": 10}).json())
    check("batch_empty.json", client.post(f"{base}/batches", json={"batch_index": 2}).json())
    check("add_question.json", client.post(f"{base}/questions", json={"question_id": str(Q1)}).json())
    check("search.json", client.get("/v1/search/game-questions", params={"q": "key"}).json())
    # The game sends its search as a POST body; both forms answer the same.
    check("search.json", client.post("/v1/search/game-questions", json={"q": "key"}).json())
    check("story.json", client.get("/v1/game/story").json())


def test_icon_catalog():
    check("icon_catalog.json", {"keys": [icon.key for icon in ICONS]})


def test_validation_error_response(client):
    check("error_validation.json", client.post("/v1/game/rounds", json={"count": 0}).json())


# --- /v2/game: composed questions -------------------------------------------------------

# Q1 composed from fixed records: its decomposition (the stored parts and entities of v1
# above, a decomposition whose answer options name what each answer proposes, a flow
# and a resolution with a consequence per answer), its objectives and its story.
V2_SEMANTIC = decomposition_reply(3)
V2_SEMANTIC["entities"][0].update({"name": "API Key", "type": "artifact", "aliases": ["key"]})
V2_SEMANTIC["entities"].append(
    {"id": "e3", "name": "x-api-key", "type": "concept", "aliases": [], "description": "A header", "source_status": "inferred", "evidence": ""}
)
V2_SEMANTIC["answer_options"] = [
    {"index": 0, "mechanism": "Send the key as a bearer token", "entity_ids": ["e1"]},
    {"index": 1, "mechanism": "Send the key in its own header", "entity_ids": ["e1", "e3"]},
    {"index": 2, "mechanism": "Send the key in a custom header", "entity_ids": ["e1"]},
]
V2_SEMANTIC["events"] = [
    {"id": "v1", "name": "sends a request with the key", "actor_id": None, "target_id": "e1", "input_ids": [],
     "output_ids": [], "sequence": 1, "source_status": "inferred", "evidence": ""}
]
V2_ICONS = {
    "question": "api",
    "answers": ["credential", "parameter", "tags"],
    "entities": {"e1": "credential", "e2": "idea", "e3": "parameter"},
    "actions": {"v1": "send"},
}
V2_RESOLUTION = resolution_reply(3, (1,))
V2_RESOLUTION["options"][2]["impact"] = "partial"
V2_DECOMPOSITION = {
    **DECOMPOSITION[0],
    "semantic": V2_SEMANTIC,
    "icons": V2_ICONS,
    "icons_ica": {"model_id": "claude-haiku-5-5", "finish_reason": "stop", "total_tokens": 1, "attempts": 1},
    "flow": flow_reply(),
    "resolution": V2_RESOLUTION,
}


def v2_cards() -> list[dict]:
    composed, lacks = compose.compose(EXAM[0], TAGS[0], STORY[0], V2_DECOMPOSITION)
    assert lacks == []
    return [{**composed, "_id": ObjectId("65f0000000000000000000f1"), "question_id": str(Q1), "held": False}]


@pytest.fixture
def client_v2():
    rounds = FakeRounds()
    app.dependency_overrides[get_async_game_question_collection] = lambda: FakeCollection(v2_cards())
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(EXAM)
    app.dependency_overrides[get_async_rounds_collection] = lambda: rounds
    return TestClient(app)


def test_game_v2_flow_responses(client_v2):
    client = client_v2
    game_round = client.post("/v2/game/rounds", json={"count": 2}).json()
    rid = game_round["round_id"]
    check("v2_round.json", {**game_round, "round_id": ROUND_ID})
    base = f"/v2/game/rounds/{rid}"
    check("v2_answer.json", client.post(
        f"{base}/answers", json={"question_id": str(Q1), "answer_index": 1}
    ).json())
    other = client.post("/v2/game/rounds", json={"count": 1}).json()["round_id"]
    check("v2_answer_wrong.json", client.post(
        f"/v2/game/rounds/{other}/answers", json={"question_id": str(Q1), "answer_index": 2}
    ).json())
    check("v2_batch_empty.json", client.post(f"{base}/batches", json={"batch_index": 1}).json())
    check("v2_add_question.json", client.post(f"{base}/questions", json={"question_id": str(Q1)}).json())
    check("v2_search.json", client.post("/v2/search/game-questions", json={"q": "key"}).json())
    check("v2_story.json", client.get("/v2/game/story").json())


@pytest.mark.parametrize(
    ("prefixes", "name"),
    [
        (("/v1/game", "/v1/search/game"), "openapi_game.json"),
        (("/v2/game", "/v2/search/game"), "openapi_game_v2.json"),
    ],
)
def test_openapi_game_schema(prefixes, name):
    spec = app.openapi()
    paths = {p: v for p, v in spec["paths"].items() if p.startswith(prefixes)}
    used = set()

    def collect(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if ref and ref.startswith("#/components/schemas/"):
                name = ref.rsplit("/", 1)[1]
                if name not in used:
                    used.add(name)
                    collect(spec["components"]["schemas"][name])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(paths)
    schemas = {n: spec["components"]["schemas"][n] for n in sorted(used)}
    snapshot = {"paths": paths, "schemas": schemas}
    path = CONTRACT_DIR / name
    text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    else:
        assert path.is_file() and path.read_text(encoding="utf-8") == text, (
            "The game API's OpenAPI schema changed; if intended, run with UPDATE_CONTRACT=1"
        )
