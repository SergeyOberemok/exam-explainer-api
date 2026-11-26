"""`/v2/game` and `/v2/search/game-questions` over composed questions (`compose.py`):
rounds, batches, answers with their outcome, the 50/50, adding a searched question, and
the browse, objective and story counts. The documents are composed by a real "full" job
over the job fakes, then served from the in-memory collections of `fakes.py`."""

import asyncio
import json

import pytest
from fakes import FakeCollection, FakeRounds
from fastapi.testclient import TestClient
from test_jobs import (
    MODEL,
    TAGGABLE,
    FakeMongo,
    ScriptedIca,
    in_chapter,
    objectives_reply,
    reply,
)

from claude_exam.api import batch
from claude_exam.api.batch import new_job, run_simplification_job
from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_game_question_collection,
    get_async_rounds_collection,
)
from claude_exam.api.main import app

FOUR = ["acceptEdits mode", "plan mode", "auto mode", "bypass mode"]


def _bank():
    """Three composed questions (the first with four answers) and one held one."""
    four = in_chapter(section=TAGGABLE)
    four["prompt"]["answers"] = list(FOUR)
    four["prompt"]["feedbacks"] = ["no", "yes", "no", "no"]
    questions = [four, *(in_chapter(section=TAGGABLE) for _ in range(3))]
    tagged = objectives_reply()
    tagged["distractors"] = [
        {"misconception": "Auto-accepting edits keeps files unchanged.", "confused_with": None},
        None,
        {"misconception": "Auto mode asks before each edit.", "confused_with": None},
        {"misconception": "Bypassing permissions is safe here.", "confused_with": None},
    ]
    mongo = FakeMongo(questions)
    ica = ScriptedIca(objectives=[reply(tagged)])
    jobs = mongo.get_default_database()["simplification_jobs"]

    async def run():
        job = await jobs.insert_one(new_job("t", MODEL, len(questions), "full"))
        await run_simplification_job(
            mongo, ica, asyncio.Semaphore(1), job.inserted_id, [q["_id"] for q in questions], MODEL, "full"
        )

    asyncio.run(run())
    db = mongo.get_default_database()
    # The last one is held afterwards by hand (`POST /evaluate/questions/{id}/hold`
    # mirrors the hold on its game document): composed, but not served.
    db["examination"].docs[3]["pipeline_mode"] = "manual"
    db["game_question"].docs[3]["held"] = True
    return questions, db


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)


@pytest.fixture
def bank():
    questions, db = _bank()
    assert [d["question_id"] for d in db["game_question"].docs] == [str(q["_id"]) for q in questions]
    cards = FakeCollection(db["game_question"].docs)
    rounds = FakeRounds()
    app.dependency_overrides[get_async_game_question_collection] = lambda: cards
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(db["examination"].docs)
    app.dependency_overrides[get_async_rounds_collection] = lambda: rounds
    return questions, TestClient(app)


def test_a_round_serves_cards_without_their_key(bank):
    questions, client = bank
    resp = client.post("/v2/game/rounds", json={"count": 10})
    assert resp.status_code == 201
    body = resp.json()
    assert body["total"] == 3 and body["fifty_fifty_charges"] == 2
    ids = [i["id"] for i in body["items"]]
    assert sorted(ids) == sorted(str(q["_id"]) for q in questions[:3])  # not the held one
    text = json.dumps(body)
    assert "Plan mode blocks edits." not in text  # the explanation
    assert not {"correct", "outcomes", "key", "simplified"} & (set(body) | set(body["items"][0]))
    card = body["items"][0]
    assert card["ask"]["text"] == "Which mode?"
    assert [t["index"] for t in card["tools"]] == list(range(len(card["tools"])))
    assert card["map"]["nodes"] and card["cast"]


def test_answering_returns_the_outcome_of_that_answer_once(bank):
    questions, client = bank
    round_id = client.post("/v2/game/rounds", json={"count": 3}).json()["round_id"]
    qid = str(questions[0]["_id"])
    url = f"/v2/game/rounds/{round_id}/answers"

    wrong = client.post(url, json={"question_id": qid, "answer_index": 2})
    assert wrong.status_code == 200
    result = wrong.json()
    assert (result["is_correct"], result["correct_index"]) == (False, 1)
    assert result["outcome"]["index"] == 2 and result["outcome"]["impact"] == "failure"
    assert result["outcome"]["consequence"] == "The edits land unreviewed."
    assert result["outcome"]["misconception"] == "Auto mode asks before each edit."
    assert result["story_line"] == "Edits land unreviewed."
    assert result["review"]["explanation"] == "Plan mode blocks edits."
    assert result["rule"] and result["resolution_flow"]["nodes"]

    assert client.post(url, json={"question_id": qid, "answer_index": 2}).json() == result
    other = client.post(url, json={"question_id": qid, "answer_index": 1})
    assert (other.status_code, other.json()["code"]) == (409, "already_answered")
    assert client.get(f"{url}/{qid}").json() == result


def test_batches_go_on_without_repeats_until_the_pool_is_used_up(bank):
    _, client = bank
    start = client.post("/v2/game/rounds", json={"count": 2}).json()
    url = f"/v2/game/rounds/{start['round_id']}/batches"
    first = client.post(url, json={"batch_index": 1, "count": 5}).json()
    assert first["remaining"] == 0 and len(first["items"]) == 1
    assert first["items"][0]["id"] not in {i["id"] for i in start["items"]}
    assert client.post(url, json={"batch_index": 1}).json() == first  # a retry
    assert client.post(url, json={"batch_index": 2}).json()["items"] == []
    late = client.post(url, json={"batch_index": 4})
    assert late.json()["code"] == "batch_out_of_order"


def test_fifty_fifty_removes_two_wrong_tools(bank):
    questions, client = bank
    round_id = client.post("/v2/game/rounds", json={"count": 3}).json()["round_id"]
    url = f"/v2/game/rounds/{round_id}/fifty-fifty"
    four = client.post(url, json={"question_id": str(questions[0]["_id"])}).json()
    assert len(four["eliminated_indices"]) == 2 and 1 not in four["eliminated_indices"]
    assert four["fifty_fifty_remaining"] == 1
    two = client.post(url, json={"question_id": str(questions[1]["_id"])})
    assert (two.status_code, two.json()["code"]) == (409, "too_few_answers")


def test_adding_a_question_needs_one_the_game_serves(bank):
    questions, client = bank
    round_id = client.post("/v2/game/rounds", json={"count": 1}).json()["round_id"]
    url = f"/v2/game/rounds/{round_id}/questions"
    added = [client.post(url, json={"question_id": str(q["_id"])}) for q in questions[:3]]
    assert all(r.status_code == 200 for r in added)
    assert added[1].json()["item"]["id"] == str(questions[1]["_id"])
    held = client.post(url, json={"question_id": str(questions[3]["_id"])})
    assert (held.status_code, held.json()["code"]) == (422, "question_not_playable")
    unknown = client.post(url, json={"question_id": "0" * 24})
    assert (unknown.status_code, unknown.json()["code"]) == (404, "question_not_found")


def test_browse_counts_and_search_serve_only_what_the_game_serves(bank):
    questions, client = bank
    listed = client.get("/v2/game/questions", params={"count": 50}).json()
    assert listed["total"] == 3 and len(listed["items"]) == 3
    one = client.get(f"/v2/game/questions/{questions[0]['_id']}").json()
    assert one["id"] == str(questions[0]["_id"])
    assert client.get(f"/v2/game/questions/{questions[3]['_id']}").json()["code"] == "question_not_found"
    assert client.get("/v2/game/sections").json() == {"items": [TAGGABLE]}
    objectives = {o["key"]: o["count"] for o in client.get("/v2/game/objectives").json()["items"]}
    assert objectives["claude-code-planning"] == 3
    chapters = {c["key"]: c for c in client.get("/v2/game/story").json()["chapters"]}
    assert chapters["pipeline"]["count"] == 3
    found = client.post("/v2/search/game-questions", json={"q": "which mode"}).json()
    assert {i["id"] for i in found["items"]} == {str(q["_id"]) for q in questions[:3]}
    assert "outcomes" not in json.dumps(found)
