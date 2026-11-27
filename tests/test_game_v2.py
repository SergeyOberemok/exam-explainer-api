"""`/v2/game` and `/v2/search/game-questions` over composed questions (`compose.py`):
rounds, batches, answers with their outcome, the 50/50, adding a searched question, and
the browse, objective and story counts. The documents are composed by a real "full" job
over the job fakes, then served from the in-memory collections of `fakes.py`."""

import asyncio
import copy
import json

import pytest
from bson import ObjectId
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
    assert [o["index"] for o in result["outcomes"]] == list(range(len(result["outcomes"])))
    assert result["outcomes"][2] == result["outcome"]
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


# --- the round rules (`game.service.RoundRules`): errors, retries and limits ------------

NO_ID = "0" * 24


def clones(db, count):
    """`count` served copies of the composed four-answer question, each its own question."""
    docs = []
    for _ in range(count):
        doc = copy.deepcopy(db["game_question"].docs[0])
        question_id = str(ObjectId())
        doc.update(_id=ObjectId(), question_id=question_id)
        doc["public"]["id"] = question_id
        docs.append(doc)
    return docs


@pytest.fixture
def many():
    """A client over four four-answer questions (enough for every 50/50 charge): their
    ids, the client and the round store."""
    _, db = _bank()
    cards = clones(db, 4)
    exam = [{"_id": ObjectId(c["question_id"])} for c in cards]
    rounds = FakeRounds()
    app.dependency_overrides[get_async_game_question_collection] = lambda: FakeCollection(cards)
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(exam)
    app.dependency_overrides[get_async_rounds_collection] = lambda: rounds
    return [c["question_id"] for c in cards], TestClient(app), rounds


def test_answer_errors_carry_a_code(many):
    ids, client, _ = many
    round_id = client.post("/v2/game/rounds", json={"count": 4}).json()["round_id"]
    url = f"/v2/game/rounds/{round_id}/answers"
    out_of_range = client.post(url, json={"question_id": ids[0], "answer_index": 4})
    assert (out_of_range.status_code, out_of_range.json()["code"]) == (422, "answer_out_of_range")
    assert client.post(url, json={"question_id": ids[0], "answer_index": -1}).status_code == 422
    other = client.post(url, json={"question_id": NO_ID, "answer_index": 0})
    assert other.json() == {"detail": "question is not in this round", "code": "question_not_in_round"}
    # Not an ObjectId hex string, so it can't reach a Mongo field path.
    assert client.post(url, json={"question_id": "$where.x", "answer_index": 0}).status_code == 422
    unknown = client.post(f"/v2/game/rounds/{NO_ID}/answers", json={"question_id": ids[0], "answer_index": 0})
    assert (unknown.status_code, unknown.json()["code"]) == (404, "round_not_found")
    bad_round = client.post("/v2/game/rounds/nope/answers", json={"question_id": ids[0], "answer_index": 0})
    assert bad_round.status_code == 404


def test_recorded_answer_errors(many):
    ids, client, _ = many
    round_id = client.post("/v2/game/rounds", json={"count": 4}).json()["round_id"]
    base = f"/v2/game/rounds/{round_id}/answers"
    unanswered = client.get(f"{base}/{ids[0]}")
    assert (unanswered.status_code, unanswered.json()["code"]) == (404, "question_not_answered")
    assert client.get(f"{base}/{NO_ID}").json()["code"] == "question_not_in_round"
    assert client.get(f"/v2/game/rounds/{NO_ID}/answers/{ids[0]}").json()["code"] == "round_not_found"


def test_fifty_fifty_repeats_blocks_its_answers_and_runs_out(many):
    ids, client, _ = many
    round_id = client.post("/v2/game/rounds", json={"count": 4}).json()["round_id"]
    base = f"/v2/game/rounds/{round_id}"
    first = client.post(f"{base}/fifty-fifty", json={"question_id": ids[0]}).json()
    # Asking again repeats the result without using a charge.
    assert client.post(f"{base}/fifty-fifty", json={"question_id": ids[0]}).json() == first
    removed = first["eliminated_indices"][0]
    blocked = client.post(f"{base}/answers", json={"question_id": ids[0], "answer_index": removed})
    assert (blocked.status_code, blocked.json()["code"]) == (422, "answer_eliminated")
    assert client.post(f"{base}/answers", json={"question_id": ids[0], "answer_index": 1}).status_code == 200

    client.post(f"{base}/answers", json={"question_id": ids[1], "answer_index": 1})
    answered = client.post(f"{base}/fifty-fifty", json={"question_id": ids[1]})
    assert (answered.status_code, answered.json()["code"]) == (409, "already_answered")
    last = client.post(f"{base}/fifty-fifty", json={"question_id": ids[2]}).json()
    assert last["fifty_fifty_remaining"] == 0
    empty = client.post(f"{base}/fifty-fifty", json={"question_id": ids[3]})
    assert (empty.status_code, empty.json()["code"]) == (409, "no_fifty_fifty_charges")


def test_batch_errors(many):
    _, client, _ = many
    round_id = client.post("/v2/game/rounds", json={"count": 1}).json()["round_id"]
    assert client.post(f"/v2/game/rounds/{round_id}/batches", json={"batch_index": 0}).status_code == 422
    missing = client.post(f"/v2/game/rounds/{NO_ID}/batches", json={"batch_index": 1})
    assert (missing.status_code, missing.json()["code"]) == (404, "round_not_found")


def test_round_is_capped(many, monkeypatch):
    from claude_exam.api.game import card_service, repository

    monkeypatch.setattr(repository, "MAX_ROUND_QUESTIONS", 2)
    monkeypatch.setattr(card_service, "MAX_ROUND_QUESTIONS", 2)
    ids, client, rounds = many
    body = client.post("/v2/game/rounds", json={"count": 1}).json()
    base = f"/v2/game/rounds/{body['round_id']}"
    more = client.post(f"{base}/batches", json={"batch_index": 1, "count": 10}).json()
    assert len(more["items"]) == 1  # Trimmed to the room left.
    in_round = next(iter(rounds.docs.values()))["question_ids"]
    left = next(i for i in ids if i not in in_round)
    full = client.post(f"{base}/questions", json={"question_id": left})
    assert (full.status_code, full.json()["code"]) == (409, "round_full")
    assert client.post(f"{base}/batches", json={"batch_index": 2}).json()["code"] == "round_full"
    # A question already in the round is still returned.
    assert client.post(f"{base}/questions", json={"question_id": in_round[0]}).status_code == 200


def test_a_round_nothing_matches_is_404_and_not_stored(many):
    _, client, rounds = many
    r = client.post("/v2/game/rounds", json={"count": 10, "objective": "no-such-objective"})
    assert (r.status_code, r.json()["code"]) == (404, "no_questions")
    assert rounds.docs == {}


def test_validation_error_detail_is_a_string(many):
    _, client, _ = many
    r = client.post("/v2/game/rounds", json={"count": 0})
    body = r.json()
    assert r.status_code == 422 and body["code"] == "validation_error" and isinstance(body["detail"], str)
    assert "count" in body["detail"] and body["errors"][0]["loc"] == ["body", "count"]


def test_round_creation_is_rate_limited(many, monkeypatch):
    from claude_exam.api.security import round_rate_limit

    monkeypatch.setattr(round_rate_limit, "times", 2)
    _, client, _ = many
    codes = [client.post("/v2/game/rounds", json={"count": 1}).status_code for _ in range(3)]
    assert codes == [201, 201, 429]


def test_the_game_needs_no_api_key_and_v1_game_is_gone(many):
    _, client, _ = many
    app.state.api_key = "secret"
    assert client.get("/questions/").status_code == 401
    assert client.post("/v2/game/rounds", json={"count": 1}).status_code == 201
    app.state.api_key = None
    for path in ("/v1/game/sections", "/game/sections", "/v1/search/game-questions?q=ok"):
        assert client.get(path).status_code == 404
