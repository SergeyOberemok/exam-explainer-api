"""The `/v2/game` rounds (`CardService` over composed `game_question` documents) against a
real MongoDB, to prove the conditional writes (and the operator semantics `tests/fakes.py`
imitates).

Opt-in: set TEST_MONGO_URL (e.g. mongodb://127.0.0.1:27017). Each run uses a fresh,
randomly named database and drops it afterwards; nothing else on the server is touched.

    TEST_MONGO_URL=mongodb://127.0.0.1:27017 uv run --with pytest pytest tests/test_game_mongo.py
"""

import os
import secrets

import pytest
from bson import ObjectId
from pymongo import AsyncMongoClient

from claude_exam.api.game import card_service, errors, repository
from claude_exam.api.game.card_repository import CardRepository
from claude_exam.api.game.card_service import CardService
from claude_exam.api.game.repository import RoundRepository

MONGO_URL = os.environ.get("TEST_MONGO_URL")
pytestmark = [
    pytest.mark.skipif(not MONGO_URL, reason="set TEST_MONGO_URL to run against MongoDB"),
    pytest.mark.anyio,
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db():
    client = AsyncMongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
    name = f"claude_exam_test_{secrets.token_hex(6)}"
    try:
        yield client[name]
    finally:
        await client.drop_database(name)
        await client.close()


def card(question_id: str, **kw):
    """A composed `game_question` document, as `compose.compose_and_store` writes one."""
    return {
        "question_id": question_id,
        "held": False,
        "section": "S",
        "assessment_type": "t",
        "scenario": "",
        "stage": "",
        "order": 0,
        "objectives": ["o1"],
        "difficulty": 1,
        "count": 4,
        "public": {"id": question_id, "section": "S", "assessment_type": "t", "question": "q", "ask": {"text": "q?"}},
        "key": {
            "correct": 1,
            "review": {"explanation": "e", "feedbacks": ["a", "b", "c", "d"]},
            "outcomes": [
                {"index": i, "verdict": "correct" if i == 1 else "incorrect",
                 "impact": "success" if i == 1 else "failure", "consequence": f"c{i}"}
                for i in range(4)
            ],
        },
        **kw,
    }


async def make_service(db, count):
    """A service over `count` composed questions (and their `examination` docs); their ids."""
    ids = [str(ObjectId()) for _ in range(count)]
    if ids:
        await db["game_question"].insert_many([card(i) for i in ids])
        await db["examination"].insert_many([{"_id": ObjectId(i)} for i in ids])
    svc = CardService(CardRepository(db["game_question"], db["examination"]), RoundRepository(db["game_rounds"]))
    return svc, ids


async def test_answer_once_eliminated_blocked_and_recorded(db):
    svc, ids = await make_service(db, 1)
    game_round = await svc.create_round(10, "", "")
    rid, qid = ObjectId(game_round.round_id), ids[0]

    fifty = await svc.fifty_fifty(rid, qid)
    assert len(fifty.eliminated_indices) == 2 and 1 not in fifty.eliminated_indices
    # `$ne` on the eliminated array refuses either removed answer.
    with pytest.raises(errors.AnswerEliminated):
        await svc.answer(rid, qid, fifty.eliminated_indices[0])

    result = await svc.answer(rid, qid, 1)
    assert result.is_correct
    with pytest.raises(errors.AlreadyAnswered):
        await svc.answer(rid, qid, 0)
    assert await svc.recorded_answer(rid, qid) == result


async def test_batches_exclude_everything_in_the_round(db):
    svc, ids = await make_service(db, 6)
    game_round = await svc.create_round(2, "", "")
    rid = ObjectId(game_round.round_id)
    started = [c.id for c in game_round.items]
    searched = next(i for i in ids if i not in started)
    await svc.add_question(rid, searched)

    batch = await svc.next_batch(rid, 1, 10)
    got = [c.id for c in batch.items]
    assert sorted(started + [searched] + got) == sorted(ids) and batch.remaining == 0
    assert await svc.next_batch(rid, 1, 10) == batch  # A retry returns the stored batch.
    assert (await svc.next_batch(rid, 2, 10)).items == []


async def test_concurrent_batch_and_add_cannot_duplicate(db):
    svc, ids = await make_service(db, 3)
    started = await svc.create_round(1, "", "")
    rid = ObjectId(started.round_id)
    rounds = RoundRepository(db.game_rounds)
    # A question the round does not have yet (the start samples one at random).
    other = next(i for i in ids if i != started.items[0].id)
    # A search add lands between sampling a batch and writing it: the batch write must miss.
    assert await rounds.add_question(rid, other, {"correct": 1, "count": 4, "key": {}})
    assert not await rounds.add_batch(rid, 1, {other: {}})
    stored = await db.game_rounds.find_one({"_id": rid})
    assert stored["question_ids"].count(other) == 1


async def test_round_cap_uses_the_question_ids_index(db, monkeypatch):
    monkeypatch.setattr(repository, "MAX_ROUND_QUESTIONS", 2)
    monkeypatch.setattr(card_service, "MAX_ROUND_QUESTIONS", 2)
    svc, ids = await make_service(db, 3)
    rid = ObjectId((await svc.create_round(2, "", "")).round_id)
    in_round = (await db.game_rounds.find_one({"_id": rid}))["question_ids"]
    left = next(i for i in ids if i not in in_round)
    with pytest.raises(errors.RoundFull):
        await svc.add_question(rid, left)


async def test_v2_rounds_serve_composed_documents_and_answer_once(db):
    ids = [str(ObjectId()) for _ in range(3)]
    await db["game_question"].insert_many([card(ids[0]), card(ids[1]), card(ids[2], held=True)])
    svc = CardService(CardRepository(db["game_question"], db["examination"]), RoundRepository(db["game_rounds"]))

    started = await svc.create_round(5, "", "")
    assert sorted(c.id for c in started.items) == sorted(ids[:2]) and started.total == 2
    rid = ObjectId(started.round_id)
    # The key is kept in the round, not sent; answering returns that answer's outcome.
    stored = await db["game_rounds"].find_one({"_id": rid})
    assert stored["questions"][ids[0]]["key"]["outcomes"][2]["consequence"] == "c2"
    result = await svc.answer(rid, ids[0], 2)
    assert (result.is_correct, result.outcome.consequence) == (False, "c2")
    with pytest.raises(errors.AlreadyAnswered):
        await svc.answer(rid, ids[0], 1)
    assert (await svc.next_batch(rid, 1, 5)).items == []
    # A held question exists but is not served; an unknown one is not found.
    await db["examination"].insert_one({"_id": ObjectId(ids[2])})
    with pytest.raises(errors.QuestionNotPlayable):
        await svc.add_question(rid, ids[2])
    with pytest.raises(errors.QuestionNotFound):
        await svc.add_question(rid, str(ObjectId()))
    listed = await svc.list_questions(10, None, None, shuffle=False)
    assert [c.id for c in listed.items] == sorted(ids[:2]) and listed.total == 2
    assert await svc.sections() == ["S"]
