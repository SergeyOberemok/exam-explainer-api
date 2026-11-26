"""The game repository and service against a real MongoDB, to prove the conditional writes
(and the operator semantics `tests/fakes.py` imitates).

Opt-in: set TEST_MONGO_URL (e.g. mongodb://127.0.0.1:27017). Each run uses a fresh,
randomly named database and drops it afterwards; nothing else on the server is touched.

    TEST_MONGO_URL=mongodb://127.0.0.1:27017 uv run --with pytest pytest tests/test_game_mongo.py
"""

import os
import secrets

import pytest
from bson import ObjectId
from pymongo import AsyncMongoClient

from claude_exam.api.game import errors, repository, service
from claude_exam.api.game.repository import QuestionRepository, RoundRepository
from claude_exam.api.game.service import GameService

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


def question(**kw):
    return {
        "_id": ObjectId(),
        "section": "S",
        "assessment_type": "t",
        "correct_response": ["b"],
        "prompt": {
            "question": "q",
            "answers": ["w", "x", "y", "z"],
            "feedbacks": ["fw", "fx", "fy", "fz"],
            "explanation": "e",
        },
        **kw,
    }


async def make_service(db, docs):
    if docs:
        await db.examination.insert_many(docs)
        # The game serves only questions with a rephrase record; these are never offered
        # (the check found them `changed`), so the questions play as their original.
        await db.examination_simplified.insert_many(
            [
                {"question_id": str(d["_id"]), "prompt": d["prompt"], "check": {"verdict": "changed"}}
                for d in docs
            ]
        )
    return GameService(
        QuestionRepository(db.examination, db.examination_simplified),
        RoundRepository(db.game_rounds),
    )


async def test_answer_once_eliminated_blocked_and_recorded(db):
    docs = [question(), question(correct_response=["a", "b"])]
    svc = await make_service(db, docs)
    game_round = await svc.create_round(10, "", "")
    # The playable pre-filter and `is_playable` keep the multi-answer question out.
    assert [q.id for q in game_round.items] == [str(docs[0]["_id"])]
    rid, qid = ObjectId(game_round.round_id), str(docs[0]["_id"])

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
    docs = [question() for _ in range(6)]
    ids = [str(d["_id"]) for d in docs]
    svc = await make_service(db, docs)
    game_round = await svc.create_round(2, "", "")
    rid = ObjectId(game_round.round_id)
    started = [q.id for q in game_round.items]
    searched = next(i for i in ids if i not in started)
    await svc.add_question(rid, searched)

    batch = await svc.next_batch(rid, 1, 10)
    got = [q.id for q in batch.items]
    assert sorted(started + [searched] + got) == sorted(ids) and batch.remaining == 0
    assert await svc.next_batch(rid, 1, 10) == batch  # A retry returns the stored batch.
    assert (await svc.next_batch(rid, 2, 10)).items == []


async def test_concurrent_batch_and_add_cannot_duplicate(db):
    docs = [question() for _ in range(3)]
    svc = await make_service(db, docs)
    started = await svc.create_round(1, "", "")
    rid = ObjectId(started.round_id)
    rounds = RoundRepository(db.game_rounds)
    # A question the round does not have yet (the start samples one at random).
    other = next(str(d["_id"]) for d in docs if str(d["_id"]) != started.items[0].id)
    # A search add lands between sampling a batch and writing it: the batch write must miss.
    assert await rounds.add_question(rid, other, {"correct": 1, "count": 4, "original": {}, "simplified": None})
    assert not await rounds.add_batch(rid, 1, {other: {}})
    stored = await db.game_rounds.find_one({"_id": rid})
    assert stored["question_ids"].count(other) == 1


async def test_round_cap_uses_the_question_ids_index(db, monkeypatch):
    monkeypatch.setattr(repository, "MAX_ROUND_QUESTIONS", 2)
    monkeypatch.setattr(service, "MAX_ROUND_QUESTIONS", 2)
    docs = [question() for _ in range(3)]
    svc = await make_service(db, docs)
    rid = ObjectId((await svc.create_round(2, "", "")).round_id)
    in_round = (await db.game_rounds.find_one({"_id": rid}))["question_ids"]
    left = next(str(d["_id"]) for d in docs if str(d["_id"]) not in in_round)
    with pytest.raises(errors.RoundFull):
        await svc.add_question(rid, left)


# --- /v2/game over composed documents (`compose.py`) -----------------------------------


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


async def test_v2_rounds_serve_composed_documents_and_answer_once(db):
    from claude_exam.api.game.card_repository import CardRepository
    from claude_exam.api.game.card_service import CardService

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
