from bson import ObjectId
from fakes import FakeCollection, FakeRounds
from fastapi.testclient import TestClient

from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_rounds_collection,
    get_async_simplified_collection,
)
from claude_exam.api.game import repository, service
from claude_exam.api.game.domain import (
    FIFTY_FIFTY_CHARGES,
    is_playable,
    letter_to_index,
    to_game_question,
    to_review,
)
from claude_exam.api.main import app
from claude_exam.api.security import round_rate_limit


def doc(**kw):
    base = {
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
    }
    return {**base, **kw}


def simplified_of(original, **kw):
    prompt = {**original["prompt"], "question": "simple q", "explanation": "simple e"}
    return {"_id": ObjectId(), **original, "question_id": str(original["_id"]), "prompt": prompt, **kw}


def test_letter_to_index_tolerates_bad_values():
    assert [letter_to_index(v) for v in ("a", " C ", "ab", "", None, 3)] == [0, 2, -1, -1, -1, -1]


def test_public_question_has_no_answer_key():
    q = to_game_question(doc()).model_dump()
    assert set(q) == {"id", "simplified", "section", "assessment_type", "question", "answers"}
    assert q["answers"][1] == {"index": 1, "text": "x"}


def test_review_pads_missing_feedback():
    review = to_review(doc(prompt={"question": "q", "answers": ["x", "y"], "feedbacks": ["fx"]}))
    assert review.feedbacks == ["fx", ""] and review.explanation == ""


def test_is_playable():
    assert is_playable(doc())
    assert not is_playable(doc(correct_response=["a", "b"]))
    assert not is_playable(doc(correct_response=["z"]))
    assert not is_playable(doc(prompt={"answers": ["only"]}))
    assert not is_playable(doc(prompt={"answers": list("abcde")}))


def test_simplified_uses_question_id():
    q = to_game_question(doc(question_id="abc"), simplified=True)
    assert q.id == "abc" and q.simplified


def client_with(exam=(), simple=(), rounds=None):
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(list(exam))
    app.dependency_overrides[get_async_simplified_collection] = lambda: FakeCollection(list(simple))
    rounds = rounds or FakeRounds()
    app.dependency_overrides[get_async_rounds_collection] = lambda: rounds
    return TestClient(app)  # no lifespan: not used as a context manager


def test_batch_dedupes_sample_duplicates():
    d = doc()
    r = client_with(exam=[d, d]).get("/v1/game/questions?count=2")
    assert r.status_code == 200 and len(r.json()["items"]) == 1


def test_batch_has_no_answer_key():
    item = client_with(exam=[doc()]).get("/v1/game/questions").json()["items"][0]
    assert "correct_indices" not in item and "explanation" not in item
    assert all(set(a) == {"index", "text"} for a in item["answers"])


def test_count_zero_is_422():
    assert client_with().get("/v1/game/questions?count=0").status_code == 422


def test_simplified_missing_is_404():
    r = client_with().get("/v1/game/simplified-questions/nope")
    assert r.status_code == 404 and r.json()["detail"] == "simplified question not found"


def test_bad_object_id_is_404():
    assert client_with().get("/v1/game/questions/not-an-id").status_code == 404


def test_sections_simplified_flag():
    c = client_with(exam=[doc(section="A")], simple=[doc(section="B")])
    assert c.get("/v1/game/sections").json() == {"items": ["A"]}
    assert c.get("/v1/game/sections?simplified=true").json() == {"items": ["B"]}


def start_round(exam, simple=(), rounds=None):
    c = client_with(exam=exam, simple=simple, rounds=rounds)
    r = c.post("/v1/game/rounds", json={"count": 10})
    assert r.status_code == 201, r.text
    return c, r.json()


def test_round_returns_questions_and_simplified_without_answer_key():
    a, b = doc(), doc()
    _, body = start_round([a, b], simple=[simplified_of(a)])

    assert [q["id"] for q in body["items"]] == [str(a["_id"]), str(b["_id"])]
    assert [q["id"] for q in body["simplified"]] == [str(a["_id"])]
    assert body["simplified"][0]["simplified"] and body["simplified"][0]["question"] == "simple q"
    assert body["fifty_fifty_charges"] == FIFTY_FIFTY_CHARGES
    text = str(body)
    assert "correct" not in text and "feedback" not in text and "explanation" not in text


def test_round_skips_unplayable_questions_and_mismatched_simplified():
    good, multi = doc(), doc(correct_response=["a", "b"])
    _, body = start_round([good, multi], simple=[simplified_of(good, correct_response=["c"])])
    assert [q["id"] for q in body["items"]] == [str(good["_id"])]
    assert body["simplified"] == []


def test_round_accepts_empty_filters_from_jsonutility():
    c = client_with(exam=[doc()])
    r = c.post("/v1/game/rounds", json={"count": 1, "section": "", "assessment_type": ""})
    assert r.status_code == 201


def test_answer_returns_result_and_reviews_once():
    a = doc()
    c, body = start_round([a], simple=[simplified_of(a)])
    url = f"/v1/game/rounds/{body['round_id']}/answers"
    qid = str(a["_id"])

    r = c.post(url, json={"question_id": qid, "answer_index": 0})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["is_correct"] is False and result["correct_index"] == 1
    assert result["original"] == {"explanation": "e", "feedbacks": ["fw", "fx", "fy", "fz"]}
    assert result["has_simplified"] and result["simplified"]["explanation"] == "simple e"

    retry = c.post(url, json={"question_id": qid, "answer_index": 0})
    assert retry.status_code == 200 and retry.json() == result
    other = c.post(url, json={"question_id": qid, "answer_index": 1})
    assert other.status_code == 409 and other.json()["detail"] == "question already answered"


def test_answer_without_simplified_has_empty_review():
    a = doc()
    c, body = start_round([a])
    r = c.post(f"/v1/game/rounds/{body['round_id']}/answers", json={"question_id": str(a["_id"]), "answer_index": 1})
    assert r.json()["is_correct"] and not r.json()["has_simplified"]
    assert r.json()["simplified"] == {"explanation": "", "feedbacks": []}


def test_answer_errors():
    a = doc()
    c, body = start_round([a])
    url = f"/v1/game/rounds/{body['round_id']}/answers"
    qid = str(a["_id"])

    assert c.post(url, json={"question_id": qid, "answer_index": 4}).status_code == 422
    assert c.post(url, json={"question_id": qid, "answer_index": -1}).status_code == 422
    other = c.post(url, json={"question_id": str(ObjectId()), "answer_index": 0})
    assert other.status_code == 404 and other.json()["detail"] == "question is not in this round"
    # Not an ObjectId hex string, so it can't reach a Mongo field path.
    assert c.post(url, json={"question_id": "$where.x", "answer_index": 0}).status_code == 422
    unknown = c.post(f"/v1/game/rounds/{ObjectId()}/answers", json={"question_id": qid, "answer_index": 0})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "round not found"
    assert c.post("/v1/game/rounds/nope/answers", json={"question_id": qid, "answer_index": 0}).status_code == 404


def test_fifty_fifty_removes_two_wrong_answers_and_blocks_them():
    a, b = doc(), doc()
    c, body = start_round([a, b])
    base = f"/v1/game/rounds/{body['round_id']}"
    qid = str(a["_id"])

    r = c.post(f"{base}/fifty-fifty", json={"question_id": qid})
    assert r.status_code == 200, r.text
    eliminated = r.json()["eliminated_indices"]
    assert len(eliminated) == 2 and 1 not in eliminated
    assert r.json()["fifty_fifty_remaining"] == FIFTY_FIFTY_CHARGES - 1

    retry = c.post(f"{base}/fifty-fifty", json={"question_id": qid})
    assert retry.status_code == 200 and retry.json() == r.json()
    blocked = c.post(f"{base}/answers", json={"question_id": qid, "answer_index": eliminated[0]})
    assert blocked.status_code == 422 and blocked.json()["detail"] == "answer was eliminated by 50/50"
    assert c.post(f"{base}/answers", json={"question_id": qid, "answer_index": 1}).status_code == 200


def test_fifty_fifty_after_answer_and_out_of_charges():
    questions = [doc() for _ in range(FIFTY_FIFTY_CHARGES + 2)]
    c, body = start_round(questions)
    base = f"/v1/game/rounds/{body['round_id']}"
    ids = [str(q["_id"]) for q in questions]

    c.post(f"{base}/answers", json={"question_id": ids[0], "answer_index": 1})
    answered = c.post(f"{base}/fifty-fifty", json={"question_id": ids[0]})
    assert answered.status_code == 409 and answered.json()["detail"] == "question already answered"

    for qid in ids[1 : 1 + FIFTY_FIFTY_CHARGES]:
        assert c.post(f"{base}/fifty-fifty", json={"question_id": qid}).status_code == 200
    empty = c.post(f"{base}/fifty-fifty", json={"question_id": ids[-1]})
    assert empty.status_code == 409 and empty.json()["detail"] == "no 50/50 charges left"


def test_add_question_to_round_makes_it_answerable():
    a, extra = doc(), doc()
    rounds = FakeRounds()
    _, body = start_round([a], rounds=rounds)
    # The round has `a`; `extra` comes from search and is added afterwards.
    c = client_with(exam=[extra], simple=[simplified_of(extra)], rounds=rounds)
    base = f"/v1/game/rounds/{body['round_id']}"
    qid = str(extra["_id"])

    r = c.post(f"{base}/questions", json={"question_id": qid})
    assert r.status_code == 200, r.text
    added = r.json()
    assert added["item"]["id"] == qid and not added["item"]["simplified"]
    assert [s["id"] for s in added["simplified"]] == [qid] and added["simplified"][0]["simplified"]
    text = str(added)
    assert "correct" not in text and "feedback" not in text and "explanation" not in text

    answer = c.post(f"{base}/answers", json={"question_id": qid, "answer_index": 1})
    assert answer.status_code == 200 and answer.json()["is_correct"]
    assert answer.json()["has_simplified"]


def test_add_question_again_keeps_the_stored_entry():
    a = doc()
    rounds = FakeRounds()
    _, body = start_round([a], rounds=rounds)
    qid = str(a["_id"])
    # Simplified after the round started: the stored entry has no review for it.
    c = client_with(exam=[a], simple=[simplified_of(a)], rounds=rounds)
    url = f"/v1/game/rounds/{body['round_id']}/questions"

    r = c.post(url, json={"question_id": qid})
    assert r.status_code == 200 and r.json()["simplified"] == []
    c.post(f"/v1/game/rounds/{body['round_id']}/answers", json={"question_id": qid, "answer_index": 0})
    assert c.post(url, json={"question_id": qid}).json() == r.json()
    stored = next(iter(rounds.docs.values()))
    assert stored["answers"] == {qid: 0} and stored["questions"][qid]["simplified"] is None


def test_add_question_errors():
    rounds = FakeRounds()
    _, body = start_round([doc()], rounds=rounds)
    url = f"/v1/game/rounds/{body['round_id']}/questions"
    qid = str(ObjectId())

    missing = client_with(rounds=rounds).post(url, json={"question_id": qid})
    assert missing.status_code == 404 and missing.json()["detail"] == "question not found"
    unplayable = doc(_id=ObjectId(qid), correct_response=["a", "b"])
    multi = client_with(exam=[unplayable], rounds=rounds)
    r = multi.post(url, json={"question_id": qid})
    assert r.status_code == 422 and r.json()["detail"] == "question is not playable"
    assert multi.post(url, json={"question_id": "$where"}).status_code == 422
    c = client_with(exam=[doc(_id=ObjectId(qid))], rounds=rounds)
    unknown = c.post(f"/v1/game/rounds/{ObjectId()}/questions", json={"question_id": qid})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "round not found"


# ---------- error body ----------


def test_errors_carry_a_code():
    a = doc()
    c, body = start_round([a])
    url = f"/v1/game/rounds/{body['round_id']}/answers"
    qid = str(a["_id"])
    c.post(url, json={"question_id": qid, "answer_index": 0})
    other = c.post(url, json={"question_id": qid, "answer_index": 1})
    assert other.json() == {"detail": "question already answered", "code": "already_answered"}
    missing = c.post(f"/v1/game/rounds/{ObjectId()}/answers", json={"question_id": qid, "answer_index": 0})
    assert missing.json()["code"] == "round_not_found"


def test_validation_error_detail_is_a_string():
    r = client_with().post("/v1/game/rounds", json={"count": 0})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error" and isinstance(body["detail"], str)
    assert "count" in body["detail"] and body["errors"][0]["loc"] == ["body", "count"]


# ---------- rounds ----------


def test_round_with_no_playable_question_is_404_and_not_stored():
    rounds = FakeRounds()
    c = client_with(exam=[doc(correct_response=["a", "b"])], rounds=rounds)
    r = c.post("/v1/game/rounds", json={"count": 10})
    assert r.status_code == 404 and r.json()["code"] == "no_questions"
    assert rounds.docs == {}


def test_round_oversamples_and_trims_to_count():
    exam = [doc(correct_response=["z"])] + [doc() for _ in range(3)]
    c = client_with(exam=exam)
    body = c.post("/v1/game/rounds", json={"count": 2}).json()
    # The unplayable first sample is skipped, and the round is still full.
    assert [q["id"] for q in body["items"]] == [str(d["_id"]) for d in exam[1:3]]


def test_versioned_routes_serve_the_game():
    a = doc()
    c = client_with(exam=[a])
    r = c.post("/v1/game/rounds", json={"count": 1})
    assert r.status_code == 201
    answer = c.post(
        f"/v1/game/rounds/{r.json()['round_id']}/answers",
        json={"question_id": str(a["_id"]), "answer_index": 1},
    )
    assert answer.status_code == 200 and answer.json()["is_correct"]


# ---------- lost answer response ----------


def test_recorded_answer_recovers_a_lost_response():
    a = doc()
    c, body = start_round([a], simple=[simplified_of(a)])
    base = f"/v1/game/rounds/{body['round_id']}"
    qid = str(a["_id"])

    first = c.post(f"{base}/answers", json={"question_id": qid, "answer_index": 0}).json()
    # The response above "was lost"; the player picks another card.
    assert c.post(f"{base}/answers", json={"question_id": qid, "answer_index": 2}).status_code == 409
    recorded = c.get(f"{base}/answers/{qid}")
    assert recorded.status_code == 200 and recorded.json() == first


def test_recorded_answer_errors():
    a, b = doc(), doc()
    c, body = start_round([a, b])
    base = f"/v1/game/rounds/{body['round_id']}"
    unanswered = c.get(f"{base}/answers/{a['_id']}")
    assert unanswered.status_code == 404 and unanswered.json()["code"] == "question_not_answered"
    assert c.get(f"{base}/answers/{ObjectId()}").json()["code"] == "question_not_in_round"
    assert c.get(f"/v1/game/rounds/{ObjectId()}/answers/{a['_id']}").json()["code"] == "round_not_found"
    assert c.get(f"{base}/answers/nope").status_code == 422


# ---------- batches: one round, no repeats ----------


def test_batches_continue_the_round_without_repeats():
    exam = [doc() for _ in range(5)]
    ids = [str(d["_id"]) for d in exam]
    rounds = FakeRounds()
    c = client_with(exam=exam, simple=[simplified_of(exam[4])], rounds=rounds)
    body = c.post("/v1/game/rounds", json={"count": 2}).json()
    base = f"/v1/game/rounds/{body['round_id']}"
    assert [q["id"] for q in body["items"]] == ids[:2]

    # A search result joins the round, so no batch may bring it again.
    assert c.post(f"{base}/questions", json={"question_id": ids[2]}).status_code == 200

    first = c.post(f"{base}/batches", json={"batch_index": 1, "count": 10})
    assert first.status_code == 200, first.text
    batch = first.json()
    assert batch["batch_index"] == 1 and batch["remaining"] == 0
    assert [q["id"] for q in batch["items"]] == ids[3:]
    assert [q["id"] for q in batch["simplified"]] == [ids[4]]
    assert "correct" not in str(batch) and "feedback" not in str(batch)

    # Batch questions can be answered in the round.
    answer = c.post(f"{base}/answers", json={"question_id": ids[4], "answer_index": 1})
    assert answer.json()["is_correct"] and answer.json()["has_simplified"]

    # A retry of batch 1 returns it again; batch 2 finds the pool used up.
    assert c.post(f"{base}/batches", json={"batch_index": 1, "count": 10}).json() == batch
    empty = c.post(f"{base}/batches", json={"batch_index": 2, "count": 10}).json()
    assert empty["items"] == [] and empty["remaining"] == 0
    stored = next(iter(rounds.docs.values()))
    assert stored["question_ids"] == ids and stored["batches"] == [ids[:2], ids[3:]]


def test_batches_keep_the_round_filters():
    exam = [doc(section="A"), doc(section="B"), doc(section="A")]
    c = client_with(exam=exam)
    body = c.post("/v1/game/rounds", json={"count": 1, "section": "A"}).json()
    batch = c.post(f"/v1/game/rounds/{body['round_id']}/batches", json={"batch_index": 1}).json()
    assert [q["id"] for q in batch["items"]] == [str(exam[2]["_id"])]


def test_batch_errors():
    c, body = start_round([doc(), doc()])
    base = f"/v1/game/rounds/{body['round_id']}"
    skipped = c.post(f"{base}/batches", json={"batch_index": 2})
    assert skipped.status_code == 409 and skipped.json()["code"] == "batch_out_of_order"
    assert c.post(f"{base}/batches", json={"batch_index": 0}).status_code == 422
    missing = c.post(f"/v1/game/rounds/{ObjectId()}/batches", json={"batch_index": 1})
    assert missing.status_code == 404 and missing.json()["code"] == "round_not_found"


def test_round_is_capped(monkeypatch):
    monkeypatch.setattr(repository, "MAX_ROUND_QUESTIONS", 2)
    monkeypatch.setattr(service, "MAX_ROUND_QUESTIONS", 2)
    exam = [doc() for _ in range(4)]
    c = client_with(exam=exam)
    body = c.post("/v1/game/rounds", json={"count": 1}).json()
    base = f"/v1/game/rounds/{body['round_id']}"

    batch = c.post(f"{base}/batches", json={"batch_index": 1, "count": 10}).json()
    assert len(batch["items"]) == 1  # Trimmed to the room left.
    full = c.post(f"{base}/questions", json={"question_id": str(exam[3]["_id"])})
    assert full.status_code == 409 and full.json()["code"] == "round_full"
    assert c.post(f"{base}/batches", json={"batch_index": 2}).json()["code"] == "round_full"
    # A question already in the round is still returned.
    again = c.post(f"{base}/questions", json={"question_id": str(exam[0]["_id"])})
    assert again.status_code == 200


# ---------- browsing ----------


def test_browse_lists_only_playable_questions():
    good, five = doc(), doc(prompt={"question": "q", "answers": list("abcde")})
    items = client_with(exam=[five, good]).get("/v1/game/questions").json()["items"]
    assert [q["id"] for q in items] == [str(good["_id"])]


def test_simplified_question_by_id():
    a = doc()
    c = client_with(simple=[simplified_of(a)])
    r = c.get(f"/v1/game/simplified-questions/{a['_id']}")
    assert r.status_code == 200 and r.json()["simplified"]
    assert c.get(f"/v1/game/simplified-questions/{ObjectId()}").json()["code"] == "simplified_not_found"


# ---------- guards ----------


def test_round_creation_is_rate_limited(monkeypatch):
    monkeypatch.setattr(round_rate_limit, "times", 2)
    c = client_with(exam=[doc()])
    codes = [c.post("/v1/game/rounds", json={"count": 1}).status_code for _ in range(3)]
    assert codes == [201, 201, 429]


def test_api_key_guards_study_routes_only():
    app.state.api_key = "secret"
    c = client_with(exam=[doc()])
    assert c.get("/questions/").status_code == 401
    assert c.get("/questions/", headers={"X-API-Key": "wrong"}).json()["code"] == "invalid_api_key"
    assert c.post("/v1/game/rounds", json={"count": 1}).status_code == 201


def test_unversioned_paths_are_gone():
    c = client_with(exam=[doc()])
    assert c.get("/game/sections").status_code == 404
    assert c.get("/search/game-questions", params={"q": "ok"}).status_code == 404
    assert c.get("/v1/game/sections").status_code == 200
