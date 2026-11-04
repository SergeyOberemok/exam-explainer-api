import copy

from bson import ObjectId
from fastapi.testclient import TestClient

from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_rounds_collection,
    get_async_simplified_collection,
)
from claude_exam.api.main import app
from claude_exam.api.routers.game import (
    FIFTY_FIFTY_CHARGES,
    is_playable,
    letter_to_index,
    to_game_question,
    to_review,
)


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    async def to_list(self):
        return self.docs


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    async def aggregate(self, pipeline):
        return FakeCursor(self.docs)

    async def count_documents(self, query):
        return len(self.docs)

    async def find_one(self, query):
        return self.docs[0] if self.docs else None

    def find(self, query):
        ids = query["question_id"]["$in"]
        return FakeCursor([d for d in self.docs if d["question_id"] in ids])

    async def distinct(self, field):
        return sorted({d[field] for d in self.docs})


MISSING = object()


def get_path(doc, path):
    for part in path.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return MISSING
        doc = doc[part]
    return doc


def set_path(doc, path, value):
    *parents, last = path.split(".")
    for part in parents:
        doc = doc.setdefault(part, {})
    doc[last] = value


def matches(doc, query):
    """The subset of Mongo query operators the round routes use."""
    for path, cond in query.items():
        value = get_path(doc, path)
        if not isinstance(cond, dict):
            if value != cond:
                return False
            continue
        for op, arg in cond.items():
            if op == "$exists" and (value is not MISSING) != arg:
                return False
            if op == "$gt" and (value is MISSING or not value > arg):
                return False
            if op == "$ne":
                if isinstance(value, list) and arg in value:
                    return False
                if value == arg:
                    return False
    return True


class FakeRounds:
    """In-memory `game_rounds`; projections are ignored (the routes read what they project)."""

    def __init__(self):
        self.docs = {}

    async def insert_one(self, doc):
        doc = {"_id": ObjectId(), **copy.deepcopy(doc)}
        self.docs[doc["_id"]] = doc

        class Result:
            inserted_id = doc["_id"]

        return Result()

    async def find_one(self, query, projection=None):
        doc = self.docs.get(query["_id"])
        return copy.deepcopy(doc) if doc and matches(doc, query) else None

    async def find_one_and_update(self, query, update, projection=None, return_document=False):
        doc = self.docs.get(query["_id"])
        if doc is None or not matches(doc, query):
            return None
        before = copy.deepcopy(doc)
        for path, value in update.get("$set", {}).items():
            set_path(doc, path, value)
        for path, value in update.get("$inc", {}).items():
            set_path(doc, path, get_path(doc, path) + value)
        return copy.deepcopy(doc) if return_document else before


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


def teardown_function():
    app.dependency_overrides.clear()


def test_batch_dedupes_sample_duplicates():
    d = doc()
    r = client_with(exam=[d, d]).get("/game/questions?count=2")
    assert r.status_code == 200 and len(r.json()["items"]) == 1


def test_batch_has_no_answer_key():
    item = client_with(exam=[doc()]).get("/game/questions").json()["items"][0]
    assert "correct_indices" not in item and "explanation" not in item
    assert all(set(a) == {"index", "text"} for a in item["answers"])


def test_count_zero_is_422():
    assert client_with().get("/game/questions?count=0").status_code == 422


def test_simplified_missing_is_404():
    r = client_with().get("/game/simplified-questions/nope")
    assert r.status_code == 404 and r.json()["detail"] == "simplified question not found"


def test_bad_object_id_is_404():
    assert client_with().get("/game/questions/not-an-id").status_code == 404


def test_sections_simplified_flag():
    c = client_with(exam=[doc(section="A")], simple=[doc(section="B")])
    assert c.get("/game/sections").json() == {"items": ["A"]}
    assert c.get("/game/sections?simplified=true").json() == {"items": ["B"]}


def start_round(exam, simple=(), rounds=None):
    c = client_with(exam=exam, simple=simple, rounds=rounds)
    r = c.post("/game/rounds", json={"count": 10})
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
    r = c.post("/game/rounds", json={"count": 1, "section": "", "assessment_type": ""})
    assert r.status_code == 201


def test_answer_returns_result_and_reviews_once():
    a = doc()
    c, body = start_round([a], simple=[simplified_of(a)])
    url = f"/game/rounds/{body['round_id']}/answers"
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
    r = c.post(f"/game/rounds/{body['round_id']}/answers", json={"question_id": str(a["_id"]), "answer_index": 1})
    assert r.json()["is_correct"] and not r.json()["has_simplified"]
    assert r.json()["simplified"] == {"explanation": "", "feedbacks": []}


def test_answer_errors():
    a = doc()
    c, body = start_round([a])
    url = f"/game/rounds/{body['round_id']}/answers"
    qid = str(a["_id"])

    assert c.post(url, json={"question_id": qid, "answer_index": 4}).status_code == 422
    assert c.post(url, json={"question_id": qid, "answer_index": -1}).status_code == 422
    other = c.post(url, json={"question_id": str(ObjectId()), "answer_index": 0})
    assert other.status_code == 404 and other.json()["detail"] == "question is not in this round"
    # Not an ObjectId hex string, so it can't reach a Mongo field path.
    assert c.post(url, json={"question_id": "$where.x", "answer_index": 0}).status_code == 422
    unknown = c.post(f"/game/rounds/{ObjectId()}/answers", json={"question_id": qid, "answer_index": 0})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "round not found"
    assert c.post("/game/rounds/nope/answers", json={"question_id": qid, "answer_index": 0}).status_code == 404


def test_fifty_fifty_removes_two_wrong_answers_and_blocks_them():
    a, b = doc(), doc()
    c, body = start_round([a, b])
    base = f"/game/rounds/{body['round_id']}"
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
    base = f"/game/rounds/{body['round_id']}"
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
    base = f"/game/rounds/{body['round_id']}"
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
    url = f"/game/rounds/{body['round_id']}/questions"

    r = c.post(url, json={"question_id": qid})
    assert r.status_code == 200 and r.json()["simplified"] == []
    c.post(f"/game/rounds/{body['round_id']}/answers", json={"question_id": qid, "answer_index": 0})
    assert c.post(url, json={"question_id": qid}).json() == r.json()
    stored = next(iter(rounds.docs.values()))
    assert stored["answers"] == {qid: 0} and stored["questions"][qid]["simplified"] is None


def test_add_question_errors():
    rounds = FakeRounds()
    _, body = start_round([doc()], rounds=rounds)
    url = f"/game/rounds/{body['round_id']}/questions"
    qid = str(ObjectId())

    missing = client_with(rounds=rounds).post(url, json={"question_id": qid})
    assert missing.status_code == 404 and missing.json()["detail"] == "question not found"
    multi = client_with(exam=[doc(correct_response=["a", "b"])], rounds=rounds)
    r = multi.post(url, json={"question_id": qid})
    assert r.status_code == 422 and r.json()["detail"] == "question is not playable"
    assert multi.post(url, json={"question_id": "$where"}).status_code == 422
    c = client_with(exam=[doc()], rounds=rounds)
    unknown = c.post(f"/game/rounds/{ObjectId()}/questions", json={"question_id": qid})
    assert unknown.status_code == 404 and unknown.json()["detail"] == "round not found"
