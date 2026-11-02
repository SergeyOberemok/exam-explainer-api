from bson import ObjectId
from fastapi.testclient import TestClient

from claude_exam.api.dependencies import (
    get_async_collection,
    get_async_simplified_collection,
)
from claude_exam.api.main import app
from claude_exam.api.routers.game import letter_to_index, to_game_question


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

    async def distinct(self, field):
        return sorted({d[field] for d in self.docs})


def doc(**kw):
    base = {
        "_id": ObjectId(),
        "section": "S",
        "assessment_type": "t",
        "correct_response": ["b"],
        "prompt": {"question": "q", "answers": ["x", "y"], "feedbacks": ["fx"], "explanation": "e"},
    }
    return {**base, **kw}


def test_letter_to_index_tolerates_bad_values():
    assert [letter_to_index(v) for v in ("a", " C ", "ab", "", None, 3)] == [0, 2, -1, -1, -1, -1]


def test_to_game_question_feedback_and_malformed_correct():
    q = to_game_question(doc(correct_response=["b", "ab", None, "z"]))
    assert q.correct_indices == [1]
    assert [a.feedback for a in q.answers] == ["fx", ""]


def test_simplified_uses_question_id():
    q = to_game_question(doc(question_id="abc"), simplified=True)
    assert q.id == "abc" and q.simplified


def client_with(exam=(), simple=()):
    app.dependency_overrides[get_async_collection] = lambda: FakeCollection(list(exam))
    app.dependency_overrides[get_async_simplified_collection] = lambda: FakeCollection(list(simple))
    return TestClient(app)  # no lifespan: not used as a context manager


def teardown_function():
    app.dependency_overrides.clear()


def test_batch_dedupes_sample_duplicates():
    d = doc()
    r = client_with(exam=[d, d]).get("/game/questions?count=2")
    assert r.status_code == 200 and len(r.json()["items"]) == 1


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
