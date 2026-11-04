from bson import ObjectId
from fastapi.testclient import TestClient

from claude_exam.api.dependencies import get_async_collection
from claude_exam.api.main import app
from claude_exam.api.search import build_filter, to_hit


def doc(**kw):
    base = {
        "_id": ObjectId(),
        "score": 2.5,
        "section": "Tools",
        "assessment_type": "practice",
        "correct_response": ["b"],
        "prompt": {
            "question": "What caches prompts?",
            "answers": ["nothing", "cache_control", "retries"],
            "explanation": "Use cache_control.",
        },
    }
    base.update(kw)
    return base


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs
        self.limit_value = None

    def sort(self, spec):
        self.sort_spec = spec
        return self

    def limit(self, n):
        self.limit_value = n
        return self

    async def to_list(self):
        return self.docs[: self.limit_value]


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs
        self.last_filter = None
        self.last_projection = None

    def find(self, flt, projection=None):
        self.last_filter, self.last_projection = flt, projection
        return FakeCursor(list(self.docs))


def teardown_function():
    app.dependency_overrides.clear()


def client_with(docs):
    fake = FakeCollection(docs)
    app.dependency_overrides[get_async_collection] = lambda: fake
    return TestClient(app), fake


def test_build_filter_text_only():
    assert build_filter("caching") == {"$text": {"$search": "caching"}}


def test_build_filter_with_filters():
    assert build_filter("x", "Tools", "practice") == {
        "$text": {"$search": "x"},
        "section": "Tools",
        "assessment_type": "practice",
    }


def test_to_hit_resolves_correct_answer_text():
    hit = to_hit(doc())
    assert hit.correct_answers == ["cache_control"]
    assert hit.score == 2.5


def test_to_hit_ignores_bad_letters_and_keeps_multiple():
    hit = to_hit(doc(correct_response=["a", "c", "z", 5]))
    assert hit.correct_answers == ["nothing", "retries"]


def test_search_returns_items():
    client, fake = client_with([doc()])
    r = client.get(
        "/search/questions", params={"q": "prompt caching", "section": "Tools"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["correct_answers"] == ["cache_control"]
    assert fake.last_filter["section"] == "Tools"
    assert fake.last_projection == {"score": {"$meta": "textScore"}}


def test_search_empty():
    client, _ = client_with([])
    r = client.get("/search/questions", params={"q": "nothing here"})
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0}


def test_search_validates_params():
    client, _ = client_with([])
    assert client.get("/search/questions", params={"q": "a"}).status_code == 422
    assert client.get("/search/questions").status_code == 422
    assert (
        client.get("/search/questions", params={"q": "ok", "limit": 51}).status_code
        == 422
    )
