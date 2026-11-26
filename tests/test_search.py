from bson import ObjectId
from fakes import FakeCollection as BaseFake
from fastapi.testclient import TestClient

from claude_exam.api.dependencies import get_async_collection, get_async_game_question_collection
from claude_exam.api.main import app
from claude_exam.api.search import build_filter, plain_terms, to_hit


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
    """The study search over `docs`; the game search (`/v2`) serves none of them."""
    fake = FakeCollection(docs)
    app.dependency_overrides[get_async_collection] = lambda: fake
    app.dependency_overrides[get_async_game_question_collection] = lambda: BaseFake([])
    return TestClient(app), fake


def test_build_filter_text_only():
    assert build_filter("caching") == {"$text": {"$search": "caching"}}


def test_build_filter_with_filters():
    assert build_filter("x", "Tools", "practice") == {
        "$text": {"$search": "x"},
        "section": "Tools",
        "assessment_type": "practice",
    }


def test_plain_terms_drops_phrases_and_negation():
    assert plain_terms('Run "claude -p" with --max-turns') == "Run claude p with max-turns"
    assert plain_terms("“smart” — dash −minus") == "smart dash minus"
    assert plain_terms('\\"escaped\\"') == "escaped"


def test_plain_terms_keeps_inner_hyphens_and_words():
    assert plain_terms("multi-agent  tool_use\nv2.1") == "multi-agent tool_use v2.1"


def test_search_sends_plain_terms_to_text_index():
    client, fake = client_with([doc()])
    client.get("/v1/search/questions", params={"q": 'use "cache_control" -retries'})
    assert fake.last_filter["$text"] == {"$search": "use cache_control retries"}


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
        "/v1/search/questions", params={"q": "prompt caching", "section": "Tools"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["correct_answers"] == ["cache_control"]
    assert fake.last_filter["section"] == "Tools"
    assert fake.last_projection == {"score": {"$meta": "textScore"}}


def test_search_empty():
    client, _ = client_with([])
    r = client.get("/v1/search/questions", params={"q": "nothing here"})
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0}


def test_search_validates_params():
    client, _ = client_with([])
    assert client.get("/v1/search/questions", params={"q": "a"}).status_code == 422
    assert client.get("/v1/search/questions").status_code == 422
    assert (
        client.get("/v1/search/questions", params={"q": "ok", "limit": 51}).status_code
        == 422
    )


def test_game_search_validates_params():
    client, _ = client_with([])
    assert client.get("/v2/search/game-questions", params={"q": "a"}).status_code == 422
    assert client.get("/v2/search/game-questions", params={"q": "ok", "limit": 0}).status_code == 422
    whole = "Which approach should you use for caching? " * 40  # ~1.7k chars
    assert client.get("/v2/search/game-questions", params={"q": whole}).status_code == 200
    assert client.get("/v2/search/game-questions", params={"q": "x" * 2001}).status_code == 422


def test_study_search_accepts_a_whole_pasted_question():
    client, _ = client_with([doc()])
    whole = "Which approach should you use for caching? " * 40
    assert client.get("/v1/search/questions", params={"q": whole}).status_code == 200
    assert client.get("/v1/search/questions", params={"q": "x" * 2001}).status_code == 422


def test_post_search_takes_the_fields_in_a_body():
    client, fake = client_with([doc()])
    pasted = "Ein ganzer „Frage“-Text über Caching? " * 45  # far past 8 KB encoded
    r = client.post(
        "/v1/search/questions",
        json={"q": pasted, "section": "Tools", "assessment_type": "practice", "limit": 2},
    )
    assert r.status_code == 200
    assert r.json()["items"][0]["correct_answers"] == ["cache_control"]
    assert fake.last_filter["section"] == "Tools"
    assert fake.last_filter["assessment_type"] == "practice"
    assert "„" not in fake.last_filter["$text"]["$search"]


def test_post_search_validates_the_body():
    client, _ = client_with([])
    for path in ("/v1/search/questions", "/v2/search/game-questions"):
        assert client.post(path, json={"q": "a"}).status_code == 422
        assert client.post(path, json={"q": "x" * 2001}).status_code == 422
        assert client.post(path, json={"q": "ok", "limit": 51}).status_code == 422
        assert client.post(path, json={}).status_code == 422
        r = client.post(path, json={"q": "ok", "limit": 0})
        assert r.json()["code"] == "validation_error"


def test_post_study_search_needs_the_api_key_when_set():
    client, _ = client_with([doc()])
    app.state.api_key = "secret"
    assert client.post("/v1/search/questions", json={"q": "caching"}).status_code == 401
    ok = client.post(
        "/v1/search/questions", json={"q": "caching"}, headers={"X-API-Key": "secret"}
    )
    assert ok.status_code == 200
    assert client.post("/v2/search/game-questions", json={"q": "caching"}).status_code == 200
