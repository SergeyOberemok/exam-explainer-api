import pytest
from fakes import FakeCollection

from claude_exam.api.dependencies import (
    get_async_decomposition_collection,
    get_async_game_question_collection,
    get_async_objectives_collection,
    get_async_story_collection,
)
from claude_exam.api.main import app
from claude_exam.api.security import ALL_RATE_LIMITS


@pytest.fixture(autouse=True)
def fresh_app_state():
    """Rate-limit counters and the API key live on module/app state; reset them per test.
    No question is tagged with objectives or has a story record unless a test supplies
    the collection."""
    for limiter in ALL_RATE_LIMITS:
        limiter.reset()
    app.state.api_key = None
    app.dependency_overrides[get_async_objectives_collection] = lambda: FakeCollection([])
    app.dependency_overrides[get_async_story_collection] = lambda: FakeCollection([])
    app.dependency_overrides[get_async_decomposition_collection] = lambda: FakeCollection([])
    app.dependency_overrides[get_async_game_question_collection] = lambda: FakeCollection([])
    yield
    app.state.api_key = None
    app.dependency_overrides.clear()
