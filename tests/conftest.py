import pytest

from claude_exam.api.main import app
from claude_exam.api.security import ALL_RATE_LIMITS


@pytest.fixture(autouse=True)
def fresh_app_state():
    """Rate-limit counters and the API key live on module/app state; reset them per test."""
    for limiter in ALL_RATE_LIMITS:
        limiter.reset()
    app.state.api_key = None
    yield
    app.state.api_key = None
    app.dependency_overrides.clear()
