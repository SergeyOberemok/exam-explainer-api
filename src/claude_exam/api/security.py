"""Route guards: an optional API key for the study routes and a per-client rate limit."""

import secrets
import time
from collections import defaultdict, deque
from typing import Annotated

from fastapi import Header, Request

from claude_exam.api.errors import ApiError


class InvalidApiKey(ApiError):
    status_code = 401
    code = "invalid_api_key"
    detail = "missing or invalid X-API-Key header"


class RateLimited(ApiError):
    status_code = 429
    code = "rate_limited"
    detail = "too many requests, slow down"


def require_api_key(
    request: Request, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """Checks `X-API-Key` when the app has a key (`app.state.api_key`, from the `API_KEY`
    setting); without one every request passes, which is also the case in tests."""
    expected = getattr(request.app.state, "api_key", None)
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise InvalidApiKey()


class RateLimiter:
    """At most `times` requests per `seconds` per client address, as a dependency.

    Kept in process memory: each worker counts on its own, and behind a reverse proxy the
    client address is only right when the server trusts the proxy's forwarded headers
    (`fastapi run --proxy-headers --forwarded-allow-ips=...`).
    """

    def __init__(self, times: int, seconds: float) -> None:
        self.times = times
        self.seconds = seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.seconds:
            hits.popleft()
        if len(hits) >= self.times:
            raise RateLimited()
        hits.append(now)
        if len(self._hits) > 10_000:  # Forget idle clients so memory stays bounded.
            cutoff = now - self.seconds
            for idle in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                del self._hits[idle]

    def reset(self) -> None:
        self._hits.clear()


# Starting a round samples the collection and writes a document; search runs a $text query.
round_rate_limit = RateLimiter(times=30, seconds=60)
search_rate_limit = RateLimiter(times=120, seconds=60)
ALL_RATE_LIMITS = (round_rate_limit, search_rate_limit)
