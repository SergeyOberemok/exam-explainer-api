import re
from typing import Any

import httpx
from fastapi import HTTPException

# The ICA gateway rejects non-browser user agents with HTTP 403.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
# Claude's extended thinking consumes max_tokens before the visible answer.
THINKING_RESERVE = 5000
TIMEOUT_SECONDS = 300

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# Models that support extended thinking and need the token reserve.
_THINKING_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}


class IcaClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.strip().rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=TIMEOUT_SECONDS,
        )
        self._assistant_ids: dict[str, str] = {}
        # Separate cache for agent name → id lookups (agents ≠ assistants).
        self._agent_ids: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = await self._http.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="ICA request timed out") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"ICA request failed: {exc}"
            ) from exc

        if response.is_error:
            # ICA reports upstream model rate limits as HTTP 400 with a litellm RateLimitError body.
            rate_limited = response.status_code == 429 or "RateLimitError" in response.text
            status = 429 if rate_limited else 502
            raise HTTPException(
                status_code=status,
                detail=f"ICA returned {response.status_code}: {response.text[:500]}",
            )
        return response.json()

    async def list_assistants(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/assistants")).get("data", [])

    async def list_agents(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/agents")).get("data", [])

    async def resolve_assistant_id(self, name_or_id: str) -> str:
        if UUID_RE.match(name_or_id):
            return name_or_id
        key = name_or_id.lower()
        if key in self._assistant_ids:
            return self._assistant_ids[key]
        for assistant in await self.list_assistants():
            if assistant.get("name", "").lower() == key:
                self._assistant_ids[key] = assistant["id"]
                return assistant["id"]
        raise HTTPException(
            status_code=404, detail=f"ICA assistant not found: {name_or_id!r}"
        )

    async def resolve_agent_id(self, name_or_id: str) -> str:
        """Resolve an agent name or id to its id (cached per name)."""
        if UUID_RE.match(name_or_id):
            return name_or_id
        key = name_or_id.lower()
        if key in self._agent_ids:
            return self._agent_ids[key]
        for agent in await self.list_agents():
            if agent.get("name", "").lower() == key:
                self._agent_ids[key] = agent["id"]
                return agent["id"]
        raise HTTPException(
            status_code=404, detail=f"ICA agent not found: {name_or_id!r}"
        )

    async def chat(
        self,
        assistant_id: str,
        prompt: str,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        # temperature/top_p are omitted on purpose: ICA's Claude only allows the default.
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": assistant_id,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens + THINKING_RESERVE
        return await self._request("POST", "/assistants/chat/completions", json=payload)

    async def agent_chat(
        self, agent_id: str, prompt: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """POST /agents/chat/completions — same shape as assistant chat."""
        payload: dict[str, Any] = {
            "model": agent_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens + THINKING_RESERVE
        return await self._request("POST", "/agents/chat/completions", json=payload)

    async def chat_model(
        self,
        model_id: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """POST /chat-models/chat/completions — raw model, no assistant system prompt."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            # Only sonnet/opus use extended thinking and need the token reserve.
            reserve = THINKING_RESERVE if model_id in _THINKING_MODELS else 0
            payload["max_tokens"] = max_tokens + reserve
        return await self._request("POST", "/chat-models/chat/completions", json=payload)
