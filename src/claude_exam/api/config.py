from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from claude_exam.paths import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str
    ica_api_key: str
    ica_url: str
    ica_assistant: str | None = None
    # When set, the study routes (/questions, /examination, /ica, /v1/search/questions),
    # which return the answer key, require an `X-API-Key` header with this value.
    # The game routes stay open. Unset keeps every route open.
    api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
