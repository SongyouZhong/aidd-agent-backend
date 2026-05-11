"""Application settings loaded from environment variables.

Per the backend design doc (§11), the source-of-truth for configuration in
production is k3s ConfigMap + Secret. ``pydantic-settings`` here merely reads
whatever is injected into the process environment. For local dev, values come
from a ``.env`` file (kept out of VCS) — this is opt-in and only used when the
file exists.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "AIDD Agent Platform"
    APP_ENV: Literal["dev", "staging", "prod"] = "dev"
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # CORS — comma-separated list of allowed origins.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:61824,http://127.0.0.1:61824"

    # --- PostgreSQL ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "aidd_user"
    POSTGRES_PASSWORD: str = "aidd_password"
    POSTGRES_DB: str = "aidd_db"

    # --- Redis ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    API_CACHE_TTL_SECONDS: int = 7 * 24 * 3600  # 7 days
    ARXIV_CACHE_TTL_SECONDS: int = 3 * 24 * 3600  # 3 days (content refreshes daily)

    # --- SeaweedFS / S3 ---
    S3_ENDPOINT_URL: str = "http://localhost:8333"
    S3_ACCESS_KEY: str = "any"
    S3_SECRET_KEY: str = "any"
    S3_REGION: str = "us-east-1"
    S3_BUCKET: str = "aidd-data"

    # --- Auth / JWT ---
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # --- LLM ---
    LLM_PRIORITY: str = "gemini,deepseek"
    GEMINI_API_KEY: str | None = None
    GEMINI_MODELS: str = "gemini-3-flash-preview"
    QWEN_BASE_URL: str | None = None
    QWEN_MODEL: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    QWEN_API_KEY: str = "EMPTY"  # vLLM ignores by default

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODELS: str = "deepseek-v4-pro,deepseek-v4-flash"

    # --- LLM Context Windows ---
    EXTERNAL_CONTEXT_WINDOW: int = 1_048_576
    LOCAL_CONTEXT_WINDOW: int = 131_072

    # --- Semantic Scholar ---
    SEMANTIC_SCHOLAR_API_KEY: str | None = None

    # --- NCBI E-utilities ---
    # Without a key: 3 req/s; with a key: 10 req/s.
    # Apply at https://www.ncbi.nlm.nih.gov/account/
    NCBI_API_KEY: str | None = None

    # --- LLM fallback (Gemini → local Qwen on 503/429) ---
    LLM_FALLBACK_ENABLED: bool = True
    LLM_CIRCUIT_BREAK_SECONDS: int = 300  # how long to skip primary after a failure

    # --- Auto-Compaction (design doc §9.2) ---
    DISABLE_AUTO_COMPACT: bool = False
    AUTOCOMPACT_THRESHOLD_PERCENT: float = 0.8
    MAX_OUTPUT_TOKENS_FOR_SUMMARY: int = 20_000
    MAX_CONSECUTIVE_COMPACT_FAILURES: int = 3

    # --- Subagent ---
    SUBAGENT_MAX_TURNS: int = 12

    # --- Derived URLs ---
    @property
    def database_url_async(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        """Used by Alembic (sync driver)."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
