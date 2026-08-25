"""Settings. Everything is env-driven so the same image runs api, worker and eval."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- infrastructure -------------------------------------------------
    database_url: str = "postgresql+asyncpg://ragent:ragent@localhost:5432/ragent"
    qdrant_url: str = "http://localhost:6333"
    valkey_url: str = "redis://localhost:6379/0"
    rabbitmq_url: str = "amqp://ragent:ragent@localhost:5672/"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "ragent"
    s3_secret_key: str = "ragentragent"
    s3_bucket: str = "ragent"

    # --- models ---------------------------------------------------------
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # `local` keeps the whole stack runnable with no API keys at all, which is
    # what makes `make up && make seed` work for a reviewer who just cloned this.
    embedding_backend: str = "local"

    model_synthesis: str = "claude-opus-5"
    model_utility: str = "claude-haiku-4-5"
    model_vision: str = "claude-sonnet-5"

    # --- ingest ---------------------------------------------------------
    pipeline_version: str = "v1"

    # Below this mean per-page text-layer confidence we rasterise and OCR.
    # Tuned against the golden set rather than guessed; see evals/harness.
    ocr_confidence_threshold: float = 0.72

    # NoDecode stops pydantic-settings JSON-parsing this before the validator
    # below runs. Without it, CHUNK_STRATEGIES=layout,recursive raises a
    # settings error instead of being read as a comma-separated list.
    chunk_strategies: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["layout"])
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- retrieval ------------------------------------------------------
    retrieval_top_k: int = 50
    rerank_top_n: int = 8
    rrf_k: int = 60
    semantic_cache_threshold: float = 0.96

    log_level: str = "INFO"

    @field_validator("chunk_strategies", mode="before")
    @classmethod
    def _split_strategies(cls, v: object) -> object:
        """Accept CHUNK_STRATEGIES=layout,recursive,fixed as a comma-separated env var."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("ocr_confidence_threshold", "semantic_cache_threshold")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be within [0, 1], got {v}")
        return v

    @property
    def sync_database_url(self) -> str:
        """Alembic and the LangGraph checkpointer want a sync driver."""
        return self.database_url.replace("+asyncpg", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
