from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_INDEX_NAME = "rag-docs"
DEFAULT_OPENAI_API_VERSION = "2024-10-21"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


class ConfigError(ValueError):
    """Raised when required ingestion configuration is missing or invalid."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for the ingestion CLI."""

    search_endpoint: str
    openai_endpoint: str
    embedding_deployment: str
    index_name: str = DEFAULT_INDEX_NAME
    openai_api_version: str = DEFAULT_OPENAI_API_VERSION
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_batch_size: int = 16
    upload_batch_size: int = 100

    @classmethod
    def from_env(cls, index_name: str | None = None) -> Settings:
        search_endpoint = _required_env("AZURE_SEARCH_ENDPOINT").rstrip("/")
        openai_endpoint = _required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")

        if not search_endpoint.startswith("https://"):
            raise ConfigError("AZURE_SEARCH_ENDPOINT must start with https://")
        if not openai_endpoint.startswith("https://"):
            raise ConfigError("AZURE_OPENAI_ENDPOINT must start with https://")

        resolved_index_name = (
            index_name or os.getenv("AZURE_SEARCH_INDEX_NAME") or DEFAULT_INDEX_NAME
        )

        return cls(
            search_endpoint=search_endpoint,
            openai_endpoint=openai_endpoint,
            embedding_deployment=_required_env("AZURE_EMBEDDING_DEPLOYMENT"),
            index_name=resolved_index_name,
            openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", DEFAULT_OPENAI_API_VERSION
            ),
            embedding_dimensions=_positive_int_env(
                "AZURE_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS
            ),
            embedding_batch_size=_positive_int_env("AZURE_EMBEDDING_BATCH_SIZE", 16),
            upload_batch_size=_positive_int_env("AZURE_SEARCH_UPLOAD_BATCH_SIZE", 100),
        )

    @classmethod
    def for_dry_run(cls, index_name: str | None = None) -> Settings:
        """Create settings that do not require Azure environment variables."""

        return cls(
            search_endpoint="https://dry-run.invalid",
            openai_endpoint="https://dry-run.invalid",
            embedding_deployment="dry-run",
            index_name=(
                index_name or os.getenv("AZURE_SEARCH_INDEX_NAME") or DEFAULT_INDEX_NAME
            ),
        )
