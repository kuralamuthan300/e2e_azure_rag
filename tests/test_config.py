from __future__ import annotations

import pytest

from rag_ingest.config import ConfigError, Settings


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://search.example.com/")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://openai.example.com/")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "custom-index")

    settings = Settings.from_env()

    assert settings.search_endpoint == "https://search.example.com"
    assert settings.openai_endpoint == "https://openai.example.com"
    assert settings.embedding_deployment == "text-embedding-3-small"
    assert settings.index_name == "custom-index"


def test_settings_rejects_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://openai.example.com")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "embedding")

    with pytest.raises(ConfigError, match="AZURE_SEARCH_ENDPOINT"):
        Settings.from_env()


def test_dry_run_settings_need_no_azure_env() -> None:
    settings = Settings.for_dry_run("local-dry-run")

    assert settings.index_name == "local-dry-run"
    assert settings.search_endpoint == "https://dry-run.invalid"
