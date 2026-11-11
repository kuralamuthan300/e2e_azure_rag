from __future__ import annotations

import pytest

from rag_agent.config import AgentSettings, ConfigError


def test_agent_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/demo/",
    )
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.setenv("TOOLBOX_NAME", "rag-tools")

    settings = AgentSettings.from_env()

    assert (
        settings.project_endpoint
        == "https://example.services.ai.azure.com/api/projects/demo"
    )
    assert settings.model_deployment == "gpt-5.4-mini"
    assert settings.toolbox_name == "rag-tools"
    assert settings.toolbox_endpoint_env == "TOOLBOX_RAG_TOOLS_MCP_ENDPOINT"


def test_agent_settings_rejects_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.setenv("TOOLBOX_NAME", "rag-tools")

    with pytest.raises(ConfigError, match="FOUNDRY_PROJECT_ENDPOINT"):
        AgentSettings.from_env()


def test_agent_settings_rejects_non_https_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "http://example.com")
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.setenv("TOOLBOX_NAME", "rag-tools")

    with pytest.raises(ConfigError, match="https"):
        AgentSettings.from_env()


def test_agent_settings_rejects_invalid_toolbox_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/demo",
    )
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.setenv("TOOLBOX_NAME", "bad toolbox name!")

    with pytest.raises(ConfigError, match="TOOLBOX_NAME"):
        AgentSettings.from_env()
