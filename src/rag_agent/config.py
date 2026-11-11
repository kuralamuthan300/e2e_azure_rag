from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_TOOLBOX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ConfigError(ValueError):
    """Raised when the agent environment is incomplete or invalid."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Validated runtime configuration for the hosted agent."""

    project_endpoint: str
    model_deployment: str
    toolbox_name: str

    @classmethod
    def from_env(cls) -> AgentSettings:
        project_endpoint = _required_env("FOUNDRY_PROJECT_ENDPOINT").rstrip("/")
        parsed = urlparse(project_endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ConfigError(
                "FOUNDRY_PROJECT_ENDPOINT must be a valid https:// project endpoint"
            )

        model_deployment = _required_env("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        toolbox_name = _required_env("TOOLBOX_NAME")
        if not _TOOLBOX_NAME.fullmatch(toolbox_name):
            raise ConfigError(
                "TOOLBOX_NAME must be 1-64 characters and contain only "
                "letters, numbers, underscores, or hyphens"
            )

        return cls(
            project_endpoint=project_endpoint,
            model_deployment=model_deployment,
            toolbox_name=toolbox_name,
        )

    @property
    def toolbox_endpoint_env(self) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", self.toolbox_name).upper()
        return f"TOOLBOX_{normalized}_MCP_ENDPOINT"
