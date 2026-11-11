"""Promote or roll back an immutable Foundry agent version.

The agent endpoint uses a fixed-ratio version selector. This script reads the
currently promoted version and can pin the endpoint to a specific version. It
preserves the endpoint's protocol configuration and authorization schemes so a
promotion does not accidentally strip Entra authentication.

Examples:
    python evals/promote.py --current
    python evals/promote.py --version 12
    python evals/promote.py --version 11 --traffic 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
from azure.ai.projects.models import (  # type: ignore[import-untyped]
    AgentEndpointConfig,
    FixedRatioVersionSelectionRule,
    VersionSelector,
)
from azure.identity import DefaultAzureCredential

DEFAULT_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")


def _client(project_endpoint: str) -> AIProjectClient:
    if not project_endpoint:
        raise SystemExit(
            "A Foundry project endpoint is required. Pass --project-endpoint or set "
            "FOUNDRY_PROJECT_ENDPOINT."
        )
    return AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
        allow_preview=True,
    )


def _latest_version(client: AIProjectClient, agent: str) -> str:
    versions = [str(item.version) for item in client.agents.list_versions(agent)]
    numeric = sorted(
        (version for version in versions if version.isdigit()),
        key=int,
    )
    if not numeric:
        raise SystemExit(f"Agent {agent!r} has no deployed versions.")
    return numeric[-1]


def _current_selector(
    client: AIProjectClient,
    agent: str,
) -> dict[str, Any]:
    details = client.agents.get(agent)
    endpoint = details.agent_endpoint
    rules: list[dict[str, Any]] = []
    if endpoint and endpoint.version_selector:
        for rule in endpoint.version_selector.version_selection_rules:
            rules.append(
                {
                    "agent_version": rule.agent_version,
                    "traffic_percentage": getattr(rule, "traffic_percentage", None),
                    "type": getattr(rule, "type", None),
                }
            )

    version = None
    traffic = None
    if rules:
        primary = rules[0]
        version = primary["agent_version"]
        traffic = primary["traffic_percentage"]
        if version == "@latest":
            version = _latest_version(client, agent)

    return {
        "agent": agent,
        "version": version,
        "traffic": traffic,
        "selector": rules,
    }


def _promote(
    client: AIProjectClient,
    agent: str,
    version: str,
    traffic: int,
) -> dict[str, Any]:
    details = client.agents.get(agent)
    current = details.agent_endpoint

    rule = FixedRatioVersionSelectionRule(
        agent_version=version,
        traffic_percentage=traffic,
    )
    config = AgentEndpointConfig(
        version_selector=VersionSelector(version_selection_rules=[rule]),
        protocol_configuration=current.protocol_configuration if current else None,
        authorization_schemes=current.authorization_schemes if current else None,
    )

    updated = client.agents.update_details(agent, agent_endpoint=config)
    endpoint = updated.agent_endpoint
    return {
        "agent": agent,
        "version": version,
        "traffic": traffic,
        "endpoint": endpoint.as_dict() if endpoint else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-endpoint", default=DEFAULT_PROJECT_ENDPOINT)
    parser.add_argument("--agent", default="rag-agent")
    parser.add_argument(
        "--current",
        action="store_true",
        help="Print the currently promoted version instead of changing it.",
    )
    parser.add_argument(
        "--version",
        help="Immutable agent version to promote, or 'latest' to resolve the newest version.",
    )
    parser.add_argument("--traffic", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    client = _client(args.project_endpoint)

    if args.current or not args.version:
        current = _current_selector(client, args.agent)
        print(json.dumps(current) if args.json else json.dumps(current, indent=2))
        return 0

    version = args.version
    if version == "latest":
        version = _latest_version(client, args.agent)

    if not version.isdigit():
        raise SystemExit(f"Agent version must be numeric or 'latest', got {version!r}.")
    if not 0 <= args.traffic <= 100:
        raise SystemExit("--traffic must be between 0 and 100.")

    result = _promote(client, args.agent, version, args.traffic)
    print(json.dumps(result) if args.json else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
