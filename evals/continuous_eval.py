"""Enable or verify hourly trace-based evaluation for a hosted Foundry agent.

Evaluation rules (``evaluation_rules``) only support prompt agents. Hosted
agents are monitored through a recurring evaluation schedule that pulls recent
traces from Application Insights, which is the same mechanism the Foundry
continuous-evaluation tooling uses for hosted agents.

The schedule runs against an eval group whose data source is ``azure_ai_traces``
and is scoped to the promoted immutable agent version (``rag-agent:<version>``).
The batch quality gate remains the eval declared in ``azure.eval.yaml``.

Examples:
    # Create or refresh the hourly schedule for the newest deployed version.
    python evals/continuous_eval.py

    # Point the schedule at the version promoted by the current CD run.
    python evals/continuous_eval.py --version 12

    # Inspect the current schedule.
    python evals/continuous_eval.py --check

    # Pause the schedule without deleting it.
    python evals/continuous_eval.py --disable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
from azure.ai.projects.models import (  # type: ignore[import-untyped]
    EvaluationScheduleTask,
    HourlyRecurrenceSchedule,
    RecurrenceTrigger,
    Schedule,
)
from azure.identity import DefaultAzureCredential

DEFAULT_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
TRACE_EVAL_NAME = "rag-agent-continuous"
SCHEDULE_ID = "rag-agent-continuous-eval"
MODEL_DEPLOYMENT = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")


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
    numeric = sorted((version for version in versions if version.isdigit()), key=int)
    if not numeric:
        raise SystemExit(f"Agent {agent!r} has no deployed versions.")
    return numeric[-1]


def _resolve_trace_eval(client: AIProjectClient) -> str:
    openai_client = client.get_openai_client()
    for item in openai_client.evals.list(limit=100):
        if item.name == TRACE_EVAL_NAME:
            return str(item.id)

    criteria = [
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": MODEL_DEPLOYMENT},
            "data_mapping": {
                "query": "{{sample.query}}",
                "response": "{{sample.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "tool_call_accuracy",
            "evaluator_name": "builtin.tool_call_accuracy",
            "initialization_parameters": {"deployment_name": MODEL_DEPLOYMENT},
            "data_mapping": {
                "query": "{{sample.query}}",
                "response": "{{sample.response}}",
                "tool_calls": "{{sample.tool_calls}}",
                "tool_definitions": "{{sample.tool_definitions}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "task_adherence",
            "evaluator_name": "builtin.task_adherence",
            "initialization_parameters": {"deployment_name": MODEL_DEPLOYMENT},
            "data_mapping": {
                "query": "{{sample.query}}",
                "response": "{{sample.response}}",
            },
        },
    ]
    created = openai_client.evals.create(
        name=TRACE_EVAL_NAME,
        data_source_config={"type": "azure_ai_source", "scenario": "traces"},
        testing_criteria=criteria,
    )
    return str(created.id)


def _schedule_summary(schedule: Any) -> dict[str, Any]:
    task = getattr(schedule, "task", None)
    trigger = getattr(schedule, "trigger", None)
    eval_run = getattr(task, "eval_run", None) or {}
    data_source = eval_run.get("data_source", {}) if isinstance(eval_run, dict) else {}
    return {
        "id": getattr(schedule, "schedule_id", None),
        "display_name": getattr(schedule, "display_name", None),
        "enabled": getattr(schedule, "enabled", None),
        "status": getattr(schedule, "provisioning_status", None),
        "trigger": getattr(trigger, "type", None),
        "interval": getattr(trigger, "interval", None),
        "eval_id": getattr(task, "eval_id", None),
        "agent_id": data_source.get("agent_id"),
        "lookback_hours": data_source.get("lookback_hours"),
        "max_traces": data_source.get("max_traces"),
    }


def _get_schedule(client: AIProjectClient) -> Any | None:
    try:
        return client.beta.schedules.get(SCHEDULE_ID)
    except Exception as exc:
        if "not found" in str(exc).lower() or getattr(exc, "status_code", None) == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-endpoint", default=DEFAULT_PROJECT_ENDPOINT)
    parser.add_argument("--agent", default="rag-agent")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--lookback-hours", type=int, default=1)
    parser.add_argument("--max-traces", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    client = _client(args.project_endpoint)
    existing = _get_schedule(client)

    if args.check:
        if existing is None:
            raise SystemExit(f"No continuous evaluation schedule named {SCHEDULE_ID!r} exists.")
        summary = _schedule_summary(existing)
        print(json.dumps(summary) if args.json else json.dumps(summary, indent=2))
        return 0

    version = args.version
    if version == "latest":
        version = _latest_version(client, args.agent)
    if not version.isdigit():
        raise SystemExit(f"Agent version must be numeric or 'latest', got {version!r}.")

    eval_id = _resolve_trace_eval(client)
    schedule = Schedule(
        enabled=not args.disable,
        display_name="RAG agent continuous evaluation",
        description=(
            "Hourly trace evaluation for the promoted RAG agent version. Pulls "
            "recent Application Insights traces and scores relevance, tool call "
            "accuracy, and task adherence."
        ),
        trigger=RecurrenceTrigger(interval=1, schedule=HourlyRecurrenceSchedule()),
        task=EvaluationScheduleTask(
            eval_id=eval_id,
            eval_run={
                "name": f"rag-agent-continuous-{version}",
                "data_source": {
                    "type": "azure_ai_traces",
                    "agent_id": f"{args.agent}:{version}",
                    "lookback_hours": args.lookback_hours,
                    "max_traces": args.max_traces,
                },
            },
        ),
    )

    created = client.beta.schedules.create_or_update(SCHEDULE_ID, schedule)
    verified = client.beta.schedules.get(SCHEDULE_ID)
    summary = _schedule_summary(verified or created)
    print(json.dumps(summary) if args.json else json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
