from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from config import AgentSettings, ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent-healthcheck",
        description="Validate agent configuration and optionally probe /readiness.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8088/readiness",
        help="Readiness endpoint to probe. Use an empty string to skip the HTTP probe.",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = AgentSettings.from_env()
    except ConfigError as exc:
        print(json.dumps({"status": "unhealthy", "error": str(exc)}))
        return 1

    result = {
        "status": "healthy",
        "model_deployment": settings.model_deployment,
        "toolbox_name": settings.toolbox_name,
        "project_endpoint": settings.project_endpoint,
    }

    if args.url:
        try:
            with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"readiness returned HTTP {response.status}")
        except (urllib.error.URLError, RuntimeError) as exc:
            result["status"] = "unhealthy"
            result["error"] = f"readiness probe failed: {exc}"
            print(json.dumps(result))
            return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
