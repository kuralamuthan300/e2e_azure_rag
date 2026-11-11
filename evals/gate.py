from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOWER_IS_BETTER = {"p50_latency_ms", "p95_latency_ms", "errors"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def evaluate(
    report: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    summary = report.get("summary", {})
    rows: list[dict[str, Any]] = []
    passed = True

    for metric, threshold in thresholds.items():
        actual = summary.get(metric)
        if actual is None:
            rows.append(
                {
                    "metric": metric,
                    "actual": None,
                    "threshold": threshold,
                    "passed": False,
                    "reason": "metric missing from report",
                }
            )
            passed = False
            continue

        if metric in LOWER_IS_BETTER:
            ok = float(actual) <= float(threshold)
        else:
            ok = float(actual) >= float(threshold)
        rows.append(
            {
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "passed": ok,
            }
        )
        passed = passed and ok

    return passed, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if RAG metrics miss thresholds.")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "evals" / "reports" / "local-latest.json",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "evals" / "thresholds.yaml",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    thresholds = load_yaml(args.thresholds)
    passed, rows = evaluate(report, thresholds)

    print(f"{'metric':<28} {'actual':>10} {'threshold':>10} {'status':>8}")
    print("-" * 60)
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        actual = row["actual"] if row["actual"] is not None else "missing"
        print(
            f"{row['metric']:<28} {str(actual):>10} "
            f"{str(row['threshold']):>10} {status:>8}"
        )

    result = {
        "passed": passed,
        "report": str(args.report),
        "thresholds": str(args.thresholds),
        "rows": rows,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
