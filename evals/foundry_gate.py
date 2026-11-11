from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_thresholds(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate a Foundry eval run by pass rate."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "evals" / "reports" / "foundry-latest.json",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "evals" / "thresholds.yaml",
    )
    args = parser.parse_args(argv)

    report = json.loads(args.run.read_text(encoding="utf-8"))
    thresholds = load_thresholds(args.thresholds)
    run = report.get("run", report)
    counts = run.get("result_counts", {})
    total = int(counts.get("total", 0))
    passed = int(counts.get("passed", 0))
    if total == 0:
        print("No result counts found in Foundry run report.")
        return 1

    pass_rate = passed / total
    threshold = float(thresholds.get("foundry_pass_rate", 0.60))
    ok = pass_rate >= threshold

    print(f"Foundry pass rate: {pass_rate:.1%} ({passed}/{total})")
    print(f"Threshold:         {threshold:.1%}")
    print(f"Status:            {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
