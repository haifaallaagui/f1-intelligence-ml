"""Evaluate all trained models on the chronological test window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.training.evaluate import evaluate_all  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOG = get_logger("scripts.evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate F1 prediction models")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = evaluate_all()
    if not args.quiet:
        summary = {
            key: value.get("metrics")
            for key, value in report.get("tasks", {}).items()
        }
        print(json.dumps(summary, indent=2, default=str))
    LOG.info("Evaluation report: %s", config.REPORTS_DIR / "evaluation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
