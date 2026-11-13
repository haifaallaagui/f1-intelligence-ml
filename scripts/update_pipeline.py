"""Automatic update and retraining entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.automation.update_pipeline import UpdatePipeline  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOG = get_logger("scripts.update")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the F1 data update / retraining pipeline")
    parser.add_argument("--force", action="store_true", help="Ignore the configured interval")
    parser.add_argument("--no-retrain", action="store_true", help="Update data but do not retrain")
    parser.add_argument("--n-iter", type=int, default=config.N_TUNING_ITER)
    parser.add_argument("--skip-openf1", action="store_true")
    parser.add_argument("--skip-fastf1", action="store_true")
    parser.add_argument("--interval-hours", type=float, default=config.UPDATE_INTERVAL_HOURS)
    args = parser.parse_args()

    pipeline = UpdatePipeline(interval_hours=args.interval_hours)
    result = pipeline.run(
        force=args.force,
        retrain=not args.no_retrain,
        n_iter=args.n_iter,
        skip_openf1=args.skip_openf1,
        skip_fastf1=args.skip_fastf1,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
