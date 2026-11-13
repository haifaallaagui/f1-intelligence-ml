"""Train and tune every model family for both prediction modes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.training.train import DRIVER_TASKS, CONSTRUCTOR_TASK, train_all  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOG = get_logger("scripts.train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all F1 prediction models")
    parser.add_argument(
        "--tasks",
        default=",".join(DRIVER_TASKS + [CONSTRUCTOR_TASK]),
        help="Comma-separated tasks",
    )
    parser.add_argument(
        "--modes",
        default="before_qualifying,after_qualifying",
        help="Comma-separated prediction modes",
    )
    parser.add_argument("--n-iter", type=int, default=config.N_TUNING_ITER)
    parser.add_argument("--force", action="store_true", help="Retrain even if the dataset hash is unchanged")
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    summary = train_all(tasks=tasks, modes=modes, n_iter=args.n_iter, force=args.force)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
