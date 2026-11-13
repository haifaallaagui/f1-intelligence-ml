"""Build processed tables and the leakage-safe feature datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.data_collection.coverage import build_coverage_report  # noqa: E402
from src.features.feature_engineering import FeatureEngineer  # noqa: E402
from src.preprocessing.cleaning import build_processed_tables  # noqa: E402
from src.utils import get_logger, load_json  # noqa: E402

LOG = get_logger("scripts.build_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed tables and feature datasets")
    parser.add_argument("--start", type=int, default=config.SEASON_START)
    parser.add_argument("--end", type=int, default=config.SEASON_END)
    parser.add_argument("--skip-openf1", action="store_true")
    parser.add_argument("--skip-fastf1", action="store_true")
    args = parser.parse_args()

    config.ensure_directories()
    tables = build_processed_tables(
        args.start,
        args.end,
        include_openf1=not args.skip_openf1,
        include_fastf1=not args.skip_fastf1,
    )
    engineer = FeatureEngineer(tables=tables)
    dataset, constructor_dataset = engineer.build()
    metadata = load_json(config.DATASET_METADATA_FILE, default={}) or {}
    build_coverage_report(args.start, args.end, write=True)
    LOG.info(
        "Dataset: %s driver-race rows, %s constructor-race rows, %s races, seasons %s",
        len(dataset),
        len(constructor_dataset),
        metadata.get("races_used"),
        metadata.get("seasons"),
    )
    LOG.info("Leakage checks: %s", metadata.get("leakage_checks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
