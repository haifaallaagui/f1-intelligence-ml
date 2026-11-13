"""Import Jolpica's official CSV database dump into the raw data layout.

Usage
-----
    python scripts/import_jolpica_dump.py --dump-dir data/raw/jolpica_dump/extracted
    python scripts/import_jolpica_dump.py --dump-dir ... --start 1995 --end 2026 --force

Download the free delayed dump first:

    curl -L "https://api.jolpi.ca/data/dumps/download/delayed/?dump_type=csv" \\
         -o data/raw/jolpica_dump/jolpica_dump.zip

The importer never fabricates data: missing rounds/sessions simply produce
empty result lists and are reported in the collection coverage report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.data_collection.coverage import build_coverage_report  # noqa: E402
from src.data_collection.jolpica_dump_importer import JolpicaDumpImporter  # noqa: E402
from src.utils import ensure_dirs, save_json, utc_now_iso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the Jolpica CSV dump")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=config.PROJECT_ROOT / "data" / "raw" / "jolpica_dump" / "extracted",
        help="Directory containing formula_one_*.csv",
    )
    parser.add_argument("--start", type=int, default=config.SEASON_START)
    parser.add_argument("--end", type=int, default=config.SEASON_END)
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dump_dir.exists():
        print(f"Dump directory not found: {args.dump_dir}", file=sys.stderr)
        return 2
    ensure_dirs(config.METADATA_DIR, config.REPORTS_DIR)
    importer = JolpicaDumpImporter(args.dump_dir)
    summary = importer.import_seasons(args.start, args.end, force=args.force)
    summary["finished_at"] = utc_now_iso()
    save_json(config.METADATA_DIR / "dump_import_report.json", summary)

    coverage = build_coverage_report(args.start, args.end, write=True)
    save_json(config.METADATA_DIR / "coverage.json", coverage)
    save_json(config.REPORTS_DIR / "data_coverage.json", coverage)

    print(f"Seasons written: {summary['seasons_written']}")
    print(f"Files written: {summary['files_written']}")
    print(f"Coverage report: {config.REPORTS_DIR / 'data_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
