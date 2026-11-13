"""Collect F1 data from Jolpica, OpenF1 and (optionally) FastF1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.data_collection.coverage import build_coverage_report  # noqa: E402
from src.data_collection.fastf1_collector import FastF1Collector  # noqa: E402
from src.data_collection.jolpica_collector import JolpicaCollector  # noqa: E402
from src.data_collection.openf1_collector import OpenF1Collector  # noqa: E402
from src.utils import ensure_dirs, get_logger, save_json, utc_now_iso  # noqa: E402

LOG = get_logger("scripts.collect_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect F1 historical data")
    parser.add_argument("--start", type=int, default=config.SEASON_START)
    parser.add_argument("--end", type=int, default=config.SEASON_END)
    parser.add_argument("--update", action="store_true", help="Refresh recent seasons only")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--skip-openf1", action="store_true")
    parser.add_argument("--skip-fastf1", action="store_true")
    parser.add_argument(
        "--fastf1-schedules",
        action="store_true",
        help="Collect FastF1 event schedules (cheap; requires fastf1)",
    )
    parser.add_argument(
        "--openf1-max-sessions",
        type=int,
        default=None,
        help="Limit OpenF1 race sessions per season (useful for smoke tests)",
    )
    parser.add_argument(
        "--openf1-endpoints",
        default="weather,drivers,position,intervals,pit,stints,laps,race_control",
        help="Comma-separated OpenF1 endpoints",
    )
    args = parser.parse_args()

    config.ensure_directories()
    ensure_dirs(config.METADATA_DIR, config.REPORTS_DIR)
    report: dict = {"started_at": utc_now_iso(), "arguments": vars(args), "sources": {}}

    jolpica = JolpicaCollector()
    if args.update:
        start = max(args.start, args.end - 2)
        LOG.info("Updating Jolpica seasons %s-%s", start, args.end)
        summary = jolpica.collect(start=start, end=args.end, update=True, force=args.force)
    else:
        LOG.info("Collecting Jolpica seasons %s-%s", args.start, args.end)
        summary = jolpica.collect(start=args.start, end=args.end, force=args.force)
    report["sources"]["jolpica"] = summary.as_dict()

    if not args.skip_openf1 and args.end >= config.OPENF1_START_SEASON:
        openf1 = OpenF1Collector()
        endpoints = tuple(item.strip() for item in args.openf1_endpoints.split(",") if item.strip())
        LOG.info("Collecting OpenF1 seasons %s-%s", max(args.start, config.OPENF1_START_SEASON), args.end)
        openf1_summary = openf1.collect(
            start=max(args.start, config.OPENF1_START_SEASON),
            end=args.end,
            update=args.update,
            force=args.force,
            endpoints=endpoints,
            max_sessions=args.openf1_max_sessions,
        )
        report["sources"]["openf1"] = openf1_summary.as_dict()

    if args.fastf1_schedules and not args.skip_fastf1:
        try:
            fastf1 = FastF1Collector()
            written = fastf1.collect_event_schedules(
                start=max(args.start, config.FASTF1_START_SEASON),
                end=args.end,
                force=args.force,
            )
            report["sources"]["fastf1_schedules"] = {"files_written": written}
        except Exception as exc:  # noqa: BLE001 - FastF1 is optional
            LOG.warning("FastF1 schedule collection skipped: %s", exc)
            report["sources"]["fastf1_schedules"] = {"error": str(exc)}

    report["coverage"] = build_coverage_report(args.start, args.end, write=True)
    report["finished_at"] = utc_now_iso()
    save_json(config.METADATA_DIR / "collection_report.json", report)
    LOG.info("Collection complete. Coverage report: %s", config.REPORTS_DIR / "data_coverage.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
