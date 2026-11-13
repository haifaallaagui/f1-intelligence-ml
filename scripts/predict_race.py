"""Predict a single race (winner, podium, full classification, performance, championship)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.prediction.engine import PredictionEngine  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOG = get_logger("scripts.predict")


def _load_entries(path: str | None) -> pd.DataFrame | list[dict] | None:
    if not path:
        return None
    file = Path(path)
    if file.suffix.lower() == ".csv":
        return pd.read_csv(file)
    if file.suffix.lower() in {".json", ".jsonl"}:
        text = file.read_text(encoding="utf-8")
        payload = json.loads(text)
        if isinstance(payload, dict) and "entries" in payload:
            payload = payload["entries"]
        return payload
    raise ValueError("Entries must be a .csv or .json file")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict an F1 race")
    parser.add_argument("--race-id", required=True, help="e.g. 2026_06 or 2026-06")
    parser.add_argument(
        "--mode",
        default="after_qualifying",
        choices=["before_qualifying", "after_qualifying"],
    )
    parser.add_argument("--entries", default=None, help="Optional CSV/JSON with driver_id, constructor_id")
    parser.add_argument("--no-championship", action="store_true")
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--simulations", type=int, default=config.CHAMPIONSHIP_SIMULATIONS)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    engine = PredictionEngine(mode=args.mode)
    result = engine.predict_race(
        args.race_id,
        entries=_load_entries(args.entries),
        include_championship=not args.no_championship,
        n_simulations=args.simulations,
        explain=not args.no_explain,
    )
    output_dir = engine.save_prediction(result, Path(args.output_dir) if args.output_dir else None)

    drivers = result["drivers"].sort_values("predicted_position")
    print(f"\nPrediction for {result['race_info'].get('race_name')} ({result['race_info'].get('race_id')})")
    print(f"Mode: {args.mode} | entries: {result['race_info'].get('entries_source')}")
    print("\nTop 10 classification:")
    for _, row in drivers.head(10).iterrows():
        print(
            f"  P{int(row['predicted_position']):>2}  {row.get('driver_name', row['driver_id']):<24} "
            f"win={float(row.get('winner_probability', 0.0) or 0.0):.1%}  "
            f"podium={float(row.get('podium_probability', 0.0) or 0.0):.1%}  "
            f"E[pts]={float(row.get('expected_points', 0.0) or 0.0):.1f}"
        )
    constructors = result.get("constructors")
    if isinstance(constructors, pd.DataFrame) and not constructors.empty:
        print("\nConstructor projection:")
        for _, row in constructors.sort_values("predicted_team_position").iterrows():
            print(
                f"  P{int(row['predicted_team_position'])}  {row.get('constructor_name', row['constructor_id']):<20} "
                f"E[pts]={float(row.get('expected_team_points', 0.0)):.1f}  "
                f"win={float(row.get('team_win_probability', 0.0)):.1%}"
            )
    championship = result.get("championship")
    if isinstance(championship, dict) and "drivers" in championship:
        print("\nChampionship projection (top 5):")
        for _, row in championship["drivers"].head(5).iterrows():
            print(
                f"  {row.get('driver_name', row['driver_id']):<24} "
                f"champion={float(row.get('championship_probability', 0.0)):.1%}  "
                f"E[pts]={float(row.get('expected_final_points', 0.0)):.1f}"
            )
    print(f"\nSaved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
