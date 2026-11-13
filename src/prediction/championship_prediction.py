"""Championship projection for a season, built on the classification model."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

import config
from src.features.feature_engineering import FeatureEngineer
from src.models.championship_model import ChampionshipModel, RaceSimulationInput
from src.models.classification_model import ClassificationModel
from src.utils import get_logger

LOG = get_logger("prediction.championship")


def _season_results(engineer: FeatureEngineer, season: int) -> pd.DataFrame:
    results = engineer.results.copy()
    if results.empty:
        return results
    return results[results["season"].astype(int) == int(season)].copy()


def _points_tables(
    engineer: FeatureEngineer,
    season: int,
    cutoff_order: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    results = _season_results(engineer, season)
    if results.empty:
        return {}, {}, {}, {}, {}
    results["race_id"] = results["race_id"].astype(str)
    race_orders = results["race_id"].map(engineer.race_order)
    included = results[race_orders <= cutoff_order].copy()
    if included.empty:
        return {}, {}, {}, {}, {}
    driver_points = (
        included.groupby("driver_id")["total_points"].sum().astype(float).to_dict()
    )
    constructor_points = (
        included.groupby("constructor_id")["total_points"].sum().astype(float).to_dict()
    )
    latest = included.sort_values("race_id").groupby("driver_id").tail(1)
    driver_constructors = dict(
        zip(latest["driver_id"].astype(str), latest["constructor_id"].astype(str))
    )
    driver_names = {
        str(row["driver_id"]): row.get("driver_name")
        for _, row in latest.iterrows()
        if row.get("driver_id") is not None
    }
    constructor_names = {
        str(row["constructor_id"]): row.get("constructor_name")
        for _, row in latest.iterrows()
        if row.get("constructor_id") is not None
    }
    return driver_points, constructor_points, driver_constructors, driver_names, constructor_names


def project_championship(
    engineer: FeatureEngineer,
    classification_model: ClassificationModel,
    feature_columns: Sequence[str],
    season: int,
    mode: str = "after_qualifying",
    *,
    points_through_order: int | None = None,
    simulate_from_order: int | None = None,
    n_simulations: int = config.CHAMPIONSHIP_SIMULATIONS,
) -> dict[str, Any]:
    season_races = engineer.races[engineer.races["season"].astype(int) == int(season)].sort_values("_order")
    if season_races.empty:
        raise ValueError(f"No races found for season {season}")
    results = _season_results(engineer, season)
    completed_orders = []
    if not results.empty:
        completed_orders = [
            int(engineer.race_order[str(race_id)])
            for race_id in results["race_id"].astype(str).unique()
            if str(race_id) in engineer.race_order
        ]
    if points_through_order is None:
        points_through_order = max(completed_orders) if completed_orders else -1
    if simulate_from_order is None:
        simulate_from_order = points_through_order + 1

    driver_points, constructor_points, driver_constructors, driver_names, constructor_names = (
        _points_tables(engineer, season, points_through_order)
    )

    simulation_races: list[RaceSimulationInput] = []
    skipped: list[str] = []
    for _, race in season_races.iterrows():
        order = int(race["_order"])
        if order < simulate_from_order:
            continue
        race_id = str(race["race_id"])
        try:
            features, info = engineer.build_prediction_features(race_id, mode=mode)
        except Exception as exc:  # noqa: BLE001 - missing entries for a future race
            LOG.warning("Cannot build features for %s: %s", race_id, exc)
            skipped.append(race_id)
            continue
        if features.empty:
            skipped.append(race_id)
            continue
        scores = classification_model.predict_scores(features, feature_columns)
        constructors = dict(
            zip(features["driver_id"].astype(str), features["constructor_id"].astype(str))
        )
        simulation_races.append(
            RaceSimulationInput(
                race_id=race_id,
                driver_ids=features["driver_id"].astype(str).tolist(),
                scores=scores,
                sprint=bool(info.get("is_sprint_weekend", False)),
                driver_constructors=constructors,
            )
        )

    if not simulation_races:
        # The season is complete: the current leader is the champion with
        # probability one.  This is not a model prediction, it is arithmetic on
        # the actual points table, and the metadata says so.
        if not driver_points:
            raise RuntimeError(f"No completed results or remaining races for season {season}")
        driver_table = pd.DataFrame(
            [
                {
                    "driver_id": driver,
                    "driver_name": driver_names.get(driver, driver),
                    "constructor_id": driver_constructors.get(driver),
                    "constructor_name": constructor_names.get(driver_constructors.get(driver), driver_constructors.get(driver)),
                    "current_points": points,
                    "expected_final_points": points,
                    "std_final_points": 0.0,
                    "championship_probability": float(points == max(driver_points.values())),
                    "top3_probability": float(points >= sorted(driver_points.values(), reverse=True)[:3][-1]),
                    "expected_final_position": float(
                        sorted(driver_points.values(), reverse=True).index(points) + 1
                    ),
                }
                for driver, points in driver_points.items()
            ]
        ).sort_values("current_points", ascending=False).reset_index(drop=True)
        constructor_table = pd.DataFrame(
            [
                {
                    "constructor_id": constructor,
                    "constructor_name": constructor_names.get(constructor, constructor),
                    "current_points": points,
                    "expected_final_points": points,
                    "std_final_points": 0.0,
                    "championship_probability": float(points == max(constructor_points.values())),
                    "top3_probability": float(points >= sorted(constructor_points.values(), reverse=True)[:3][-1]),
                    "expected_final_position": float(
                        sorted(constructor_points.values(), reverse=True).index(points) + 1
                    ),
                }
                for constructor, points in constructor_points.items()
            ]
        ).sort_values("current_points", ascending=False).reset_index(drop=True)
        return {
            "drivers": driver_table,
            "constructors": constructor_table,
            "metadata": {
                "season": int(season),
                "mode": mode,
                "n_simulations": 0,
                "races_simulated": [],
                "skipped_races": skipped,
                "points_through_order": points_through_order,
                "simulate_from_order": simulate_from_order,
                "complete_season": True,
                "notes": ["Season complete: final points are read from actual results."],
            },
        }

    simulator = ChampionshipModel(n_simulations=int(n_simulations))
    result = simulator.simulate(
        int(season),
        simulation_races,
        current_driver_points=driver_points,
        current_constructor_points=constructor_points,
        driver_names=driver_names,
        constructor_names=constructor_names,
        driver_constructors=driver_constructors,
    )
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "mode": mode,
            "points_through_order": int(points_through_order),
            "simulate_from_order": int(simulate_from_order),
            "skipped_races": skipped,
            "complete_season": False,
        }
    )
    return {
        "drivers": result["drivers"],
        "constructors": result["constructors"],
        "metadata": metadata,
    }
