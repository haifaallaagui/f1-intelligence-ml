"""Shared fixtures and synthetic (non-fabricated) test data.

The tests never call a live API.  External services are replaced by fake
sessions/modules, and the feature tests use a small, deterministic processed
dataset whose *values* are synthetic but whose *schema* mirrors the real
cleaning output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLUMNS = [
    "form_score",
    "grid_position",
    "quali_position",
    "driver_championship_position",
    "driver_last5_avg_finish",
    "driver_career_avg_finish",
    "field_size",
    "season",
    "round",
]


def make_driver_frame(
    seasons: tuple[int, ...] = (2016, 2017, 2018),
    races_per_season: int = 4,
    n_drivers: int = 6,
    seed: int = 11,
) -> pd.DataFrame:
    """Deterministic driver-race frame with a learnable strength signal."""
    rng = np.random.default_rng(seed)
    strength = np.linspace(1.0, 0.0, n_drivers)
    rows: list[dict] = []
    for season in seasons:
        for rnd in range(1, races_per_season + 1):
            race_id = f"{season}_{rnd:02d}"
            noisy = strength + rng.normal(0.0, 0.12, n_drivers)
            order = np.argsort(-noisy)
            for position, driver_index in enumerate(order, start=1):
                driver_id = f"driver_{driver_index:02d}"
                points = float(max(0, 26 - position * 2))
                rows.append(
                    {
                        "race_id": race_id,
                        "season": season,
                        "round": rnd,
                        "driver_id": driver_id,
                        "driver_name": f"Driver {driver_index}",
                        "constructor_id": f"team_{driver_index // 2}",
                        "constructor_name": f"Team {driver_index // 2}",
                        "finish_position": position,
                        "classified_position": position,
                        "classified": True,
                        "win": int(position == 1),
                        "podium": int(position <= 3),
                        "top10": int(position <= 10),
                        "dnf": 0,
                        "total_points": points,
                        "field_size": n_drivers,
                        "form_score": float(strength[driver_index] + rng.normal(0, 0.05)),
                        "grid_position": float(position + rng.integers(-1, 2)),
                        "quali_position": float(position),
                        "driver_championship_position": float(position),
                        "driver_last5_avg_finish": float(position + rng.normal(0, 0.5)),
                        "driver_career_avg_finish": float(position + rng.normal(0, 0.5)),
                    }
                )
    return pd.DataFrame(rows)


def make_constructor_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in (2016, 2017, 2018):
        for rnd in range(1, 5):
            for team in range(3):
                base = 30.0 - team * 8.0
                rows.append(
                    {
                        "race_id": f"{season}_{rnd:02d}",
                        "season": season,
                        "round": rnd,
                        "constructor_id": f"team_{team}",
                        "constructor_name": f"Team {team}",
                        "field_size": 6,
                        "form_score": float(base + rng.normal(0, 1.0)),
                        "constructor_championship_position": float(team + 1),
                        "team_total_points": float(max(0.0, base + rng.normal(0, 3.0))),
                        "team_best_finish": float(team * 2 + 1 + rng.normal(0, 0.2)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def driver_frame() -> pd.DataFrame:
    return make_driver_frame()


@pytest.fixture
def constructor_frame() -> pd.DataFrame:
    return make_constructor_frame()


def _make_processed_tables() -> dict[str, pd.DataFrame]:
    """Minimal processed tables matching ``src.preprocessing.cleaning`` output."""
    seasons = (2016, 2017)
    races_per_season = 3
    drivers = [
        ("driver_a", "Driver A", "team_1", "Team 1", "1990-01-01"),
        ("driver_b", "Driver B", "team_1", "Team 1", "1992-02-02"),
        ("driver_c", "Driver C", "team_2", "Team 2", "1988-03-03"),
        ("driver_d", "Driver D", "team_2", "Team 2", "1994-04-04"),
    ]
    # A fixed strength order per race makes the leakage assertions meaningful.
    race_orders = [
        ["driver_a", "driver_c", "driver_b", "driver_d"],
        ["driver_a", "driver_b", "driver_c", "driver_d"],
        ["driver_c", "driver_a", "driver_d", "driver_b"],
    ]
    race_rows: list[dict] = []
    result_rows: list[dict] = []
    quali_rows: list[dict] = []
    for season in seasons:
        for rnd in range(1, races_per_season + 1):
            race_id = f"{season}_{rnd:02d}"
            race_rows.append(
                {
                    "race_id": race_id,
                    "season": season,
                    "round": rnd,
                    "race_name": f"Test Grand Prix {rnd}",
                    "race_date": f"{season}-0{rnd}-15",
                    "circuit_id": "circuit_1" if rnd % 2 else "circuit_2",
                    "circuit_name": "Test Circuit",
                    "total_rounds": races_per_season,
                    "is_sprint_weekend": False,
                    "latitude": 1.0,
                    "longitude": 2.0,
                    "altitude": 10.0,
                }
            )
            order = race_orders[(rnd - 1) % len(race_orders)]
            for position, driver_id in enumerate(order, start=1):
                driver = next(item for item in drivers if item[0] == driver_id)
                points = float(max(0, 11 - position))
                # A deliberate grid penalty for driver_d in the first race.
                grid = position
                if race_id == "2016_01" and driver_id == "driver_d":
                    grid = position + 2
                result_rows.append(
                    {
                        "race_id": race_id,
                        "season": season,
                        "round": rnd,
                        "driver_id": driver_id,
                        "driver_name": driver[1],
                        "constructor_id": driver[2],
                        "constructor_name": driver[3],
                        "driver_dob": driver[4],
                        "finish_position": position,
                        "classified_position": position,
                        "classified": True,
                        "is_dnf": False,
                        "grid_raw": grid,
                        "grid_position": grid,
                        "position_gain": grid - position,
                        "points": points,
                        "race_points": points,
                        "sprint_points": 0.0,
                        "total_points": points,
                        "win": int(position == 1),
                        "podium": int(position <= 3),
                        "top10": int(position <= 10),
                        "laps": 50.0,
                        "fastest_lap_rank": 1 if position == 1 else 2,
                    }
                )
                quali_rows.append(
                    {
                        "race_id": race_id,
                        "driver_id": driver_id,
                        "quali_position": position,
                        "quali_gap_to_pole_seconds": float(position - 1) * 0.1,
                        "quali_gap_to_pole_pct": float(position - 1) * 0.001,
                        "quali_gap_to_teammate_seconds": 0.05,
                        "quali_gap_to_teammate_pct": 0.0005,
                        "quali_reached_q2": int(position <= 3),
                        "quali_reached_q3": int(position <= 2),
                    }
                )
    races = pd.DataFrame(race_rows)
    results = pd.DataFrame(result_rows)
    qualifying = pd.DataFrame(quali_rows)
    reference_drivers = pd.DataFrame(
        {
            "driver_id": [d[0] for d in drivers],
            "date_of_birth": [d[4] for d in drivers],
        }
    )
    constructors = pd.DataFrame(
        {
            "constructor_id": ["team_1", "team_2"],
            "constructor_name": ["Team 1", "Team 2"],
        }
    )
    circuits = pd.DataFrame(
        {"circuit_id": ["circuit_1", "circuit_2"], "circuit_name": ["Circuit 1", "Circuit 2"]}
    )
    empty = pd.DataFrame()
    return {
        "races": races,
        "results": results,
        "qualifying": qualifying,
        "sprints": empty.copy(),
        "drivers": reference_drivers,
        "constructors": constructors,
        "circuits": circuits,
        "status": empty.copy(),
        "driver_standings": empty.copy(),
        "constructor_standings": empty.copy(),
        "openf1_race_summary": empty.copy(),
        "openf1_driver_race_summary": empty.copy(),
        "fastf1_qualifying_summary": empty.copy(),
    }


@pytest.fixture
def processed_tables() -> dict[str, pd.DataFrame]:
    return _make_processed_tables()
