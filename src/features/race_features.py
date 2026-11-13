"""Circuit/race history features and static circuit metadata."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.features.driver_features import RunningStats
from src.utils import load_json, safe_div, unwrap


@dataclass
class CircuitHistory:
    circuit_id: str
    stats: RunningStats = field(default_factory=RunningStats)
    races: int = 0
    sum_field_size: float = 0.0
    sum_dnf: float = 0.0
    sum_winning_laps: float = 0.0
    pole_wins: int = 0
    weather_races: int = 0
    sum_air_temp: float = 0.0
    sum_track_temp: float = 0.0
    rain_races: int = 0
    safety_car_races: int = 0
    red_flag_races: int = 0
    pit_n: int = 0
    pit_sum: float = 0.0
    stint_n: int = 0
    stint_sum: float = 0.0

    def features(self, metadata: dict[str, float]) -> dict[str, float]:
        features = {
            "circuit_races_held": float(self.races),
            "circuit_avg_field_size": safe_div(self.sum_field_size, self.races),
            "circuit_avg_dnf_rate": safe_div(self.sum_dnf, self.sum_field_size),
            "circuit_avg_position_gain": safe_div(self.stats.sum_gain, self.stats.gain_count),
            "circuit_pole_win_rate": safe_div(self.pole_wins, self.races),
            "circuit_avg_winning_laps": safe_div(self.sum_winning_laps, self.races),
            "circuit_avg_points": self.stats.points_per_race,
            "circuit_avg_air_temp": safe_div(self.sum_air_temp, self.weather_races),
            "circuit_avg_track_temp": safe_div(self.sum_track_temp, self.weather_races),
            "circuit_rain_rate": safe_div(self.rain_races, self.weather_races),
            "circuit_safety_car_rate": safe_div(self.safety_car_races, self.races),
            "circuit_red_flag_rate": safe_div(self.red_flag_races, self.races),
            "circuit_avg_pit_stop_duration": safe_div(self.pit_sum, self.pit_n),
            "circuit_avg_stint_laps": safe_div(self.stint_sum, self.stint_n),
        }
        features.update(metadata)
        return features

    def update_race(
        self,
        *,
        results: pd.DataFrame,
        openf1_race: dict[str, Any] | None = None,
        openf1_drivers: pd.DataFrame | None = None,
    ) -> None:
        if results.empty:
            return
        entries = len(results)
        self.races += 1
        self.sum_field_size += entries
        dnf_count = int(results["is_dnf"].fillna(False).sum())
        self.sum_dnf += dnf_count
        winners = results[results["classified_position"] == 1]
        if not winners.empty:
            winner = winners.iloc[0]
            self.sum_winning_laps += float(winner.get("laps", 0) or 0)
            self.pole_wins += int(float(winner.get("grid_raw", 0) or 0) == 1)
        for _, row in results.iterrows():
            self.stats.update(
                finish_position=float(row.get("finish_position", math.nan)),
                classified=bool(row.get("classified", False)),
                points=float(row.get("total_points", row.get("points", 0.0)) or 0.0),
                dnf=bool(row.get("is_dnf", False)),
                win=bool(row.get("win", 0)),
                podium=bool(row.get("podium", 0)),
                top10=bool(row.get("top10", 0)),
                quali_position=row.get("quali_position"),
                grid_position=row.get("grid_position"),
                position_gain=row.get("position_gain"),
                laps=row.get("laps"),
            )
        if openf1_race:
            air = openf1_race.get("weather_air_temp_mean")
            track = openf1_race.get("weather_track_temp_mean")
            rain = openf1_race.get("weather_rainfall_mean")
            if air is not None and not (isinstance(air, float) and math.isnan(air)):
                self.sum_air_temp += float(air)
                self.weather_races += 1
            if track is not None and not (isinstance(track, float) and math.isnan(track)):
                self.sum_track_temp += float(track)
            if rain is not None and not (isinstance(rain, float) and math.isnan(rain)) and float(rain) > 0:
                self.rain_races += 1
            if float(openf1_race.get("race_control_safety_car_messages", 0) or 0) > 0:
                self.safety_car_races += 1
            if float(openf1_race.get("race_control_red_flags", 0) or 0) > 0:
                self.red_flag_races += 1
        if openf1_drivers is not None and not openf1_drivers.empty:
            pits = pd.to_numeric(openf1_drivers.get("openf1_mean_pit_seconds"), errors="coerce").dropna()
            stints = pd.to_numeric(openf1_drivers.get("openf1_mean_stint_laps"), errors="coerce").dropna()
            if not pits.empty:
                self.pit_sum += float(pits.mean())
                self.pit_n += 1
            if not stints.empty:
                self.stint_sum += float(stints.mean())
                self.stint_n += 1


def circuit_length_from_openf1(year: int, circuit_key: Any) -> float:
    """Estimate track length from the OpenF1/MultiViewer circuit geometry.

    The public circuit endpoint provides x/y coordinates.  Summing Euclidean
    distances between consecutive points gives a length in metres for the
    circuits where geometry is available.  If the result is implausible the
    value is treated as missing rather than guessed.
    """
    if circuit_key is None:
        return math.nan
    path = config.OPENF1_RAW_DIR / str(year) / "circuit_info" / f"{circuit_key}.json"
    payload = load_json(path, default={}) or {}
    data = unwrap(payload) or {}
    x = data.get("x")
    y = data.get("y")
    if not isinstance(x, list) or not isinstance(y, list) or len(x) < 10 or len(x) != len(y):
        return math.nan
    try:
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        length_km = float(distances.sum() / 1000.0)
        if 1.0 <= length_km <= 25.0:
            return length_km
    except Exception:
        return math.nan
    return math.nan


def build_circuit_metadata(
    races: pd.DataFrame,
    circuits: pd.DataFrame,
    openf1_race: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Static/known-before-race circuit metadata keyed by circuit_id.

    No race outcome is used.  Length/type come from OpenF1/MultiViewer where
    available and remain missing otherwise.
    """
    metadata: dict[str, dict[str, float]] = {}
    if races.empty:
        return metadata
    for circuit_id, group in races.groupby("circuit_id"):
        latest = group.sort_values(["season", "round"]).iloc[-1]
        row: dict[str, float] = {
            "circuit_latitude": float(latest.get("latitude", math.nan)) if pd.notna(latest.get("latitude")) else math.nan,
            "circuit_longitude": float(latest.get("longitude", math.nan)) if pd.notna(latest.get("longitude")) else math.nan,
            "circuit_altitude": float(latest.get("altitude", math.nan)) if pd.notna(latest.get("altitude")) else math.nan,
            "circuit_type_street": math.nan,
            "circuit_length_km": math.nan,
        }
        if not openf1_race.empty:
            of1 = openf1_race[openf1_race["race_id"].isin(group["race_id"])].sort_values(
                ["season", "round"]
            )
            if not of1.empty:
                last = of1.iloc[-1]
                circuit_type = str(last.get("openf1_circuit_type", "") or "").lower()
                if "street" in circuit_type or "temporary" in circuit_type:
                    row["circuit_type_street"] = 1.0
                elif "permanent" in circuit_type:
                    row["circuit_type_street"] = 0.0
                length = circuit_length_from_openf1(
                    int(last.get("season", 0) or 0), last.get("openf1_circuit_key")
                )
                row["circuit_length_km"] = length
        metadata[str(circuit_id)] = row
    return metadata


def race_context_features(race: pd.Series) -> dict[str, float]:
    total_rounds = float(race.get("total_rounds", 0) or 0)
    round_number = float(race.get("round", 0) or 0)
    season_progress = (round_number - 1.0) / total_rounds if total_rounds > 0 else math.nan
    return {
        "season": float(race.get("season", 0) or 0),
        "round": round_number,
        "total_rounds": total_rounds,
        "season_progress": season_progress,
        "races_remaining": max(0.0, total_rounds - round_number + 1.0),
        "is_season_opener": float(round_number == 1),
        "is_season_finale": float(total_rounds > 0 and round_number >= total_rounds),
        "is_sprint_weekend": float(bool(race.get("is_sprint_weekend", False))),
    }
