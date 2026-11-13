"""Leakage-safe constructor history and feature construction."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.features.driver_features import RunningStats, _nanmean, _slope, _weighted_mean
from src.utils import safe_div


@dataclass
class ConstructorHistory:
    constructor_id: str
    career: RunningStats = field(default_factory=RunningStats)
    season: RunningStats = field(default_factory=RunningStats)
    season_number: int | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    circuit: dict[str, RunningStats] = field(default_factory=dict)
    team_entries: int = 0
    team_dnfs: int = 0
    season_team_entries: int = 0
    season_team_dnfs: int = 0
    sprint_starts: int = 0
    sprint_points: float = 0.0
    sprint_finish_sum: float = 0.0
    sprint_finish_n: int = 0
    pit_n: int = 0
    pit_sum: float = 0.0
    stint_n: int = 0
    stint_sum: float = 0.0

    def ensure_season(self, season: int) -> None:
        if self.season_number != int(season):
            self.season_number = int(season)
            self.season = RunningStats()
            self.season_team_entries = 0
            self.season_team_dnfs = 0

    def features(self, circuit_id: str | None) -> dict[str, float]:
        features: dict[str, float] = {}
        features.update(self.career.as_features("constructor_career"))
        features.update(self.season.as_features("constructor_season"))
        features["constructor_season_race_number"] = float(self.season.n + 1)
        features["constructor_career_race_number"] = float(self.career.n + 1)
        features["constructor_reliability"] = (
            1.0 - safe_div(self.team_dnfs, self.team_entries) if self.team_entries else math.nan
        )
        features["constructor_season_reliability"] = (
            1.0 - safe_div(self.season_team_dnfs, self.season_team_entries)
            if self.season_team_entries
            else math.nan
        )
        records = list(self.recent)
        for window in (3, 5, 10):
            subset = records[-window:]
            prefix = f"constructor_last{window}"
            features[f"{prefix}_races"] = float(len(subset))
            features[f"{prefix}_avg_points"] = _nanmean([r.get("points") for r in subset])
            features[f"{prefix}_avg_finish"] = _nanmean([r.get("finish") for r in subset])
            features[f"{prefix}_avg_quali"] = _nanmean([r.get("quali") for r in subset])
            features[f"{prefix}_dnf_rate"] = _nanmean([r.get("dnf") for r in subset])
            features[f"{prefix}_win_rate"] = _nanmean([r.get("win") for r in subset])
            features[f"{prefix}_podium_rate"] = _nanmean([r.get("podium") for r in subset])
        last5 = records[-5:]
        features["constructor_points_trend_last5"] = _slope([r.get("points") for r in last5])
        features["constructor_finish_trend_last5"] = _slope([r.get("finish") for r in last5])
        features["constructor_form_points_weighted_last5"] = _weighted_mean(
            [r.get("points") for r in last5]
        )

        circuit_stats = self.circuit.get(str(circuit_id), RunningStats()) if circuit_id else RunningStats()
        features.update(circuit_stats.as_features("constructor_circuit"))
        features["constructor_circuit_points_per_race"] = circuit_stats.points_per_race
        features["constructor_circuit_win_rate"] = circuit_stats.win_rate
        features["constructor_circuit_podium_rate"] = circuit_stats.podium_rate

        features["constructor_sprint_starts"] = float(self.sprint_starts)
        features["constructor_sprint_points"] = float(self.sprint_points)
        features["constructor_sprint_avg_finish"] = safe_div(self.sprint_finish_sum, self.sprint_finish_n)
        features["constructor_sprint_points_per_sprint"] = safe_div(
            self.sprint_points, self.sprint_starts
        )
        features["constructor_avg_pit_stop_duration"] = safe_div(self.pit_sum, self.pit_n)
        features["constructor_avg_stint_laps"] = safe_div(self.stint_sum, self.stint_n)
        return features

    def update_race(
        self,
        *,
        season: int,
        team_result: dict[str, Any],
        circuit_id: str | None,
        sprint_points: float = 0.0,
        sprint_positions: list[float] | None = None,
        openf1: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ensure_season(season)
        entries = int(team_result.get("team_entries", 0) or 0)
        dnfs = int(team_result.get("team_dnf_count", 0) or 0)
        self.team_entries += entries
        self.team_dnfs += dnfs
        self.season_team_entries += entries
        self.season_team_dnfs += dnfs
        for stats in (self.career, self.season):
            stats.update(
                finish_position=team_result.get("team_avg_finish", math.nan),
                classified=bool(team_result.get("team_classified", False)),
                points=float(team_result.get("team_total_points", 0.0) or 0.0),
                dnf=bool(dnfs > 0),
                win=bool(team_result.get("team_win", 0)),
                podium=bool(team_result.get("team_podium_count", 0)),
                top10=bool(team_result.get("team_top10_count", 0)),
                quali_position=team_result.get("team_avg_quali", math.nan),
                grid_position=team_result.get("team_avg_grid", math.nan),
                position_gain=team_result.get("team_avg_gain", math.nan),
                laps=team_result.get("team_laps", math.nan),
            )
        if circuit_id:
            self.circuit.setdefault(str(circuit_id), RunningStats()).update(
                finish_position=team_result.get("team_avg_finish", math.nan),
                classified=bool(team_result.get("team_classified", False)),
                points=float(team_result.get("team_total_points", 0.0) or 0.0),
                dnf=bool(dnfs > 0),
                win=bool(team_result.get("team_win", 0)),
                podium=bool(team_result.get("team_podium_count", 0)),
                top10=bool(team_result.get("team_top10_count", 0)),
                quali_position=team_result.get("team_avg_quali", math.nan),
                grid_position=team_result.get("team_avg_grid", math.nan),
                position_gain=team_result.get("team_avg_gain", math.nan),
                laps=team_result.get("team_laps", math.nan),
            )
        self.recent.append(
            {
                "points": float(team_result.get("team_total_points", 0.0) or 0.0),
                "finish": team_result.get("team_avg_finish", math.nan),
                "quali": team_result.get("team_avg_quali", math.nan),
                "dnf": float(dnfs > 0),
                "win": float(team_result.get("team_win", 0)),
                "podium": float(bool(team_result.get("team_podium_count", 0))),
            }
        )
        if sprint_points or sprint_positions:
            self.sprint_starts += 1
            self.sprint_points += float(sprint_points or 0.0)
            positions = [p for p in (sprint_positions or []) if p is not None and not (isinstance(p, float) and math.isnan(p))]
            if positions:
                self.sprint_finish_sum += float(np.mean(positions))
                self.sprint_finish_n += 1
        for row in openf1 or []:
            pit = row.get("openf1_mean_pit_seconds")
            if pit is not None and not (isinstance(pit, float) and math.isnan(pit)):
                self.pit_sum += float(pit)
                self.pit_n += 1
            stint = row.get("openf1_mean_stint_laps")
            if stint is not None and not (isinstance(stint, float) and math.isnan(stint)):
                self.stint_sum += float(stint)
                self.stint_n += 1
