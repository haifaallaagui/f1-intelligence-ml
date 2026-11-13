"""Leakage-safe driver history and feature construction."""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils import safe_div


def _nanmean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else math.nan


def _slope(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return math.nan
    x = np.arange(len(vals), dtype=float)
    try:
        return float(np.polyfit(x, np.asarray(vals, dtype=float), 1)[0])
    except Exception:
        return math.nan


@dataclass
class RunningStats:
    """Aggregate statistics updated only after a race has finished."""

    n: int = 0
    wins: int = 0
    podiums: int = 0
    top10: int = 0
    points: float = 0.0
    finishes: int = 0
    dnfs: int = 0
    sum_finish: float = 0.0
    sum_classified: float = 0.0
    sum_quali: float = 0.0
    quali_count: int = 0
    sum_grid: float = 0.0
    grid_count: int = 0
    best_finish: float = math.inf
    sum_gain: float = 0.0
    gain_count: int = 0
    sum_laps: float = 0.0
    laps_count: int = 0

    def update(
        self,
        *,
        finish_position: float,
        classified: bool,
        points: float,
        dnf: bool,
        win: bool,
        podium: bool,
        top10: bool,
        quali_position: float | None = None,
        grid_position: float | None = None,
        position_gain: float | None = None,
        laps: float | None = None,
    ) -> None:
        self.n += 1
        self.wins += int(win)
        self.podiums += int(podium)
        self.top10 += int(top10)
        self.points += float(points or 0.0)
        self.dnfs += int(dnf)
        if not (isinstance(finish_position, float) and math.isnan(finish_position)):
            self.sum_finish += float(finish_position)
        if classified:
            self.finishes += 1
            self.sum_classified += float(finish_position)
            self.best_finish = min(self.best_finish, float(finish_position))
        if quali_position is not None and not (isinstance(quali_position, float) and math.isnan(quali_position)):
            self.sum_quali += float(quali_position)
            self.quali_count += 1
        if grid_position is not None and not (isinstance(grid_position, float) and math.isnan(grid_position)):
            self.sum_grid += float(grid_position)
            self.grid_count += 1
        if position_gain is not None and not (isinstance(position_gain, float) and math.isnan(position_gain)):
            self.sum_gain += float(position_gain)
            self.gain_count += 1
        if laps is not None and not (isinstance(laps, float) and math.isnan(laps)):
            self.sum_laps += float(laps)
            self.laps_count += 1

    @property
    def avg_finish(self) -> float:
        return safe_div(self.sum_finish, self.n)

    @property
    def avg_classified(self) -> float:
        return safe_div(self.sum_classified, self.finishes)

    @property
    def avg_quali(self) -> float:
        return safe_div(self.sum_quali, self.quali_count)

    @property
    def avg_grid(self) -> float:
        return safe_div(self.sum_grid, self.grid_count)

    @property
    def dnf_rate(self) -> float:
        return safe_div(self.dnfs, self.n)

    @property
    def win_rate(self) -> float:
        return safe_div(self.wins, self.n)

    @property
    def podium_rate(self) -> float:
        return safe_div(self.podiums, self.n)

    @property
    def top10_rate(self) -> float:
        return safe_div(self.top10, self.n)

    @property
    def points_per_race(self) -> float:
        return safe_div(self.points, self.n)

    @property
    def best(self) -> float:
        return self.best_finish if math.isfinite(self.best_finish) else math.nan

    def as_features(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_starts": float(self.n),
            f"{prefix}_wins": float(self.wins),
            f"{prefix}_podiums": float(self.podiums),
            f"{prefix}_top10s": float(self.top10),
            f"{prefix}_points": float(self.points),
            f"{prefix}_avg_finish": self.avg_finish,
            f"{prefix}_avg_classified_finish": self.avg_classified,
            f"{prefix}_avg_quali": self.avg_quali,
            f"{prefix}_avg_grid": self.avg_grid,
            f"{prefix}_dnf_rate": self.dnf_rate,
            f"{prefix}_win_rate": self.win_rate,
            f"{prefix}_podium_rate": self.podium_rate,
            f"{prefix}_top10_rate": self.top10_rate,
            f"{prefix}_points_per_race": self.points_per_race,
            f"{prefix}_best_finish": self.best,
            f"{prefix}_avg_position_gain": safe_div(self.sum_gain, self.gain_count),
            f"{prefix}_avg_laps": safe_div(self.sum_laps, self.laps_count),
        }


@dataclass
class TeammateRecord:
    races: int = 0
    race_h2h: float = 0.0
    quali_h2h: float = 0.0
    finish_delta_sum: float = 0.0
    finish_delta_n: int = 0
    quali_delta_sum: float = 0.0
    quali_delta_n: int = 0


@dataclass
class DriverHistory:
    driver_id: str
    date_of_birth: Any = None
    career: RunningStats = field(default_factory=RunningStats)
    season: RunningStats = field(default_factory=RunningStats)
    season_number: int | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    circuit: dict[str, RunningStats] = field(default_factory=dict)
    teammates: dict[str, TeammateRecord] = field(default_factory=dict)
    constructor_races: Counter = field(default_factory=Counter)
    sprint_starts: int = 0
    sprint_wins: int = 0
    sprint_points: float = 0.0
    sprint_finish_sum: float = 0.0
    sprint_finish_n: int = 0
    pit_n: int = 0
    pit_sum: float = 0.0
    stint_n: int = 0
    stint_sum: float = 0.0
    lap_std_n: int = 0
    lap_std_sum: float = 0.0
    pos_change_n: int = 0
    pos_change_sum: float = 0.0

    def ensure_season(self, season: int) -> None:
        if self.season_number != int(season):
            self.season_number = int(season)
            self.season = RunningStats()

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------
    def base_features(self, circuit_id: str | None, race_date: Any) -> dict[str, float]:
        features: dict[str, float] = {}
        features.update(self.career.as_features("driver_career"))
        features.update(self.season.as_features("driver_season"))
        features["driver_season_race_number"] = float(self.season.n + 1)
        features["driver_career_race_number"] = float(self.career.n + 1)
        features["driver_is_rookie"] = float(self.career.n == 0)
        features["driver_constructor_races"] = float(max(self.constructor_races.values(), default=0))
        features["driver_age"] = _age_at(race_date, self.date_of_birth)

        # Recent form: last 3, 5 and 10 races (available history, not future).
        records = list(self.recent)
        for window in (3, 5, 10):
            subset = records[-window:]
            prefix = f"driver_last{window}"
            features[f"{prefix}_races"] = float(len(subset))
            features[f"{prefix}_avg_finish"] = _nanmean([r.get("finish") for r in subset])
            features[f"{prefix}_avg_points"] = _nanmean([r.get("points") for r in subset])
            features[f"{prefix}_avg_quali"] = _nanmean([r.get("quali") for r in subset])
            features[f"{prefix}_dnf_rate"] = _nanmean([r.get("dnf") for r in subset])
            features[f"{prefix}_win_rate"] = _nanmean([r.get("win") for r in subset])
            features[f"{prefix}_podium_rate"] = _nanmean([r.get("podium") for r in subset])
            features[f"{prefix}_avg_gain"] = _nanmean([r.get("gain") for r in subset])
        last5 = records[-5:]
        features["driver_finish_trend_last5"] = _slope([r.get("finish") for r in last5])
        features["driver_points_trend_last5"] = _slope([r.get("points") for r in last5])
        features["driver_form_points_weighted_last5"] = _weighted_mean(
            [r.get("points") for r in last5]
        )
        features["driver_form_finish_weighted_last5"] = _weighted_mean(
            [r.get("finish") for r in last5]
        )

        # Circuit-specific history.
        circuit_stats = self.circuit.get(str(circuit_id), RunningStats()) if circuit_id else RunningStats()
        c = circuit_stats.as_features("driver_circuit")
        features.update(c)
        features["driver_circuit_affinity"] = (
            circuit_stats.avg_finish - self.career.avg_finish
            if circuit_stats.n and self.career.n and not math.isnan(circuit_stats.avg_finish) and not math.isnan(self.career.avg_finish)
            else math.nan
        )
        features["driver_circuit_points_per_race"] = circuit_stats.points_per_race
        features["driver_circuit_win_rate"] = circuit_stats.win_rate
        features["driver_circuit_podium_rate"] = circuit_stats.podium_rate

        # Sprint history.
        features["driver_sprint_starts"] = float(self.sprint_starts)
        features["driver_sprint_points"] = float(self.sprint_points)
        features["driver_sprint_avg_finish"] = safe_div(self.sprint_finish_sum, self.sprint_finish_n)
        features["driver_sprint_win_rate"] = safe_div(self.sprint_wins, self.sprint_starts)
        features["driver_sprint_points_per_sprint"] = safe_div(self.sprint_points, self.sprint_starts)

        # Prior OpenF1 pit/stint/consistency summaries.
        features["driver_avg_pit_stop_duration"] = safe_div(self.pit_sum, self.pit_n)
        features["driver_avg_stint_laps"] = safe_div(self.stint_sum, self.stint_n)
        features["driver_avg_lap_time_std"] = safe_div(self.lap_std_sum, self.lap_std_n)
        features["driver_avg_position_changes"] = safe_div(self.pos_change_sum, self.pos_change_n)
        return features

    def teammate_features(
        self,
        teammate_ids: list[str],
        driver_states: dict[str, "DriverHistory"],
        championship_positions: dict[str, float],
    ) -> dict[str, float]:
        features: dict[str, float] = {
            "driver_teammate_count": float(len(teammate_ids)),
            "driver_vs_teammate_races": 0.0,
            "driver_vs_teammate_race_h2h_rate": math.nan,
            "driver_vs_teammate_quali_h2h_rate": math.nan,
            "driver_vs_teammate_avg_finish_delta": math.nan,
            "driver_vs_teammate_avg_quali_delta": math.nan,
            "teammate_season_avg_finish": math.nan,
            "teammate_season_avg_quali": math.nan,
            "teammate_season_points_per_race": math.nan,
            "teammate_career_avg_finish": math.nan,
            "teammate_championship_position": math.nan,
            "driver_teammate_changed": math.nan,
        }
        if not teammate_ids:
            return features
        records = [self.teammates.get(tid, TeammateRecord()) for tid in teammate_ids]
        total_races = sum(r.races for r in records)
        features["driver_vs_teammate_races"] = float(total_races)
        features["driver_teammate_changed"] = float(total_races == 0)
        features["driver_vs_teammate_race_h2h_rate"] = _nanmean(
            [safe_div(r.race_h2h, r.races) for r in records]
        )
        features["driver_vs_teammate_quali_h2h_rate"] = _nanmean(
            [safe_div(r.quali_h2h, r.races) for r in records]
        )
        features["driver_vs_teammate_avg_finish_delta"] = _nanmean(
            [safe_div(r.finish_delta_sum, r.finish_delta_n) for r in records]
        )
        features["driver_vs_teammate_avg_quali_delta"] = _nanmean(
            [safe_div(r.quali_delta_sum, r.quali_delta_n) for r in records]
        )
        states = [driver_states[tid] for tid in teammate_ids if tid in driver_states]
        if states:
            features["teammate_season_avg_finish"] = _nanmean([s.season.avg_finish for s in states])
            features["teammate_season_avg_quali"] = _nanmean([s.season.avg_quali for s in states])
            features["teammate_season_points_per_race"] = _nanmean(
                [s.season.points_per_race for s in states]
            )
            features["teammate_career_avg_finish"] = _nanmean([s.career.avg_finish for s in states])
            features["teammate_championship_position"] = _nanmean(
                [championship_positions.get(tid, math.nan) for tid in teammate_ids]
            )
        return features

    # ------------------------------------------------------------------
    # Updates (always after the race)
    # ------------------------------------------------------------------
    def update_race(
        self,
        *,
        season: int,
        result: dict[str, Any],
        quali_position: float | None,
        grid_position: float | None,
        teammate_results: list[dict[str, Any]],
        circuit_id: str | None,
        sprint: dict[str, Any] | None = None,
        openf1: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_season(season)
        finish = float(result.get("finish_position", math.nan))
        classified = bool(result.get("classified", False))
        points = float(result.get("total_points", result.get("points", 0.0)) or 0.0)
        dnf = bool(result.get("is_dnf", False))
        win = bool(result.get("win", 0))
        podium = bool(result.get("podium", 0))
        top10 = bool(result.get("top10", 0))
        gain = result.get("position_gain")
        laps = result.get("laps")

        for stats in (self.career, self.season):
            stats.update(
                finish_position=finish,
                classified=classified,
                points=points,
                dnf=dnf,
                win=win,
                podium=podium,
                top10=top10,
                quali_position=quali_position,
                grid_position=grid_position,
                position_gain=gain,
                laps=laps,
            )
        if circuit_id:
            self.circuit.setdefault(str(circuit_id), RunningStats()).update(
                finish_position=finish,
                classified=classified,
                points=points,
                dnf=dnf,
                win=win,
                podium=podium,
                top10=top10,
                quali_position=quali_position,
                grid_position=grid_position,
                position_gain=gain,
                laps=laps,
            )
        self.recent.append(
            {
                "finish": finish,
                "points": points,
                "quali": quali_position,
                "dnf": float(dnf),
                "win": float(win),
                "podium": float(podium),
                "gain": gain,
            }
        )
        constructor_id = result.get("constructor_id")
        if constructor_id:
            self.constructor_races[str(constructor_id)] += 1

        # Teammate head-to-head records.
        for teammate in teammate_results:
            tid = str(teammate.get("driver_id"))
            if not tid or tid == str(self.driver_id):
                continue
            rec = self.teammates.setdefault(tid, TeammateRecord())
            rec.races += 1
            other_finish = float(teammate.get("finish_position", math.nan))
            if not math.isnan(finish) and not math.isnan(other_finish):
                rec.race_h2h += 1.0 if finish < other_finish else (0.5 if finish == other_finish else 0.0)
                rec.finish_delta_sum += finish - other_finish
                rec.finish_delta_n += 1
            other_quali = teammate.get("quali_position")
            if (
                quali_position is not None
                and other_quali is not None
                and not (isinstance(quali_position, float) and math.isnan(quali_position))
                and not (isinstance(other_quali, float) and math.isnan(other_quali))
            ):
                rec.quali_h2h += 1.0 if quali_position < other_quali else (0.5 if quali_position == other_quali else 0.0)
                rec.quali_delta_sum += float(quali_position) - float(other_quali)
                rec.quali_delta_n += 1

        if sprint:
            self.sprint_starts += 1
            self.sprint_points += float(sprint.get("sprint_points", 0.0) or 0.0)
            sprint_finish = sprint.get("sprint_position")
            if sprint_finish is not None and not (isinstance(sprint_finish, float) and math.isnan(sprint_finish)):
                self.sprint_finish_sum += float(sprint_finish)
                self.sprint_finish_n += 1
            self.sprint_wins += int(float(sprint_finish or 99) == 1)

        if openf1:
            pit = openf1.get("openf1_mean_pit_seconds")
            if pit is not None and not (isinstance(pit, float) and math.isnan(pit)):
                self.pit_sum += float(pit)
                self.pit_n += 1
            stint = openf1.get("openf1_mean_stint_laps")
            if stint is not None and not (isinstance(stint, float) and math.isnan(stint)):
                self.stint_sum += float(stint)
                self.stint_n += 1
            lap_std = openf1.get("openf1_lap_std_seconds")
            if lap_std is not None and not (isinstance(lap_std, float) and math.isnan(lap_std)):
                self.lap_std_sum += float(lap_std)
                self.lap_std_n += 1
            changes = openf1.get("openf1_position_changes")
            if changes is not None and not (isinstance(changes, float) and math.isnan(changes)):
                self.pos_change_sum += float(changes)
                self.pos_change_n += 1


def _age_at(race_date: Any, date_of_birth: Any) -> float:
    if race_date is None or date_of_birth is None:
        return math.nan
    try:
        race = pd.to_datetime(race_date, errors="coerce", utc=True)
        dob = pd.to_datetime(date_of_birth, errors="coerce", utc=True)
        if pd.isna(race) or pd.isna(dob):
            return math.nan
        return float((race - dob).days / 365.25)
    except Exception:
        return math.nan


def _weighted_mean(values: list[float]) -> float:
    vals = [(i + 1, v) for i, v in enumerate(values) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan
    total_weight = sum(w for w, _ in vals)
    return float(sum(w * float(v) for w, v in vals) / total_weight)
