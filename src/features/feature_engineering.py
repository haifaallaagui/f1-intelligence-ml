"""End-to-end leakage-safe feature engineering.

The central invariant is simple: features for race R are computed from state
that has been updated only with races strictly before R.  The current race's
result is used *after* all feature rows for R have been created, never before.
Qualifying/grid features are produced in separate columns and are only used by
the `after_qualifying` models.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import config
from src.features.constructor_features import ConstructorHistory
from src.features.driver_features import DriverHistory
from src.features.race_features import (
    CircuitHistory,
    build_circuit_metadata,
    race_context_features,
)
from src.preprocessing.cleaning import load_processed_tables
from src.scoring import race_points_system
from src.utils import (
    dataframe_hash,
    ensure_dirs,
    get_logger,
    make_race_id,
    parse_race_id,
    safe_div,
    save_json,
    save_parquet,
    utc_now_iso,
)

LOG = get_logger("features.engineering")

TARGET_COLUMNS = [
    "finish_position",
    "classified_position",
    "race_points",
    "sprint_points",
    "total_points",
    "win",
    "podium",
    "top10",
    "dnf",
    "position_gain",
    "is_p1",
    "is_p2",
    "is_p3",
]

CONSTRUCTOR_TARGET_COLUMNS = [
    "team_total_points",
    "team_race_points",
    "team_sprint_points",
    "team_best_finish",
    "team_avg_finish",
    "team_best_classified",
    "team_avg_classified",
    "team_dnf_count",
    "team_entries",
    "team_win",
    "team_podium_count",
    "team_top10_count",
]

ID_COLUMNS = [
    "race_id",
    "season",
    "round",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "race_name",
    "driver_name",
    "constructor_name",
    "circuit_name",
    "race_date",
]

POST_QUALI_FEATURES = [
    "quali_position",
    "quali_position_norm",
    "quali_gap_to_pole_seconds",
    "quali_gap_to_pole_pct",
    "quali_gap_to_teammate_seconds",
    "quali_gap_to_teammate_pct",
    "quali_reached_q2",
    "quali_reached_q3",
    "grid_position",
    "grid_position_norm",
    "grid_penalty",
    "grid_gain_vs_quali",
    "grid_position_estimated",
    "sprint_position_current",
    "sprint_points_current",
    "sprint_gain_current",
    "sprint_grid_current",
    "fastf1_best_lap_seconds",
    "fastf1_gap_to_pole_seconds",
    "fastf1_gap_to_pole_pct",
    "fastf1_sector1_seconds",
    "fastf1_sector2_seconds",
    "fastf1_sector3_seconds",
    "fastf1_compound_code",
]

BASE_CONTEXT_FEATURES = [
    "season",
    "round",
    "total_rounds",
    "season_progress",
    "races_remaining",
    "is_season_opener",
    "is_season_finale",
    "is_sprint_weekend",
    "field_size",
    "season_races_completed",
]

DRIVER_AGG_FEATURES = [
    "driver_career_starts",
    "driver_career_points",
    "driver_career_avg_finish",
    "driver_career_avg_quali",
    "driver_career_dnf_rate",
    "driver_season_points",
    "driver_season_avg_finish",
    "driver_season_avg_quali",
    "driver_season_points_per_race",
    "driver_last5_avg_points",
    "driver_last5_avg_finish",
    "driver_last5_avg_quali",
    "driver_last5_dnf_rate",
    "driver_form_points_weighted_last5",
    "driver_circuit_avg_finish",
    "driver_circuit_avg_quali",
    "driver_championship_position",
    "driver_championship_points",
    "driver_age",
    "driver_vs_teammate_avg_finish_delta",
    "driver_vs_teammate_avg_quali_delta",
    "driver_avg_pit_stop_duration",
    "driver_avg_stint_laps",
]

_COMPOUND_CODES = {
    "soft": 0.0,
    "medium": 1.0,
    "hard": 2.0,
    "intermediate": 3.0,
    "inter": 3.0,
    "wet": 4.0,
}


def _is_base_feature(name: str) -> bool:
    if name in ID_COLUMNS or name in TARGET_COLUMNS:
        return False
    if name in BASE_CONTEXT_FEATURES:
        return True
    if name.startswith(("driver_", "constructor_", "teammate_", "circuit_")):
        # Identifier/name columns are not model inputs.
        if name in {
            "driver_id",
            "driver_name",
            "driver_dob",
            "driver_code",
            "driver_nationality",
            "driver_url",
            "constructor_id",
            "constructor_name",
            "constructor_nationality",
            "constructor_url",
            "circuit_id",
            "circuit_name",
            "circuit_url",
        }:
            return False
        return True
    return False


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return value


class FeatureEngineer:
    def __init__(self, tables: dict[str, pd.DataFrame] | None = None):
        self.tables = tables or load_processed_tables()
        self.races = self.tables.get("races", pd.DataFrame()).copy()
        self.results = self.tables.get("results", pd.DataFrame()).copy()
        self.qualifying = self.tables.get("qualifying", pd.DataFrame()).copy()
        self.sprints = self.tables.get("sprints", pd.DataFrame()).copy()
        self.drivers = self.tables.get("drivers", pd.DataFrame()).copy()
        self.constructors = self.tables.get("constructors", pd.DataFrame()).copy()
        self.circuits = self.tables.get("circuits", pd.DataFrame()).copy()
        self.openf1_race = self.tables.get("openf1_race_summary", pd.DataFrame()).copy()
        self.openf1_driver = self.tables.get("openf1_driver_race_summary", pd.DataFrame()).copy()
        self.fastf1_quali = self.tables.get("fastf1_qualifying_summary", pd.DataFrame()).copy()
        self._prepare()
        self._reset_state()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        if self.races.empty or self.results.empty:
            raise RuntimeError("Processed races/results are empty; build processed tables first.")
        self.races["race_date"] = pd.to_datetime(self.races["race_date"], errors="coerce")
        self.races = self.races.sort_values(["race_date", "season", "round"]).reset_index(drop=True)
        self.races["_order"] = range(len(self.races))
        self.race_by_id = {str(r["race_id"]): r for _, r in self.races.iterrows()}
        self.race_order = {str(r["race_id"]): int(r["_order"]) for _, r in self.races.iterrows()}

        self.results_by_race = {
            str(race_id): group.copy()
            for race_id, group in self.results.groupby("race_id", sort=False)
        }
        self.quali_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.qualifying.empty:
            for row in self.qualifying.to_dict("records"):
                self.quali_lookup[(str(row["race_id"]), str(row["driver_id"]))] = row
        self.sprint_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.sprints.empty:
            for row in self.sprints.to_dict("records"):
                self.sprint_lookup[(str(row["race_id"]), str(row["driver_id"]))] = row
        self.fastf1_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.fastf1_quali.empty:
            for row in self.fastf1_quali.to_dict("records"):
                if row.get("driver_id") is not None and not pd.isna(row.get("driver_id")):
                    self.fastf1_lookup[(str(row["race_id"]), str(row["driver_id"]))] = row
        self.openf1_race_by_id = {
            str(row["race_id"]): row for row in self.openf1_race.to_dict("records")
        } if not self.openf1_race.empty else {}
        self.openf1_driver_by_race: dict[str, pd.DataFrame] = {}
        if not self.openf1_driver.empty:
            for race_id, group in self.openf1_driver.groupby("race_id", sort=False):
                self.openf1_driver_by_race[str(race_id)] = group.copy()

        self.driver_dob = {}
        if not self.drivers.empty:
            for row in self.drivers.to_dict("records"):
                self.driver_dob[str(row["driver_id"])] = row.get("date_of_birth")
        for row in self.results.to_dict("records"):
            if row.get("driver_dob"):
                self.driver_dob.setdefault(str(row["driver_id"]), row.get("driver_dob"))

        self.circuit_metadata = build_circuit_metadata(
            self.races, self.circuits, self.openf1_race
        )
        self.circuit_names = {
            str(row["circuit_id"]): row.get("circuit_name")
            for row in self.races.to_dict("records")
            if row.get("circuit_id") is not None
        }

    def _reset_state(self) -> None:
        self.driver_states: dict[str, DriverHistory] = {}
        self.constructor_states: dict[str, ConstructorHistory] = {}
        self.circuit_histories: dict[str, CircuitHistory] = {}
        self.driver_season_points: dict[int, dict[str, float]] = {}
        self.constructor_season_points: dict[int, dict[str, float]] = {}
        self.driver_season_wins: dict[int, dict[str, float]] = {}
        self.constructor_season_wins: dict[int, dict[str, float]] = {}
        self.driver_season_best: dict[int, dict[str, float]] = {}
        self.constructor_season_best: dict[int, dict[str, float]] = {}
        self.season_races_completed: dict[int, int] = {}

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _driver_state(self, driver_id: str, dob: Any = None) -> DriverHistory:
        driver_id = str(driver_id)
        if driver_id not in self.driver_states:
            self.driver_states[driver_id] = DriverHistory(
                driver_id=driver_id, date_of_birth=dob or self.driver_dob.get(driver_id)
            )
        return self.driver_states[driver_id]

    def _constructor_state(self, constructor_id: str) -> ConstructorHistory:
        constructor_id = str(constructor_id)
        if constructor_id not in self.constructor_states:
            self.constructor_states[constructor_id] = ConstructorHistory(constructor_id=constructor_id)
        return self.constructor_states[constructor_id]

    def _circuit_history(self, circuit_id: str) -> CircuitHistory:
        circuit_id = str(circuit_id)
        if circuit_id not in self.circuit_histories:
            self.circuit_histories[circuit_id] = CircuitHistory(circuit_id=circuit_id)
        return self.circuit_histories[circuit_id]

    # ------------------------------------------------------------------
    # Championship standings (computed from prior races only)
    # ------------------------------------------------------------------
    def _standings_features(
        self,
        race: dict[str, Any],
        driver_ids: Sequence[str],
        constructor_ids: Sequence[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        season = int(race.get("season", 0))
        completed = int(self.season_races_completed.get(season, 0))
        dp = self.driver_season_points.setdefault(season, {})
        dw = self.driver_season_wins.setdefault(season, {})
        db = self.driver_season_best.setdefault(season, {})
        cp = self.constructor_season_points.setdefault(season, {})
        cw = self.constructor_season_wins.setdefault(season, {})
        cb = self.constructor_season_best.setdefault(season, {})
        max_points = max(race_points_system(season)) if race_points_system(season) else 25

        driver_participants = set(dp) | {str(d) for d in driver_ids}
        driver_order = sorted(
            driver_participants,
            key=lambda d: (-dp.get(d, 0.0), -dw.get(d, 0.0), db.get(d, 999.0), d),
        )
        driver_pos = {d: i + 1 for i, d in enumerate(driver_order)}
        driver_leader_points = max((dp.get(d, 0.0) for d in driver_order), default=0.0)

        constructor_participants = set(cp) | {str(c) for c in constructor_ids}
        constructor_order = sorted(
            constructor_participants,
            key=lambda c: (-cp.get(c, 0.0), -cw.get(c, 0.0), cb.get(c, 999.0), c),
        )
        constructor_pos = {c: i + 1 for i, c in enumerate(constructor_order)}
        constructor_leader_points = max((cp.get(c, 0.0) for c in constructor_order), default=0.0)

        prev_dp = self.driver_season_points.get(season - 1, {})
        prev_dw = self.driver_season_wins.get(season - 1, {})
        prev_db = self.driver_season_best.get(season - 1, {})
        prev_cp = self.constructor_season_points.get(season - 1, {})
        prev_cw = self.constructor_season_wins.get(season - 1, {})
        prev_cb = self.constructor_season_best.get(season - 1, {})
        prev_driver_pos: dict[str, int] = {}
        if prev_dp:
            for i, d in enumerate(
                sorted(prev_dp, key=lambda x: (-prev_dp.get(x, 0), -prev_dw.get(x, 0), prev_db.get(x, 999), x))
            ):
                prev_driver_pos[d] = i + 1
        prev_constructor_pos: dict[str, int] = {}
        if prev_cp:
            for i, c in enumerate(
                sorted(prev_cp, key=lambda x: (-prev_cp.get(x, 0), -prev_cw.get(x, 0), prev_cb.get(x, 999), x))
            ):
                prev_constructor_pos[c] = i + 1

        driver_features: dict[str, dict[str, float]] = {}
        for driver_id in driver_ids:
            driver_id = str(driver_id)
            position = float(driver_pos[driver_id]) if completed > 0 else math.nan
            points = float(dp.get(driver_id, 0.0))
            if completed > 0 and position > 1:
                ahead = driver_order[int(position) - 2]
                ahead_points = float(dp.get(ahead, 0.0))
            else:
                ahead_points = points
            driver_features[driver_id] = {
                "driver_championship_position": position,
                "driver_championship_points": points,
                "driver_gap_to_leader": (driver_leader_points - points) if completed > 0 else math.nan,
                "driver_gap_to_next": (ahead_points - points) if completed > 0 else math.nan,
                "driver_championship_position_norm": (
                    position / max(1.0, float(len(driver_order))) if completed > 0 else math.nan
                ),
                "driver_points_share_season": (
                    points / (completed * max_points) if completed > 0 else math.nan
                ),
                "driver_prev_season_points": float(prev_dp.get(driver_id, math.nan)) if prev_dp else math.nan,
                "driver_prev_season_position": float(prev_driver_pos[driver_id]) if driver_id in prev_driver_pos else math.nan,
                "season_races_completed": float(completed),
            }

        constructor_features: dict[str, dict[str, float]] = {}
        for constructor_id in constructor_ids:
            constructor_id = str(constructor_id)
            position = float(constructor_pos[constructor_id]) if completed > 0 else math.nan
            points = float(cp.get(constructor_id, 0.0))
            if completed > 0 and position > 1:
                ahead = constructor_order[int(position) - 2]
                ahead_points = float(cp.get(ahead, 0.0))
            else:
                ahead_points = points
            constructor_features[constructor_id] = {
                "constructor_championship_position": position,
                "constructor_championship_points": points,
                "constructor_gap_to_leader": (
                    constructor_leader_points - points if completed > 0 else math.nan
                ),
                "constructor_gap_to_next": (ahead_points - points) if completed > 0 else math.nan,
                "constructor_championship_position_norm": (
                    position / max(1.0, float(len(constructor_order))) if completed > 0 else math.nan
                ),
                "constructor_points_share_season": (
                    points / (completed * max_points * 2.0) if completed > 0 else math.nan
                ),
                "constructor_prev_season_points": float(prev_cp.get(constructor_id, math.nan)) if prev_cp else math.nan,
                "constructor_prev_season_position": float(prev_constructor_pos[constructor_id]) if constructor_id in prev_constructor_pos else math.nan,
            }
        return {"driver": driver_features, "constructor": constructor_features}

    # ------------------------------------------------------------------
    # Post-qualifying features
    # ------------------------------------------------------------------
    def _post_quali_features(
        self,
        race_id: str,
        driver_id: str,
        *,
        field_size: int,
        mode: str,
        actual_grid: float | None = None,
    ) -> dict[str, float]:
        features = {name: math.nan for name in POST_QUALI_FEATURES}
        if mode == "before_qualifying":
            return features
        q = self.quali_lookup.get((str(race_id), str(driver_id)))
        if q:
            quali_position = q.get("quali_position")
            features["quali_position"] = float(quali_position) if quali_position is not None and not pd.isna(quali_position) else math.nan
            features["quali_position_norm"] = (
                features["quali_position"] / field_size if field_size and not math.isnan(features["quali_position"]) else math.nan
            )
            features["quali_gap_to_pole_seconds"] = float(q.get("quali_gap_to_pole_seconds", math.nan))
            features["quali_gap_to_pole_pct"] = float(q.get("quali_gap_to_pole_pct", math.nan))
            features["quali_gap_to_teammate_seconds"] = float(q.get("quali_gap_to_teammate_seconds", math.nan))
            features["quali_gap_to_teammate_pct"] = float(q.get("quali_gap_to_teammate_pct", math.nan))
            features["quali_reached_q2"] = float(q.get("quali_reached_q2", math.nan))
            features["quali_reached_q3"] = float(q.get("quali_reached_q3", math.nan))
        sprint = self.sprint_lookup.get((str(race_id), str(driver_id)))
        if sprint:
            features["sprint_position_current"] = float(sprint.get("sprint_position", math.nan))
            features["sprint_points_current"] = float(sprint.get("sprint_points", 0.0) or 0.0)
            features["sprint_gain_current"] = float(sprint.get("sprint_position_gain", math.nan))
            features["sprint_grid_current"] = float(sprint.get("sprint_grid", math.nan))
        fast = self.fastf1_lookup.get((str(race_id), str(driver_id)))
        if fast:
            features["fastf1_best_lap_seconds"] = float(fast.get("fastf1_best_lap_seconds", math.nan))
            features["fastf1_gap_to_pole_seconds"] = float(fast.get("fastf1_gap_to_pole_seconds", math.nan))
            features["fastf1_gap_to_pole_pct"] = float(fast.get("fastf1_gap_to_pole_pct", math.nan))
            features["fastf1_sector1_seconds"] = float(fast.get("fastf1_sector1_seconds", math.nan))
            features["fastf1_sector2_seconds"] = float(fast.get("fastf1_sector2_seconds", math.nan))
            features["fastf1_sector3_seconds"] = float(fast.get("fastf1_sector3_seconds", math.nan))
            compound = str(fast.get("fastf1_compound", "") or "").strip().lower()
            features["fastf1_compound_code"] = _COMPOUND_CODES.get(compound, math.nan)

        # Grid: actual post-qualifying grid if the race has happened; otherwise
        # use the qualifying order as a clearly flagged proxy.
        if actual_grid is not None and not (isinstance(actual_grid, float) and math.isnan(actual_grid)):
            grid = float(actual_grid)
            features["grid_position"] = grid
            features["grid_position_estimated"] = 0.0
        elif not math.isnan(features["quali_position"]):
            grid = features["quali_position"]
            features["grid_position"] = grid
            features["grid_position_estimated"] = 1.0
        else:
            grid = math.nan
        features["grid_position_norm"] = (
            grid / field_size if field_size and not math.isnan(grid) else math.nan
        )
        features["grid_penalty"] = (
            grid - features["quali_position"]
            if not math.isnan(grid) and not math.isnan(features["quali_position"])
            else math.nan
        )
        features["grid_gain_vs_quali"] = (
            features["quali_position"] - grid
            if not math.isnan(grid) and not math.isnan(features["quali_position"])
            else math.nan
        )
        return features

    # ------------------------------------------------------------------
    # Feature rows
    # ------------------------------------------------------------------
    def _features_for_race(
        self,
        race: pd.Series | dict[str, Any],
        results: pd.DataFrame,
        *,
        mode: str = "both",
        use_target: bool = True,
        actual_results: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        race_dict = race.to_dict() if isinstance(race, pd.Series) else dict(race)
        race_id = str(race_dict.get("race_id"))
        field_size = int(len(results))
        circuit_id = str(race_dict.get("circuit_id"))
        circuit_features = self._circuit_history(circuit_id).features(
            self.circuit_metadata.get(circuit_id, {})
        )
        context = race_context_features(pd.Series(race_dict))
        context["field_size"] = float(field_size)
        records = results.to_dict("records")
        season = int(race_dict.get("season", 0))
        # Season-scoped running stats must be reset before any feature is read.
        # Otherwise the first race of a season would leak the previous season's
        # totals into ``driver_season_*`` / ``constructor_season_*``.
        for row in records:
            self._driver_state(
                str(row.get("driver_id")), row.get("driver_dob")
            ).ensure_season(season)
            self._constructor_state(str(row.get("constructor_id"))).ensure_season(season)
        driver_ids = [str(r.get("driver_id")) for r in records]
        constructor_ids: list[str] = []
        teammates_by_constructor: dict[str, list[str]] = {}
        for row in records:
            cid = str(row.get("constructor_id"))
            if cid not in teammates_by_constructor:
                constructor_ids.append(cid)
                teammates_by_constructor[cid] = []
            teammates_by_constructor[cid].append(str(row.get("driver_id")))
        standings = self._standings_features(race_dict, driver_ids, constructor_ids)
        # During dataset construction the `results` frame itself contains the
        # actual post-qualifying grid.  During prediction the caller may pass
        # the target results separately.  Both are allowed only in Mode 2.
        actual_grid_map: dict[str, float] = {}
        for row in records:
            grid = row.get("grid_position")
            if grid is not None and not pd.isna(grid):
                actual_grid_map[str(row.get("driver_id"))] = float(grid)
        if actual_results is not None and not actual_results.empty:
            for row in actual_results.to_dict("records"):
                grid = row.get("grid_position")
                if grid is not None and not pd.isna(grid):
                    actual_grid_map[str(row.get("driver_id"))] = float(grid)

        rows: list[dict[str, Any]] = []
        for result in records:
            driver_id = str(result.get("driver_id"))
            constructor_id = str(result.get("constructor_id"))
            driver = self._driver_state(driver_id, result.get("driver_dob"))
            constructor = self._constructor_state(constructor_id)
            teammate_ids = [d for d in teammates_by_constructor.get(constructor_id, []) if d != driver_id]
            features: dict[str, Any] = {}
            features.update(context)
            features.update(circuit_features)
            features.update(driver.base_features(circuit_id, race_dict.get("race_date")))
            features.update(
                driver.teammate_features(
                    teammate_ids,
                    self.driver_states,
                    {
                        did: vals.get("driver_championship_position", math.nan)
                        for did, vals in standings["driver"].items()
                    },
                )
            )
            features.update(constructor.features(circuit_id))
            features.update(standings["driver"].get(driver_id, {}))
            features.update(standings["constructor"].get(constructor_id, {}))
            features["field_size"] = float(field_size)

            if mode == "before_qualifying":
                post = {name: math.nan for name in POST_QUALI_FEATURES}
            else:
                post = self._post_quali_features(
                    race_id,
                    driver_id,
                    field_size=field_size,
                    mode=mode,
                    actual_grid=actual_grid_map.get(driver_id),
                )
            features.update(post)

            features.update(
                {
                    "race_id": race_id,
                    "season": int(race_dict.get("season", 0)),
                    "round": int(race_dict.get("round", 0)),
                    "driver_id": driver_id,
                    "constructor_id": constructor_id,
                    "circuit_id": circuit_id,
                    "race_name": race_dict.get("race_name"),
                    "driver_name": result.get("driver_name"),
                    "constructor_name": result.get("constructor_name"),
                    "circuit_name": race_dict.get("circuit_name") or self.circuit_names.get(circuit_id),
                    "race_date": race_dict.get("race_date"),
                }
            )
            if use_target:
                finish = float(result.get("finish_position", math.nan))
                classified = result.get("classified_position")
                classified = float(classified) if classified is not None and not pd.isna(classified) else math.nan
                features.update(
                    {
                        "finish_position": finish,
                        "classified_position": classified,
                        "race_points": float(result.get("race_points", result.get("points", 0.0)) or 0.0),
                        "sprint_points": float(result.get("sprint_points", 0.0) or 0.0),
                        "total_points": float(
                            result.get("total_points", result.get("points", 0.0)) or 0.0
                        ),
                        "win": int(result.get("win", 0) or 0),
                        "podium": int(result.get("podium", 0) or 0),
                        "top10": int(result.get("top10", 0) or 0),
                        "dnf": int(result.get("is_dnf", False)),
                        "position_gain": float(result.get("position_gain", math.nan)),
                        "is_p1": int(classified == 1),
                        "is_p2": int(classified == 2),
                        "is_p3": int(classified == 3),
                    }
                )
            rows.append(features)
        return rows

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def _update_state(self, race: pd.Series | dict[str, Any], results: pd.DataFrame) -> None:
        race_dict = race.to_dict() if isinstance(race, pd.Series) else dict(race)
        race_id = str(race_dict.get("race_id"))
        season = int(race_dict.get("season", 0))
        circuit_id = str(race_dict.get("circuit_id"))
        openf1_race = self.openf1_race_by_id.get(race_id)
        openf1_drivers = self.openf1_driver_by_race.get(race_id)
        openf1_by_driver = {}
        if openf1_drivers is not None and not openf1_drivers.empty:
            openf1_by_driver = {
                str(row["driver_id"]): row
                for row in openf1_drivers.to_dict("records")
                if row.get("driver_id") is not None and not pd.isna(row.get("driver_id"))
            }

        enriched = results.copy()
        enriched["quali_position"] = [
            self.quali_lookup.get((race_id, str(d)), {}).get("quali_position", math.nan)
            for d in enriched["driver_id"]
        ]
        enriched["quali_position"] = pd.to_numeric(enriched["quali_position"], errors="coerce")

        records = enriched.to_dict("records")
        records_by_constructor: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            records_by_constructor.setdefault(str(r.get("constructor_id")), []).append(r)
        for result in records:
            driver_id = str(result.get("driver_id"))
            constructor_id = str(result.get("constructor_id"))
            teammate_results = [
                r for r in records_by_constructor.get(constructor_id, [])
                if str(r.get("driver_id")) != driver_id
            ]
            self._driver_state(driver_id, result.get("driver_dob")).update_race(
                season=season,
                result=result,
                quali_position=result.get("quali_position"),
                grid_position=result.get("grid_position"),
                teammate_results=teammate_results,
                circuit_id=circuit_id,
                sprint=self.sprint_lookup.get((race_id, driver_id)),
                openf1=openf1_by_driver.get(driver_id),
            )

        # Constructor-level state: one update per constructor per race.
        for constructor_id, group in enriched.groupby("constructor_id", sort=False):
            constructor_id = str(constructor_id)
            records_group = group.to_dict("records")
            classified_positions = pd.to_numeric(
                group["classified_position"], errors="coerce"
            ).dropna()
            quali_positions = pd.to_numeric(group["quali_position"], errors="coerce").dropna()
            grid_positions = pd.to_numeric(group["grid_position"], errors="coerce").dropna()
            gains = pd.to_numeric(group["position_gain"], errors="coerce").dropna()
            sprint_rows = [
                self.sprint_lookup.get((race_id, str(r["driver_id"]))) for r in records_group
            ]
            sprint_rows = [s for s in sprint_rows if s]
            team_result = {
                "team_entries": len(group),
                "team_dnf_count": int(group["is_dnf"].fillna(False).sum()),
                "team_classified": bool(group["classified"].fillna(False).any()),
                "team_total_points": float(group["total_points"].fillna(0.0).sum()),
                "team_race_points": float(group["race_points"].fillna(0.0).sum()),
                "team_sprint_points": float(group["sprint_points"].fillna(0.0).sum()),
                "team_avg_finish": float(pd.to_numeric(group["finish_position"], errors="coerce").mean()),
                "team_best_finish": float(pd.to_numeric(group["finish_position"], errors="coerce").min()),
                "team_avg_classified": float(classified_positions.mean()) if len(classified_positions) else math.nan,
                "team_best_classified": float(classified_positions.min()) if len(classified_positions) else math.nan,
                "team_win": int(group["win"].fillna(0).max()),
                "team_podium_count": int(group["podium"].fillna(0).sum()),
                "team_top10_count": int(group["top10"].fillna(0).sum()),
                "team_avg_quali": float(quali_positions.mean()) if len(quali_positions) else math.nan,
                "team_best_quali": float(quali_positions.min()) if len(quali_positions) else math.nan,
                "team_avg_grid": float(grid_positions.mean()) if len(grid_positions) else math.nan,
                "team_avg_gain": float(gains.mean()) if len(gains) else math.nan,
                "team_laps": float(pd.to_numeric(group["laps"], errors="coerce").sum()),
            }
            self._constructor_state(constructor_id).update_race(
                season=season,
                team_result=team_result,
                circuit_id=circuit_id,
                sprint_points=float(group["sprint_points"].fillna(0.0).sum()),
                sprint_positions=[s.get("sprint_position") for s in sprint_rows],
                openf1=[
                    openf1_by_driver.get(str(r["driver_id"]))
                    for r in records_group
                    if openf1_by_driver.get(str(r["driver_id"])) is not None
                ],
            )

        # Circuit state: update after the race with actual results/weather.
        self._circuit_history(circuit_id).update_race(
            results=enriched,
            openf1_race=openf1_race,
            openf1_drivers=openf1_drivers,
        )

        # Championship points are updated from actual, already-completed results.
        self.season_races_completed[season] = self.season_races_completed.get(season, 0) + 1
        dp = self.driver_season_points.setdefault(season, {})
        dw = self.driver_season_wins.setdefault(season, {})
        db = self.driver_season_best.setdefault(season, {})
        cp = self.constructor_season_points.setdefault(season, {})
        cw = self.constructor_season_wins.setdefault(season, {})
        cb = self.constructor_season_best.setdefault(season, {})
        for result in records:
            driver_id = str(result.get("driver_id"))
            constructor_id = str(result.get("constructor_id"))
            points = float(result.get("total_points", result.get("points", 0.0)) or 0.0)
            dp[driver_id] = dp.get(driver_id, 0.0) + points
            dw[driver_id] = dw.get(driver_id, 0.0) + float(result.get("win", 0) or 0)
            classified = result.get("classified_position")
            if classified is not None and not pd.isna(classified):
                db[driver_id] = min(db.get(driver_id, 999.0), float(classified))
            cp[constructor_id] = cp.get(constructor_id, 0.0) + points
            cw[constructor_id] = cw.get(constructor_id, 0.0) + float(result.get("win", 0) or 0)
            if classified is not None and not pd.isna(classified):
                cb[constructor_id] = min(cb.get(constructor_id, 999.0), float(classified))

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ensure_dirs(config.DATASETS_DIR, config.METADATA_DIR)
        self._reset_state()
        rows: list[dict[str, Any]] = []
        races_used = 0
        for _, race in self.races.sort_values("_order").iterrows():
            race_id = str(race["race_id"])
            results = self.results_by_race.get(race_id)
            if results is None or results.empty:
                continue
            # A race can be in the schedule before its results are available;
            # it must never update state until results exist.
            rows.extend(self._features_for_race(race, results, mode="both", use_target=True))
            self._update_state(race, results)
            races_used += 1
        if not rows:
            raise RuntimeError("No feature rows were produced; check processed results.")
        df = pd.DataFrame(rows)
        df = self._finalise_dataset(df)
        constructor_df = self._aggregate_constructor(df)
        save_parquet(df, config.RACE_FEATURES_FILE)
        save_parquet(constructor_df, config.CONSTRUCTOR_FEATURES_FILE)
        self._write_metadata(df, constructor_df, races_used)
        LOG.info(
            "Built %s driver-race rows and %s constructor-race rows",
            len(df),
            len(constructor_df),
        )
        return df, constructor_df

    def _finalise_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df = df.sort_values(["race_date", "season", "round", "finish_position", "driver_id"]).reset_index(drop=True)
        return df

    def _aggregate_constructor(self, df: pd.DataFrame) -> pd.DataFrame:
        base_columns = [
            c for c in df.columns
            if c.startswith(("constructor_", "circuit_"))
            or c in BASE_CONTEXT_FEATURES
            or c in {"race_id", "season", "round", "circuit_id", "race_name", "circuit_name", "race_date", "field_size"}
        ]
        group_keys = ["race_id", "constructor_id"]
        first_columns = [c for c in base_columns if c in df.columns and c not in group_keys]
        keys = [df["race_id"], df["constructor_id"]]
        grouped = df.groupby(group_keys, sort=False)

        # First value per group for race/circuit/constructor context.
        firsts = grouped[first_columns].first()

        pieces: list[pd.DataFrame] = [firsts]

        # Driver-level aggregate features: mean/max/min per constructor-race.
        driver_agg_columns = [c for c in DRIVER_AGG_FEATURES if c in df.columns]
        if driver_agg_columns:
            numeric_driver = df[driver_agg_columns].apply(pd.to_numeric, errors="coerce")
            grouped_driver = numeric_driver.groupby(keys, sort=False)
            pieces.extend(
                [
                    grouped_driver.mean().add_prefix("team_avg_driver_"),
                    grouped_driver.max().add_prefix("team_max_driver_"),
                    grouped_driver.min().add_prefix("team_min_driver_"),
                ]
            )

        # Team result aggregates.
        finish = pd.to_numeric(df["finish_position"], errors="coerce")
        classified = pd.to_numeric(df["classified_position"], errors="coerce")
        total_points = pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
        race_points = pd.to_numeric(df["race_points"], errors="coerce").fillna(0.0)
        sprint_points = pd.to_numeric(df["sprint_points"], errors="coerce").fillna(0.0)
        dnf = pd.to_numeric(df["dnf"], errors="coerce").fillna(0.0)
        win = pd.to_numeric(df["win"], errors="coerce").fillna(0.0)
        podium = pd.to_numeric(df["podium"], errors="coerce").fillna(0.0)
        top10 = pd.to_numeric(df["top10"], errors="coerce").fillna(0.0)

        team_stats = pd.DataFrame(
            {
                "team_total_points": total_points.groupby(keys, sort=False).sum(),
                "team_race_points": race_points.groupby(keys, sort=False).sum(),
                "team_sprint_points": sprint_points.groupby(keys, sort=False).sum(),
                "team_best_finish": finish.groupby(keys, sort=False).min(),
                "team_avg_finish": finish.groupby(keys, sort=False).mean(),
                "team_best_classified": classified.groupby(keys, sort=False).min(),
                "team_avg_classified": classified.groupby(keys, sort=False).mean(),
                "team_dnf_count": dnf.groupby(keys, sort=False).sum().astype(int),
                "team_entries": grouped.size().astype(int),
                "team_win": win.groupby(keys, sort=False).max().astype(int),
                "team_podium_count": podium.groupby(keys, sort=False).sum().astype(int),
                "team_top10_count": top10.groupby(keys, sort=False).sum().astype(int),
            }
        )
        pieces.append(team_stats)

        constructor_df = pd.concat(pieces, axis=1).reset_index()
        constructor_df["team_driver_count"] = constructor_df["team_entries"].astype(float)
        constructor_df["season"] = pd.to_numeric(constructor_df["season"], errors="coerce").astype(int)
        constructor_df["round"] = pd.to_numeric(constructor_df["round"], errors="coerce").astype(int)
        constructor_df = constructor_df.replace([np.inf, -np.inf], np.nan)
        return constructor_df.sort_values(["race_date", "season", "round", "constructor_id"]).reset_index(drop=True)

    def _write_metadata(self, df: pd.DataFrame, constructor_df: pd.DataFrame, races_used: int) -> None:
        def _active(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
            # Columns that are entirely missing in the collected history carry no
            # signal; they are documented but excluded from the model inputs.
            return [c for c in columns if c in frame.columns and frame[c].notna().any()]

        base_all = [c for c in df.columns if _is_base_feature(c)]
        post_all = [c for c in POST_QUALI_FEATURES if c in df.columns]
        constructor_all = [
            c
            for c in constructor_df.columns
            if c not in set(ID_COLUMNS) | set(CONSTRUCTOR_TARGET_COLUMNS) | {"constructor_id"}
            and c not in {"driver_name", "driver_dob"}
        ]
        base_features = _active(df, base_all)
        post_features = _active(df, post_all)
        constructor_features = _active(constructor_df, constructor_all)
        leakage_checks = {
            "targets_not_in_base_features": sorted(set(TARGET_COLUMNS) & set(base_features)),
            "targets_not_in_post_features": sorted(set(TARGET_COLUMNS) & set(post_features)),
            "targets_not_in_constructor_features": sorted(
                set(CONSTRUCTOR_TARGET_COLUMNS) & set(constructor_features)
            ),
            "post_quali_columns": post_features,
            "mode_1_feature_count": len(base_features),
            "mode_2_feature_count": len(base_features) + len(post_features),
        }
        if any(
            leakage_checks[key]
            for key in (
                "targets_not_in_base_features",
                "targets_not_in_post_features",
                "targets_not_in_constructor_features",
            )
        ):
            raise AssertionError(f"Leakage guard failed: {leakage_checks}")
        metadata = {
            "built_at": utc_now_iso(),
            "rows": int(len(df)),
            "constructor_rows": int(len(constructor_df)),
            "races_used": int(races_used),
            "seasons": [int(df["season"].min()), int(df["season"].max())],
            "feature_columns_base": base_features,
            "feature_columns_post_quali": post_features,
            "feature_columns_mode_1": base_features,
            "feature_columns_mode_2": base_features + post_features,
            "feature_columns_base_all": base_all,
            "feature_columns_post_quali_all": post_all,
            "dropped_all_nan_features": {
                "driver": sorted(set(base_all + post_all) - set(base_features) - set(post_features)),
                "constructor": sorted(set(constructor_all) - set(constructor_features)),
            },
            "target_columns": TARGET_COLUMNS,
            "constructor_feature_columns": constructor_features,
            "constructor_target_columns": CONSTRUCTOR_TARGET_COLUMNS,
            "leakage_checks": leakage_checks,
            "dataset_hash": dataframe_hash(df),
            "notes": [
                "All historical aggregates are updated only after each race.",
                "Qualifying/grid/sprint columns are isolated as post-qualifying features.",
                "Target race results are never used to construct its own feature row.",
                "Missing source data remains missing; no synthetic results are inserted.",
            ],
        }
        save_json(config.DATASET_METADATA_FILE, metadata)

    # ------------------------------------------------------------------
    # Prediction-time features
    # ------------------------------------------------------------------
    def build_prediction_features(
        self,
        race_id: str,
        *,
        entries: pd.DataFrame | list[dict[str, Any]] | None = None,
        mode: str = "after_qualifying",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if mode not in {"before_qualifying", "after_qualifying"}:
            raise ValueError("mode must be 'before_qualifying' or 'after_qualifying'")
        season, round_number = parse_race_id(race_id)
        race_id = make_race_id(season, round_number)
        if race_id not in self.race_by_id:
            raise KeyError(
                f"Race {race_id} is not in the collected schedule. "
                "Collect the season schedule first."
            )
        target_race = self.race_by_id[race_id]
        target_order = self.race_order[race_id]
        self._reset_state()

        # Process every race strictly before the target.
        prior_races = self.races[self.races["_order"] < target_order].sort_values("_order")
        for _, race in prior_races.iterrows():
            rid = str(race["race_id"])
            results = self.results_by_race.get(rid)
            if results is not None and not results.empty:
                self._update_state(race, results)

        # Reset season statistics when predicting the first race of a new season.
        for state in self.driver_states.values():
            state.ensure_season(season)
        for state in self.constructor_states.values():
            state.ensure_season(season)

        actual_results = self.results_by_race.get(race_id)
        entry_df = self._resolve_entries(race_id, target_order, entries, actual_results)
        if entry_df.empty:
            raise RuntimeError(
                f"No entries available for {race_id}. Provide --entries or collect the season."
            )
        rows = self._features_for_race(
            target_race,
            entry_df,
            mode=mode,
            use_target=False,
            actual_results=actual_results,
        )
        features = pd.DataFrame(rows)
        features["race_date"] = pd.to_datetime(features["race_date"], errors="coerce")
        info = {
            "race_id": race_id,
            "season": season,
            "round": round_number,
            "race_name": target_race.get("race_name"),
            "circuit_id": target_race.get("circuit_id"),
            "race_date": str(target_race.get("race_date")),
            "mode": mode,
            "entries_source": "actual_results" if actual_results is not None and not actual_results.empty else ("provided" if entries is not None else "latest_known_lineup"),
            "has_actual_results": bool(actual_results is not None and not actual_results.empty),
            "has_qualifying": bool(any((race_id, str(d)) in self.quali_lookup for d in entry_df["driver_id"])),
            "has_sprint_results": bool(any((race_id, str(d)) in self.sprint_lookup for d in entry_df["driver_id"])),
            "is_sprint_weekend": bool(target_race.get("is_sprint_weekend", False)),
        }
        return features, info

    def _resolve_entries(
        self,
        race_id: str,
        target_order: int,
        entries: pd.DataFrame | list[dict[str, Any]] | None,
        actual_results: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if actual_results is not None and not actual_results.empty:
            cols = ["driver_id", "constructor_id", "driver_name", "constructor_name", "driver_dob"]
            return actual_results[[c for c in cols if c in actual_results.columns]].drop_duplicates("driver_id")
        if entries is not None:
            entry_df = entries.copy() if isinstance(entries, pd.DataFrame) else pd.DataFrame(entries)
            if "driver_id" not in entry_df.columns or "constructor_id" not in entry_df.columns:
                raise ValueError("entries must contain driver_id and constructor_id")
            if "driver_dob" not in entry_df.columns:
                entry_df["driver_dob"] = entry_df["driver_id"].astype(str).map(self.driver_dob)
            if "driver_name" not in entry_df.columns:
                entry_df["driver_name"] = entry_df["driver_id"].astype(str)
            if "constructor_name" not in entry_df.columns:
                entry_df["constructor_name"] = entry_df["constructor_id"].astype(str)
            return entry_df.drop_duplicates("driver_id")
        # Fallback: latest known lineup before the target race.
        prior = self.races[self.races["_order"] < target_order].sort_values("_order", ascending=False)
        for _, race in prior.iterrows():
            rid = str(race["race_id"])
            results = self.results_by_race.get(rid)
            if results is not None and not results.empty:
                cols = ["driver_id", "constructor_id", "driver_name", "constructor_name", "driver_dob"]
                return results[[c for c in cols if c in results.columns]].drop_duplicates("driver_id")
        return pd.DataFrame()


def load_feature_dataset() -> pd.DataFrame:
    return pd.read_parquet(config.RACE_FEATURES_FILE) if config.RACE_FEATURES_FILE.exists() else pd.DataFrame()


def load_constructor_dataset() -> pd.DataFrame:
    return (
        pd.read_parquet(config.CONSTRUCTOR_FEATURES_FILE)
        if config.CONSTRUCTOR_FEATURES_FILE.exists()
        else pd.DataFrame()
    )


def get_feature_columns(mode: str = "after_qualifying") -> list[str]:
    from src.utils import load_json

    metadata = load_json(config.DATASET_METADATA_FILE, default=None)
    if metadata is None:
        raise FileNotFoundError("Dataset metadata not found; run scripts/build_dataset.py first.")
    key = "feature_columns_mode_2" if mode == "after_qualifying" else "feature_columns_mode_1"
    return list(metadata.get(key, []))


def get_constructor_feature_columns() -> list[str]:
    from src.utils import load_json

    metadata = load_json(config.DATASET_METADATA_FILE, default=None)
    if metadata is None:
        raise FileNotFoundError("Dataset metadata not found; run scripts/build_dataset.py first.")
    return list(metadata.get("constructor_feature_columns", []))
