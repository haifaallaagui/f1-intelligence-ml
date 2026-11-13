"""Import Jolpica's official CSV database dump into Ergast-shaped raw JSON.

Jolpica publishes a free, delayed CSV dump of its complete Formula 1 database
(https://api.jolpi.ca/data/dumps/download/delayed/?dump_type=csv).  The public
API is rate limited to 500 requests/hour, which makes a full 1995-2026
collection impractically slow.  The dump contains exactly the same data, so
this module converts it into the raw file layout that the rest of the project
already consumes.

Provenance
----------
* Source: Jolpica-F1 (https://api.jolpi.ca/), CC BY-NC-SA 4.0.
* The dump is downloaded by the user; this module never invents a value.
* Rows that the dump does not contain stay missing (``None``), they are never
  forward-filled from another race.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
from src.data_collection.base import BaseCollector
from src.utils import ensure_dirs, get_logger, load_json, save_json, slug, utc_now_iso

LOG = get_logger("jolpica.dump")

QUALIFYING_TYPES = ("QB", "QO", "QA", "Q1", "Q2", "Q3")
RACE_TYPE = "R"
SPRINT_TYPE = "SR"

_UNCLASSIFIED_TEXT = {
    "Disqualified": "D",
    "Excluded": "E",
    "Withdrew": "W",
    "Did not qualify": "F",
    "Did not pre-qualify": "F",
    "Not classified": "N",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NaT:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _num(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def _clean(value: Any) -> Any:
    """Recursively replace NaN/NaT/numpy scalars so the JSON is valid."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if not _is_missing(v)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if _is_missing(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class JolpicaDumpImporter(BaseCollector):
    """Convert ``formula_one_*.csv`` dump tables into raw Ergast JSON files."""

    source_name = "jolpica"

    def __init__(self, dump_dir: Path, raw_dir: Path | None = None):
        super().__init__(raw_dir or config.JOLPICA_RAW_DIR)
        self.dump_dir = Path(dump_dir)
        self._loaded = False
        self._best_lap_by_entry: dict[int, dict[str, Any]] = {}
        self._fastest_lap_by_entry: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _read(self, table: str, **kwargs: Any) -> pd.DataFrame:
        path = self.dump_dir / f"formula_one_{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Jolpica dump table: {path}")
        return pd.read_csv(path, low_memory=False, **kwargs)

    def load(self) -> None:
        if self._loaded:
            return
        self.seasons = self._read("season")
        self.rounds = self._read("round")
        self.circuits = self._read("circuit")
        self.teams = self._read("team")
        self.drivers = self._read("driver")
        self.team_drivers = self._read("teamdriver")
        self.round_entries = self._read("roundentry")
        self.sessions = self._read("session")
        self.session_entries = self._read("sessionentry")
        self.driver_championship = self._read("driverchampionship")
        self.team_championship = self._read("teamchampionship")

        self.season_year = dict(zip(self.seasons["id"], self.seasons["year"]))
        self.circuit_by_id = self.circuits.set_index("id").to_dict("index")
        self.team_by_id = self.teams.set_index("id").to_dict("index")
        self.driver_by_id = self.drivers.set_index("id").to_dict("index")
        self.team_driver_by_id = self.team_drivers.set_index("id").to_dict("index")
        self.round_entry_by_id = self.round_entries.set_index("id").to_dict("index")
        self.session_by_id = self.sessions.set_index("id").to_dict("index")

        # driver -> teams for each season (used in driver standings)
        self.season_driver_teams: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in self.team_drivers.itertuples(index=False):
            key = (int(row.season_id), int(row.driver_id))
            team = self.team_by_id.get(int(row.team_id))
            if team is not None:
                self.season_driver_teams.setdefault(key, []).append(team)

        self._load_laps(self.session_entries["id"].tolist())
        self._loaded = True

    # ------------------------------------------------------------------
    # JSON fragment helpers
    # ------------------------------------------------------------------
    def _driver_json(self, driver_id: int) -> dict[str, Any]:
        row = self.driver_by_id.get(int(driver_id), {})
        given = _text(row.get("forename"))
        family = _text(row.get("surname"))
        return _clean(
            {
                "driverId": row.get("reference") or slug(f"{given or ''} {family or ''}"),
                "permanentNumber": _text(row.get("permanent_car_number")),
                "code": _text(row.get("abbreviation")),
                "givenName": given,
                "familyName": family,
                "dateOfBirth": _text(row.get("date_of_birth")),
                "nationality": _text(row.get("nationality")),
                "url": _text(row.get("wikipedia")),
            }
        )

    def _constructor_json(self, team_id: int) -> dict[str, Any]:
        row = self.team_by_id.get(int(team_id), {})
        name = _text(row.get("name"))
        return _clean(
            {
                "constructorId": row.get("reference") or slug(name or "unknown"),
                "name": name,
                "nationality": _text(row.get("nationality")),
                "url": _text(row.get("wikipedia")),
            }
        )

    def _circuit_json(self, circuit_id: int) -> dict[str, Any]:
        row = self.circuit_by_id.get(int(circuit_id), {})
        name = _text(row.get("name"))
        return _clean(
            {
                "circuitId": row.get("reference") or slug(name or "unknown"),
                "circuitName": name,
                "url": _text(row.get("wikipedia")),
                "Location": {
                    "lat": _text(row.get("latitude")),
                    "long": _text(row.get("longitude")),
                    "locality": _text(row.get("locality")),
                    "country": _text(row.get("country")),
                    "alt": _text(row.get("altitude")),
                },
            }
        )

    def _entry_context(self, entry: pd.Series) -> dict[str, Any] | None:
        """Resolve a session entry to its round, driver, team and car number."""
        round_entry = self.round_entry_by_id.get(int(entry["round_entry_id"]))
        if round_entry is None:
            return None
        team_driver = self.team_driver_by_id.get(int(round_entry["team_driver_id"]))
        if team_driver is None:
            return None
        return {
            "driver_id": int(team_driver["driver_id"]),
            "team_id": int(team_driver["team_id"]),
            "car_number": _text(round_entry.get("car_number")),
            "round_id": int(round_entry["round_id"]),
        }

    # ------------------------------------------------------------------
    # Laps (only used for fastest-lap and qualifying times)
    # ------------------------------------------------------------------
    def _load_laps(self, entry_ids: Iterable[int]) -> None:
        entry_ids = {int(e) for e in entry_ids}
        if not entry_ids:
            return
        path = self.dump_dir / "formula_one_lap.csv"
        if not path.exists():
            LOG.warning("Lap table missing; fastest-lap and qualifying times will be unavailable")
            return
        laps = pd.read_csv(
            path,
            usecols=["session_entry_id", "time", "number", "average_speed", "is_entry_fastest_lap"],
            low_memory=False,
        )
        laps = laps[laps["session_entry_id"].isin(entry_ids)].copy()
        laps["seconds"] = pd.to_timedelta(laps["time"], errors="coerce").dt.total_seconds()
        laps = laps[laps["seconds"].notna()]

        # Best lap per session entry (used for qualifying times).
        best_idx = laps.groupby("session_entry_id")["seconds"].idxmin()
        for row in laps.loc[best_idx].itertuples(index=False):
            self._best_lap_by_entry[int(row.session_entry_id)] = {
                "time": _text(row.time),
                "number": _int(row.number),
                "average_speed": _num(row.average_speed),
            }

        # Officially flagged fastest lap per session entry.
        flagged = laps[laps["is_entry_fastest_lap"].astype(str).str.lower() == "t"]
        for row in flagged.itertuples(index=False):
            self._fastest_lap_by_entry[int(row.session_entry_id)] = {
                "time": _text(row.time),
                "number": _int(row.number),
                "average_speed": _num(row.average_speed),
            }

    def _fastest_lap(self, entry_id: int) -> dict[str, Any] | None:
        chosen = self._fastest_lap_by_entry.get(int(entry_id)) or self._best_lap_by_entry.get(int(entry_id))
        if not chosen or not chosen.get("time"):
            return None
        payload: dict[str, Any] = {"Time": {"time": chosen["time"]}}
        if chosen.get("number") is not None:
            payload["lap"] = str(chosen["number"])
        if chosen.get("average_speed") is not None:
            payload["AverageSpeed"] = {
                "speed": str(round(chosen["average_speed"], 3)),
                "units": "kph",
            }
        return payload

    def _best_lap_time(self, entry_id: int) -> str | None:
        chosen = self._best_lap_by_entry.get(int(entry_id))
        return chosen["time"] if chosen else None

    # ------------------------------------------------------------------
    # Season conversion
    # ------------------------------------------------------------------
    def _season_frames(self, season_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rounds = self.rounds[
            (self.rounds["season_id"] == season_id)
            & (self.rounds["is_cancelled"].astype(str).str.lower() != "t")
        ].copy()
        sessions = self.sessions[self.sessions["round_id"].isin(rounds["id"])].copy()
        entries = self.session_entries[self.session_entries["session_id"].isin(sessions["id"])].copy()
        return rounds, sessions, entries

    def _round_meta(self, round_row: pd.Series, season: int) -> dict[str, Any]:
        return {
            "season": str(season),
            "round": str(_int(round_row.get("number")) or 0),
            "url": _text(round_row.get("wikipedia")),
            "raceName": _text(round_row.get("name")),
            "Circuit": self._circuit_json(int(round_row["circuit_id"])),
            "date": _text(round_row.get("date")),
        }

    def _schedule(self, season: int, rounds: pd.DataFrame) -> dict[str, Any]:
        return {
            "RaceTable": {
                "season": str(season),
                "Races": [self._round_meta(row, season) for _, row in rounds.sort_values("number").iterrows()],
            }
        }

    def _result_entry(
        self,
        entry: pd.Series,
        context: dict[str, Any],
        scheduled_laps: int | None,
        order: int,
    ) -> dict[str, Any]:
        position = _int(entry.get("position"))
        detail = _text(entry.get("detail"))
        status_code = _int(entry.get("status"))
        classified = entry.get("is_classified")
        classified = bool(classified) if not _is_missing(classified) else position is not None

        if classified and position is not None:
            position_text = str(position)
        else:
            position_text = _UNCLASSIFIED_TEXT.get(detail or "", "R")

        if status_code == 0:
            status = "Finished"
        elif status_code == 1:
            laps = _int(entry.get("laps_completed"))
            gap = (scheduled_laps - laps) if (scheduled_laps and laps is not None) else 1
            status = f"+{max(1, gap)} Lap" + ("s" if max(1, gap) > 1 else "")
        else:
            status = detail or ("Retired" if not classified else "Finished")

        payload: dict[str, Any] = {
            "number": context.get("car_number"),
            "position": str(position) if position is not None else None,
            "positionText": position_text,
            "points": str(_num(entry.get("points")) or 0.0),
            "Driver": self._driver_json(context["driver_id"]),
            "Constructor": self._constructor_json(context["team_id"]),
            "grid": str(_int(entry.get("grid")) or 0),
            "laps": str(_int(entry.get("laps_completed")) or 0),
            "status": status,
            "statusId": str(status_code) if status_code is not None else None,
            "order": str(order),
        }
        fastest_rank = _int(entry.get("fastest_lap_rank"))
        fastest = self._fastest_lap(int(entry["id"]))
        if fastest is not None and fastest_rank:
            fastest["rank"] = str(fastest_rank)
            payload["FastestLap"] = fastest
        if _text(entry.get("time")):
            payload["Time"] = {"time": _text(entry.get("time"))}
        return _clean(payload)

    def _race_results(
        self,
        season: int,
        round_row: pd.Series,
        sessions: pd.DataFrame,
        entries: pd.DataFrame,
    ) -> dict[str, Any] | None:
        race_sessions = sessions[(sessions["round_id"] == round_row["id"]) & (sessions["type"] == RACE_TYPE)]
        if race_sessions.empty:
            return None
        race_session = race_sessions.iloc[0]
        rows = entries[entries["session_id"] == race_session["id"]]
        if rows.empty:
            return None

        scheduled_laps = _int(race_session.get("scheduled_laps"))
        resolved = []
        for _, entry in rows.iterrows():
            context = self._entry_context(entry)
            if context is None:
                continue
            resolved.append((entry, context))

        def sort_key(item: tuple[pd.Series, dict[str, Any]]) -> tuple[int, float, float]:
            entry, _ = item
            position = _num(entry.get("position"))
            classified = entry.get("is_classified")
            is_classified = bool(classified) if not _is_missing(classified) else position is not None
            laps = _num(entry.get("laps_completed")) or 0.0
            return (0 if is_classified else 1, position if position is not None else 1e9, -laps)

        resolved.sort(key=sort_key)
        results = [
            self._result_entry(entry, context, scheduled_laps, order=index + 1)
            for index, (entry, context) in enumerate(resolved)
        ]
        race = self._round_meta(round_row, season)
        race["Results"] = results
        return _clean(race)

    def _sprint_results(
        self,
        season: int,
        round_row: pd.Series,
        sessions: pd.DataFrame,
        entries: pd.DataFrame,
    ) -> dict[str, Any] | None:
        sprint_sessions = sessions[(sessions["round_id"] == round_row["id"]) & (sessions["type"] == SPRINT_TYPE)]
        if sprint_sessions.empty:
            return None
        sprint_session = sprint_sessions.iloc[0]
        rows = entries[entries["session_id"] == sprint_session["id"]]
        if rows.empty:
            return None
        scheduled_laps = _int(sprint_session.get("scheduled_laps"))
        resolved = []
        for _, entry in rows.iterrows():
            context = self._entry_context(entry)
            if context is not None:
                resolved.append((entry, context))
        resolved.sort(key=lambda item: _num(item[0].get("position")) or 1e9)
        sprint_results = []
        for index, (entry, context) in enumerate(resolved):
            item = self._result_entry(entry, context, scheduled_laps, order=index + 1)
            sprint_results.append(item)
        race = self._round_meta(round_row, season)
        race["SprintResults"] = sprint_results
        return _clean(race)

    def _qualifying_results(
        self,
        season: int,
        round_row: pd.Series,
        sessions: pd.DataFrame,
        entries: pd.DataFrame,
    ) -> dict[str, Any] | None:
        q_sessions = sessions[
            (sessions["round_id"] == round_row["id"]) & (sessions["type"].isin(QUALIFYING_TYPES))
        ]
        if q_sessions.empty:
            return None

        # Modern knockout qualifying: positions inside Q3 are the overall top
        # 10, positions inside Q2 are 1..15 and Q1 is 1..20.  The official
        # overall order is therefore Q3 position, then Q2, then Q1.
        by_type = {str(row["type"]): row for _, row in q_sessions.iterrows()}
        modern = all(t in by_type for t in ("Q1", "Q2", "Q3")) or "Q1" in by_type
        if modern:
            per_type: dict[str, pd.DataFrame] = {}
            for qtype in ("Q1", "Q2", "Q3"):
                session = by_type.get(qtype)
                if session is None:
                    continue
                per_type[qtype] = entries[entries["session_id"] == session["id"]]
            if not per_type:
                return None
            drivers: dict[int, dict[str, Any]] = {}
            for qtype in ("Q1", "Q2", "Q3"):
                frame = per_type.get(qtype)
                if frame is None:
                    continue
                for _, entry in frame.iterrows():
                    context = self._entry_context(entry)
                    if context is None:
                        continue
                    record = drivers.setdefault(
                        context["driver_id"],
                        {"context": context, "position": None, "Q1": None, "Q2": None, "Q3": None},
                    )
                    record[qtype] = self._best_lap_time(int(entry["id"])) or _text(entry.get("time"))
                    # Later sessions override earlier ones: Q3 position is the
                    # overall top-10 order, Q2 covers 1..15 and Q1 1..20.
                    position = _int(entry.get("position"))
                    if position is not None:
                        record["position"] = position

            results = []
            for driver_id, record in drivers.items():
                if record["position"] is None:
                    continue
                results.append(
                    _clean(
                        {
                            "number": record["context"].get("car_number"),
                            "position": str(record["position"]),
                            "Driver": self._driver_json(driver_id),
                            "Constructor": self._constructor_json(record["context"]["team_id"]),
                            "Q1": record["Q1"],
                            "Q2": record["Q2"],
                            "Q3": record["Q3"],
                        }
                    )
                )
        else:
            # Pre-2006 qualifying (single session, two-session aggregate or
            # one-shot qualifying).  The populated session with the most
            # entries carries the official classification.
            candidates = []
            for _, session in q_sessions.iterrows():
                frame = entries[entries["session_id"] == session["id"]]
                if not frame.empty:
                    candidates.append((len(frame), session, frame))
            if not candidates:
                return None
            _, session, frame = max(candidates, key=lambda item: item[0])
            results = []
            for _, entry in frame.sort_values("position").iterrows():
                context = self._entry_context(entry)
                if context is None:
                    continue
                best = self._best_lap_time(int(entry["id"])) or _text(entry.get("time"))
                results.append(
                    _clean(
                        {
                            "number": context.get("car_number"),
                            "position": str(_int(entry.get("position"))) if _int(entry.get("position")) else None,
                            "Driver": self._driver_json(context["driver_id"]),
                            "Constructor": self._constructor_json(context["team_id"]),
                            "Q1": best,
                        }
                    )
                )

        if not results:
            return None
        results.sort(key=lambda item: int(item.get("position") or 10**6))
        race = self._round_meta(round_row, season)
        race["QualifyingResults"] = results
        return _clean(race)

    def _driver_standings(self, season: int) -> dict[str, Any]:
        rows = self.driver_championship[
            (self.driver_championship["year"] == season)
            & self.driver_championship["round_number"].notna()
            & (self.driver_championship["round_number"] > 0)
            & self.driver_championship["position"].notna()
        ].copy()
        rows = rows.sort_values(["round_number", "id"]).drop_duplicates(
            ["round_number", "driver_id"], keep="last"
        )
        lists = []
        for round_number, group in rows.groupby("round_number"):
            standings = []
            for _, item in group.sort_values("position").iterrows():
                driver_id = int(item["driver_id"])
                teams = self.season_driver_teams.get((season, driver_id), [])
                standings.append(
                    _clean(
                        {
                            "position": str(_int(item.get("position"))),
                            "positionText": str(_int(item.get("position"))),
                            "points": str(_num(item.get("points")) or 0.0),
                            "wins": str(_int(item.get("win_count")) or 0),
                            "Driver": self._driver_json(driver_id),
                            "Constructors": [self._constructor_json(int(t["id"])) for t in teams],
                        }
                    )
                )
            if standings:
                lists.append(
                    {
                        "season": str(season),
                        "round": str(_int(round_number)),
                        "DriverStandings": standings,
                    }
                )
        return {"StandingsTable": {"season": str(season), "StandingsLists": lists}}

    def _constructor_standings(self, season: int) -> dict[str, Any]:
        rows = self.team_championship[
            (self.team_championship["year"] == season)
            & self.team_championship["round_number"].notna()
            & (self.team_championship["round_number"] > 0)
            & self.team_championship["position"].notna()
        ].copy()
        rows = rows.sort_values(["round_number", "id"]).drop_duplicates(
            ["round_number", "team_id"], keep="last"
        )
        lists = []
        for round_number, group in rows.groupby("round_number"):
            standings = []
            for _, item in group.sort_values("position").iterrows():
                standings.append(
                    _clean(
                        {
                            "position": str(_int(item.get("position"))),
                            "positionText": str(_int(item.get("position"))),
                            "points": str(_num(item.get("points")) or 0.0),
                            "wins": str(_int(item.get("win_count")) or 0),
                            "Constructor": self._constructor_json(int(item["team_id"])),
                        }
                    )
                )
            if standings:
                lists.append(
                    {
                        "season": str(season),
                        "round": str(_int(round_number)),
                        "ConstructorStandings": standings,
                    }
                )
        return {"StandingsTable": {"season": str(season), "StandingsLists": lists}}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    @staticmethod
    def _count_existing(path: Path, key: str, subkey: str) -> int:
        payload = load_json(path, default={}) or {}
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        races = (data.get("RaceTable", {}) or {}).get("Races", []) or []
        return sum(1 for race in races if race.get(subkey))

    @staticmethod
    def _count_standings(path: Path) -> int:
        payload = load_json(path, default={}) or {}
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        return len((data.get("StandingsTable", {}) or {}).get("StandingsLists", []) or [])

    def _write_season_file(
        self,
        path: Path,
        payload: dict[str, Any],
        endpoint: str,
        season: int,
        *,
        existing_count: int,
        new_count: int,
        force: bool,
    ) -> bool:
        if path.exists() and not force and existing_count >= new_count:
            return False
        self.write_json_payload(
            path,
            payload,
            endpoint=endpoint,
            metadata={
                "season": season,
                "endpoint": path.stem,
                "total": new_count,
                "available": new_count > 0,
                "imported_from": "jolpica_csv_dump",
            },
        )
        return True

    def import_seasons(self, start: int, end: int, *, force: bool = False) -> dict[str, Any]:
        self.load()
        written = 0
        seasons_written: list[int] = []
        latest_race: str | None = None

        for season in range(int(start), int(end) + 1):
            season_ids = [sid for sid, year in self.season_year.items() if int(year) == season]
            if not season_ids:
                LOG.warning("Season %s is not present in the Jolpica dump", season)
                continue
            rounds, sessions, entries = self._season_frames(season_ids[0])
            if rounds.empty:
                continue
            season_dir = self.raw_dir / str(season)
            ensure_dirs(season_dir)

            schedule = self._schedule(season, rounds)
            race_payloads = []
            qualifying_payloads = []
            sprint_payloads = []
            for _, round_row in rounds.sort_values("number").iterrows():
                result = self._race_results(season, round_row, sessions, entries)
                if result is not None:
                    race_payloads.append(result)
                qualifying = self._qualifying_results(season, round_row, sessions, entries)
                if qualifying is not None:
                    qualifying_payloads.append(qualifying)
                sprint = self._sprint_results(season, round_row, sessions, entries)
                if sprint is not None:
                    sprint_payloads.append(sprint)

            schedule_payload = {"RaceTable": {"season": str(season), "Races": schedule["RaceTable"]["Races"]}}
            results_payload = {"RaceTable": {"season": str(season), "Races": race_payloads}}
            qualifying_payload = {"RaceTable": {"season": str(season), "Races": qualifying_payloads}}
            sprint_payload = {"RaceTable": {"season": str(season), "Races": sprint_payloads}}
            driver_standings = self._driver_standings(season)
            constructor_standings = self._constructor_standings(season)

            changed = False
            changed |= self._write_season_file(
                season_dir / "schedule.json",
                schedule_payload,
                f"/{season}",
                season,
                existing_count=self._count_existing(season_dir / "schedule.json", "RaceTable", "round"),
                new_count=len(schedule_payload["RaceTable"]["Races"]),
                force=force,
            )
            changed |= self._write_season_file(
                season_dir / "results.json",
                results_payload,
                f"/{season}/results",
                season,
                existing_count=self._count_existing(season_dir / "results.json", "RaceTable", "Results"),
                new_count=len(race_payloads),
                force=force,
            )
            # The dump has incomplete qualifying classifications for 1999-2002.
            # Skip writing in that case so the API collector can fill the gap
            # instead of persisting a misleadingly partial file.
            qualifying_complete = len(qualifying_payloads) >= max(1, int(0.8 * len(race_payloads)))
            if qualifying_complete:
                changed |= self._write_season_file(
                    season_dir / "qualifying.json",
                    qualifying_payload,
                    f"/{season}/qualifying",
                    season,
                    existing_count=self._count_existing(
                        season_dir / "qualifying.json", "RaceTable", "QualifyingResults"
                    ),
                    new_count=len(qualifying_payloads),
                    force=force,
                )
            else:
                LOG.warning(
                    "Season %s: dump has qualifying for only %s/%s races; leaving the API collector to fill it",
                    season,
                    len(qualifying_payloads),
                    len(race_payloads),
                )
            if sprint_payloads:
                changed |= self._write_season_file(
                    season_dir / "sprint.json",
                    sprint_payload,
                    f"/{season}/sprint",
                    season,
                    existing_count=self._count_existing(season_dir / "sprint.json", "RaceTable", "SprintResults"),
                    new_count=len(sprint_payloads),
                    force=force,
                )
            changed |= self._write_season_file(
                season_dir / "driverstandings.json",
                driver_standings,
                f"/{season}/driverstandings",
                season,
                existing_count=self._count_standings(season_dir / "driverstandings.json"),
                new_count=len(driver_standings["StandingsTable"]["StandingsLists"]),
                force=force,
            )
            changed |= self._write_season_file(
                season_dir / "constructorstandings.json",
                constructor_standings,
                f"/{season}/constructorstandings",
                season,
                existing_count=self._count_standings(season_dir / "constructorstandings.json"),
                new_count=len(constructor_standings["StandingsTable"]["StandingsLists"]),
                force=force,
            )

            if changed:
                written += 1
                seasons_written.append(season)
                if race_payloads:
                    last = race_payloads[-1]
                    latest_race = f"{season}_{int(last.get('round', 0)):02d}"
            LOG.info(
                "Jolpica dump season %s: %s races with results, %s qualifying, %s sprints",
                season,
                len(race_payloads),
                len(qualifying_payloads),
                len(sprint_payloads),
            )

        self.merge_reference()
        self.save_manifest()
        return {
            "source": self.source_name,
            "imported_from": str(self.dump_dir),
            "seasons_requested": [int(start), int(end)],
            "seasons_written": seasons_written,
            "files_written": written,
            "latest_race": latest_race,
            "imported_at": utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # Reference tables
    # ------------------------------------------------------------------
    def merge_reference(self) -> None:
        """Add any dump drivers/teams/circuits missing from the reference files."""
        ref_dir = self.raw_dir / "reference"
        ensure_dirs(ref_dir)

        drivers_path = ref_dir / "drivers.json"
        drivers_payload = load_json(drivers_path, default={}) or {}
        drivers_data = drivers_payload.get("data", drivers_payload) if isinstance(drivers_payload, dict) else {}
        drivers = (drivers_data.get("DriverTable", {}) or {}).get("Drivers", []) or []
        known = {str(d.get("driverId")) for d in drivers}
        added = 0
        for row in self.drivers.itertuples(index=False):
            driver_id = _text(row.reference)
            if not driver_id or driver_id in known:
                continue
            drivers.append(self._driver_json(int(row.id)))
            known.add(driver_id)
            added += 1
        if added:
            self.write_json_payload(
                drivers_path,
                {"DriverTable": {"season": None, "Drivers": drivers}},
                endpoint="/drivers",
                metadata={"endpoint": "drivers", "merged_from_dump": added},
            )

        constructors_path = ref_dir / "constructors.json"
        constructors_payload = load_json(constructors_path, default={}) or {}
        constructors_data = (
            constructors_payload.get("data", constructors_payload) if isinstance(constructors_payload, dict) else {}
        )
        constructors = (constructors_data.get("ConstructorTable", {}) or {}).get("Constructors", []) or []
        known = {str(c.get("constructorId")) for c in constructors}
        added = 0
        for row in self.teams.itertuples(index=False):
            constructor_id = _text(row.reference)
            if not constructor_id or constructor_id in known:
                continue
            constructors.append(self._constructor_json(int(row.id)))
            known.add(constructor_id)
            added += 1
        if added:
            self.write_json_payload(
                constructors_path,
                {"ConstructorTable": {"Constructors": constructors}},
                endpoint="/constructors",
                metadata={"endpoint": "constructors", "merged_from_dump": added},
            )

        circuits_path = ref_dir / "circuits.json"
        circuits_payload = load_json(circuits_path, default={}) or {}
        circuits_data = circuits_payload.get("data", circuits_payload) if isinstance(circuits_payload, dict) else {}
        circuits = (circuits_data.get("CircuitTable", {}) or {}).get("Circuits", []) or []
        known = {str(c.get("circuitId")) for c in circuits}
        added = 0
        for row in self.circuits.itertuples(index=False):
            circuit_id = _text(row.reference)
            if not circuit_id or circuit_id in known:
                continue
            circuits.append(self._circuit_json(int(row.id)))
            known.add(circuit_id)
            added += 1
        if added:
            self.write_json_payload(
                circuits_path,
                {"CircuitTable": {"Circuits": circuits}},
                endpoint="/circuits",
                metadata={"endpoint": "circuits", "merged_from_dump": added},
            )

        if added:
            LOG.info("Merged %s circuits from the dump into the reference table", added)
