"""Cleaning and normalisation of raw F1 data into analysis tables.

The functions in this module are deliberately explicit about provenance and
missing data.  They never forward-fill a race result, a qualifying time or a
standing from a different event.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
from src.utils import (
    ensure_dirs,
    get_logger,
    load_json,
    load_parquet,
    make_race_id,
    save_json,
    save_parquet,
    utc_now_iso,
)

LOG = get_logger("preprocessing.cleaning")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def read_jolpica(season: int, name: str) -> dict[str, Any]:
    payload = load_json(config.JOLPICA_RAW_DIR / str(season) / f"{name}.json", default={}) or {}
    data = unwrap(payload)
    return data if isinstance(data, dict) else {}


def parse_time_to_seconds(value: Any) -> float:
    """Parse '1:23.456', '23.456' or a numeric value to seconds."""
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return int(float(value))
    except (TypeError, ValueError):
        return np.nan


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _date_only(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:10]


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------
def extract_drivers() -> pd.DataFrame:
    payload = load_json(config.JOLPICA_RAW_DIR / "reference" / "drivers.json", default={}) or {}
    data = unwrap(payload) or {}
    drivers = (data.get("DriverTable", {}) or {}).get("Drivers", []) or []
    rows = []
    for d in drivers:
        rows.append(
            {
                "driver_id": d.get("driverId"),
                "driver_code": d.get("code"),
                "given_name": d.get("givenName"),
                "family_name": d.get("familyName"),
                "driver_name": f"{d.get('givenName', '')} {d.get('familyName', '')}".strip(),
                "date_of_birth": d.get("dateOfBirth"),
                "nationality": d.get("nationality"),
                "permanent_number": _to_int(d.get("permanentNumber")),
                "url": d.get("url"),
                "source": "jolpica",
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["driver_id"]) if rows else pd.DataFrame()
    return df


def extract_constructors() -> pd.DataFrame:
    payload = load_json(config.JOLPICA_RAW_DIR / "reference" / "constructors.json", default={}) or {}
    data = unwrap(payload) or {}
    constructors = (data.get("ConstructorTable", {}) or {}).get("Constructors", []) or []
    rows = [
        {
            "constructor_id": c.get("constructorId"),
            "constructor_name": c.get("name"),
            "constructor_nationality": c.get("nationality"),
            "constructor_url": c.get("url"),
            "source": "jolpica",
        }
        for c in constructors
    ]
    return pd.DataFrame(rows).drop_duplicates(subset=["constructor_id"]) if rows else pd.DataFrame()


def extract_circuits() -> pd.DataFrame:
    payload = load_json(config.JOLPICA_RAW_DIR / "reference" / "circuits.json", default={}) or {}
    data = unwrap(payload) or {}
    circuits = (data.get("CircuitTable", {}) or {}).get("Circuits", []) or []
    rows = []
    for c in circuits:
        loc = c.get("Location", {}) or {}
        rows.append(
            {
                "circuit_id": c.get("circuitId"),
                "circuit_name": c.get("circuitName"),
                "circuit_url": c.get("url"),
                "locality": loc.get("locality"),
                "country": loc.get("country"),
                "latitude": _to_float(loc.get("lat")),
                "longitude": _to_float(loc.get("long")),
                "altitude": _to_float(loc.get("alt")),
                "source": "jolpica",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["circuit_id"]) if rows else pd.DataFrame()


def extract_status() -> pd.DataFrame:
    payload = load_json(config.JOLPICA_RAW_DIR / "reference" / "status.json", default={}) or {}
    data = unwrap(payload) or {}
    statuses = (data.get("StatusTable", {}) or {}).get("Status", []) or []
    rows = [
        {"status_id": _to_int(s.get("statusId")), "status": s.get("status")}
        for s in statuses
    ]
    return pd.DataFrame(rows).drop_duplicates(subset=["status_id"]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Race / results / qualifying / sprint tables
# ---------------------------------------------------------------------------
def extract_races(start: int, end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        schedule = read_jolpica(season, "schedule")
        races = (schedule.get("RaceTable", {}) or {}).get("Races", []) or []
        if not races:
            # If the schedule endpoint is unavailable, results still give us
            # the race identity.  This is an availability fallback, not invented data.
            results = read_jolpica(season, "results")
            races = (results.get("RaceTable", {}) or {}).get("Races", []) or []
        for race in races:
            circuit = race.get("Circuit", {}) or {}
            loc = circuit.get("Location", {}) or {}
            rows.append(
                {
                    "season": int(race.get("season", season)),
                    "round": int(race.get("round", 0) or 0),
                    "race_id": make_race_id(race.get("season", season), race.get("round", 0)),
                    "race_name": race.get("raceName"),
                    "race_date": race.get("date"),
                    "race_time": race.get("time"),
                    "circuit_id": circuit.get("circuitId"),
                    "circuit_name": circuit.get("circuitName"),
                    "locality": loc.get("locality"),
                    "country": loc.get("country"),
                    "latitude": _to_float(loc.get("lat")),
                    "longitude": _to_float(loc.get("long")),
                    "altitude": _to_float(loc.get("alt")),
                    "race_url": race.get("url"),
                    "source": "jolpica",
                }
            )
    if not rows:
        return pd.DataFrame()
    races = pd.DataFrame(rows).drop_duplicates(subset=["race_id"], keep="last")
    races["total_rounds"] = races.groupby("season")["round"].transform("max")
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races["race_datetime"] = pd.to_datetime(
        races["race_date"].dt.strftime("%Y-%m-%d").fillna("") + " " + races["race_time"].fillna("00:00:00"),
        errors="coerce",
        utc=True,
    )
    # Sprint weekend detection from sprint results and, for recent seasons,
    # FastF1 event formats (which include future events).
    sprint_rounds: set[tuple[int, int]] = set()
    for season in range(int(start), int(end) + 1):
        sprint = read_jolpica(season, "sprint")
        for race in (sprint.get("RaceTable", {}) or {}).get("Races", []) or []:
            if race.get("SprintResults"):
                sprint_rounds.add((int(race.get("season", season)), int(race.get("round", 0))))
    ff1_sprint_rounds = _fastf1_sprint_rounds(start, end)
    sprint_rounds |= ff1_sprint_rounds
    races["is_sprint_weekend"] = races.apply(
        lambda r: (int(r["season"]), int(r["round"])) in sprint_rounds, axis=1
    )
    return races.sort_values(["season", "round"]).reset_index(drop=True)


def _fastf1_sprint_rounds(start: int, end: int) -> set[tuple[int, int]]:
    rounds: set[tuple[int, int]] = set()
    for season in range(max(int(start), config.FASTF1_START_SEASON), int(end) + 1):
        payload = load_json(config.FASTF1_RAW_DIR / str(season) / "event_schedule.json", default={}) or {}
        events = unwrap(payload) or []
        for event in events:
            event_format = str(event.get("EventFormat", "") or "").lower()
            if "sprint" in event_format:
                round_number = _to_int(event.get("RoundNumber"))
                if not np.isnan(round_number):
                    rounds.add((season, int(round_number)))
    return rounds


def extract_results(start: int, end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        results = read_jolpica(season, "results")
        for race in (results.get("RaceTable", {}) or {}).get("Races", []) or []:
            race_id = make_race_id(race.get("season", season), race.get("round", 0))
            field_size = len(race.get("Results", []) or [])
            for result in race.get("Results", []) or []:
                driver = result.get("Driver", {}) or {}
                constructor = result.get("Constructor", {}) or {}
                fastest = result.get("FastestLap", {}) or {}
                position = _to_int(result.get("position"))
                position_text = str(result.get("positionText", ""))
                status = str(result.get("status", "") or "")
                classified = position_text.isdigit() or (not np.isnan(position) and position_text not in {"R", "D", "E", "W", "F", "N"})
                is_dnf = (not classified) or (
                    status not in {"Finished", "Disqualified"} and not status.startswith("+")
                )
                if position_text in {"D", "E", "W", "F", "N"}:
                    is_dnf = True
                if status == "Disqualified":
                    is_dnf = True
                rows.append(
                    {
                        "race_id": race_id,
                        "season": int(race.get("season", season)),
                        "round": int(race.get("round", 0)),
                        "driver_id": driver.get("driverId"),
                        "driver_code": driver.get("code"),
                        "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                        "driver_dob": driver.get("dateOfBirth"),
                        "driver_nationality": driver.get("nationality"),
                        "constructor_id": constructor.get("constructorId"),
                        "constructor_name": constructor.get("name"),
                        "grid_raw": _to_int(result.get("grid")),
                        "position_raw": position,
                        "position_text": position_text,
                        "classified": bool(classified),
                        "is_dnf": bool(is_dnf),
                        "points": _to_float(result.get("points")),
                        "laps": _to_int(result.get("laps")),
                        "status_id": _to_int(result.get("statusId")),
                        "status": status,
                        "order": _to_int(result.get("order")),
                        "fastest_lap_rank": _to_int((fastest.get("rank"))),
                        "fastest_lap_lap": _to_int(fastest.get("lap")),
                        "fastest_lap_time": parse_time_to_seconds((fastest.get("Time", {}) or {}).get("time")),
                        "fastest_lap_speed": _to_float((fastest.get("AverageSpeed", {}) or {}).get("speed")),
                        "field_size": field_size,
                        "source": "jolpica",
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    # Classified position is the official numeric position.  For a coherent
    # ranking target, retired/unclassified drivers are placed after classified
    # drivers rather than dropped.
    df["classified_position"] = np.where(df["classified"], df["position_raw"], np.nan)
    df["finish_position"] = df["classified_position"].fillna(df["field_size"] + 1)
    df["grid_position"] = np.where(
        (df["grid_raw"] > 0) & df["grid_raw"].notna(), df["grid_raw"], df["field_size"] + 1
    )
    df["position_gain"] = df["grid_position"] - df["finish_position"]
    df["win"] = (df["classified_position"] == 1).astype(int)
    df["podium"] = (df["classified_position"].fillna(99) <= 3).astype(int)
    df["top10"] = (df["classified_position"].fillna(99) <= 10).astype(int)
    df["fastest_lap"] = (df["fastest_lap_rank"] == 1).astype(int)
    df["race_points"] = df["points"].fillna(0.0)
    df["sprint_points"] = 0.0
    df["total_points"] = df["race_points"]
    return df.sort_values(["season", "round", "finish_position", "driver_id"]).reset_index(drop=True)


def extract_qualifying(start: int, end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        qualifying = read_jolpica(season, "qualifying")
        for race in (qualifying.get("RaceTable", {}) or {}).get("Races", []) or []:
            race_id = make_race_id(race.get("season", season), race.get("round", 0))
            for result in race.get("QualifyingResults", []) or []:
                driver = result.get("Driver", {}) or {}
                constructor = result.get("Constructor", {}) or {}
                q1 = parse_time_to_seconds(result.get("Q1"))
                q2 = parse_time_to_seconds(result.get("Q2"))
                q3 = parse_time_to_seconds(result.get("Q3"))
                best = np.nanmin([q1, q2, q3]) if not all(np.isnan([q1, q2, q3])) else np.nan
                rows.append(
                    {
                        "race_id": race_id,
                        "season": int(race.get("season", season)),
                        "round": int(race.get("round", 0)),
                        "driver_id": driver.get("driverId"),
                        "driver_code": driver.get("code"),
                        "constructor_id": constructor.get("constructorId"),
                        "quali_position": _to_int(result.get("position")),
                        "q1_seconds": q1,
                        "q2_seconds": q2,
                        "q3_seconds": q3,
                        "quali_best_seconds": best,
                        "quali_reached_q2": int(not np.isnan(q2) or not np.isnan(q3)),
                        "quali_reached_q3": int(not np.isnan(q3)),
                        "source": "jolpica",
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    pole = df.groupby("race_id")["quali_best_seconds"].transform("min")
    df["quali_gap_to_pole_seconds"] = df["quali_best_seconds"] - pole
    df["quali_gap_to_pole_pct"] = np.where(
        pole > 0, df["quali_gap_to_pole_seconds"] / pole * 100.0, np.nan
    )
    # Teammate comparison: use the median best time of the other cars entered
    # by the same constructor in that race.
    teammate = []
    for _, row in df.iterrows():
        peers = df[
            (df["race_id"] == row["race_id"])
            & (df["constructor_id"] == row["constructor_id"])
            & (df["driver_id"] != row["driver_id"])
        ]["quali_best_seconds"]
        teammate.append(peers.median() if len(peers) else np.nan)
    df["teammate_quali_seconds"] = teammate
    df["quali_gap_to_teammate_seconds"] = df["quali_best_seconds"] - df["teammate_quali_seconds"]
    df["quali_gap_to_teammate_pct"] = np.where(
        df["teammate_quali_seconds"] > 0,
        df["quali_gap_to_teammate_seconds"] / df["teammate_quali_seconds"] * 100.0,
        np.nan,
    )
    return df.sort_values(["season", "round", "quali_position"]).reset_index(drop=True)


def extract_sprints(start: int, end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        sprint = read_jolpica(season, "sprint")
        for race in (sprint.get("RaceTable", {}) or {}).get("Races", []) or []:
            race_id = make_race_id(race.get("season", season), race.get("round", 0))
            field_size = len(race.get("SprintResults", []) or [])
            for result in race.get("SprintResults", []) or []:
                driver = result.get("Driver", {}) or {}
                constructor = result.get("Constructor", {}) or {}
                position = _to_int(result.get("position"))
                position_text = str(result.get("positionText", ""))
                status = str(result.get("status", "") or "")
                classified = position_text.isdigit() or (not np.isnan(position) and position_text not in {"R", "D", "E", "W", "F", "N"})
                is_dnf = (not classified) or (status not in {"Finished", "Disqualified"} and not status.startswith("+"))
                if status == "Disqualified":
                    is_dnf = True
                rows.append(
                    {
                        "race_id": race_id,
                        "season": int(race.get("season", season)),
                        "round": int(race.get("round", 0)),
                        "driver_id": driver.get("driverId"),
                        "constructor_id": constructor.get("constructorId"),
                        "sprint_position": position,
                        "sprint_position_text": position_text,
                        "sprint_classified": bool(classified),
                        "sprint_dnf": bool(is_dnf),
                        "sprint_points": _to_float(result.get("points")),
                        "sprint_grid": _to_int(result.get("grid")),
                        "sprint_laps": _to_int(result.get("laps")),
                        "sprint_status": status,
                        "sprint_field_size": field_size,
                        "source": "jolpica",
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    df["sprint_position"] = df["sprint_position"].fillna(df["sprint_field_size"] + 1)
    df["sprint_position_gain"] = df["sprint_grid"] - df["sprint_position"]
    return df.sort_values(["season", "round", "sprint_position"]).reset_index(drop=True)


def extract_standings(start: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_rows: list[dict[str, Any]] = []
    constructor_rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        ds = read_jolpica(season, "driverstandings")
        for standings_list in (ds.get("StandingsTable", {}) or {}).get("StandingsLists", []) or []:
            for item in standings_list.get("DriverStandings", []) or []:
                driver = item.get("Driver", {}) or {}
                driver_rows.append(
                    {
                        "season": int(standings_list.get("season", season)),
                        "round": int(standings_list.get("round", 0) or 0),
                        "driver_id": driver.get("driverId"),
                        "standing_position": _to_int(item.get("position")),
                        "standing_points": _to_float(item.get("points")),
                        "standing_wins": _to_int(item.get("wins")),
                        "source": "jolpica",
                    }
                )
        cs = read_jolpica(season, "constructorstandings")
        for standings_list in (cs.get("StandingsTable", {}) or {}).get("StandingsLists", []) or []:
            for item in standings_list.get("ConstructorStandings", []) or []:
                constructor = item.get("Constructor", {}) or {}
                constructor_rows.append(
                    {
                        "season": int(standings_list.get("season", season)),
                        "round": int(standings_list.get("round", 0) or 0),
                        "constructor_id": constructor.get("constructorId"),
                        "standing_position": _to_int(item.get("position")),
                        "standing_points": _to_float(item.get("points")),
                        "standing_wins": _to_int(item.get("wins")),
                        "source": "jolpica",
                    }
                )
    driver_df = pd.DataFrame(driver_rows).drop_duplicates(subset=["season", "round", "driver_id"]) if driver_rows else pd.DataFrame()
    constructor_df = pd.DataFrame(constructor_rows).drop_duplicates(subset=["season", "round", "constructor_id"]) if constructor_rows else pd.DataFrame()
    return driver_df, constructor_df


# ---------------------------------------------------------------------------
# OpenF1 and FastF1 enrichment
# ---------------------------------------------------------------------------
def _load_openf1_year(year: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meetings_payload = load_json(config.OPENF1_RAW_DIR / str(year) / "meetings.json", default={}) or {}
    sessions_payload = load_json(config.OPENF1_RAW_DIR / str(year) / "sessions.json", default={}) or {}
    return unwrap(meetings_payload) or [], unwrap(sessions_payload) or []


def _match_openf1_race(session: dict[str, Any], races: pd.DataFrame) -> pd.Series | None:
    year = int(session.get("year", 0) or 0)
    session_date = _date_only(session.get("date_start"))
    candidates = races[races["season"] == year]
    if candidates.empty:
        return None
    exact = candidates[candidates["race_date"].dt.strftime("%Y-%m-%d") == session_date]
    if not exact.empty:
        if len(exact) == 1:
            return exact.iloc[0]
        circuit = _normalise_text(session.get("circuit_short_name"))
        location = _normalise_text(session.get("location"))
        for _, row in exact.iterrows():
            if circuit and (
                circuit in _normalise_text(row.get("circuit_name"))
                or _normalise_text(row.get("locality")) in circuit
            ):
                return row
            if location and location in _normalise_text(row.get("locality")):
                return row
        return exact.iloc[0]
    # Time-zone shifts can move the UTC date by one day.
    session_ts = pd.to_datetime(session.get("date_start"), errors="coerce", utc=True)
    if pd.isna(session_ts):
        return None
    delta = (candidates["race_datetime"] - session_ts).abs().dt.total_seconds()
    if len(delta) and delta.min() <= 2 * 86400:
        return candidates.loc[delta.idxmin()]
    return None


def _openf1_driver_map(year: int, session_key: int, drivers_ref: pd.DataFrame) -> dict[int, str]:
    payload = load_json(
        config.OPENF1_RAW_DIR / str(year) / str(session_key) / "drivers.json", default={}
    ) or {}
    rows = unwrap(payload) or []
    code_to_id = {}
    name_to_id = {}
    if not drivers_ref.empty:
        code_to_id = {
            str(code).upper(): driver_id
            for code, driver_id in zip(drivers_ref["driver_code"], drivers_ref["driver_id"])
            if code and not pd.isna(code)
        }
        name_to_id = {
            _normalise_text(name): driver_id
            for name, driver_id in zip(drivers_ref["driver_name"], drivers_ref["driver_id"])
            if name and not pd.isna(name)
        }
    mapping: dict[int, str] = {}
    for row in rows:
        number = row.get("driver_number")
        if number is None:
            continue
        code = str(row.get("name_acronym", "") or "").upper()
        full_name = _normalise_text(row.get("full_name"))
        driver_id = code_to_id.get(code) or name_to_id.get(full_name)
        if driver_id:
            mapping[int(number)] = driver_id
    return mapping


def extract_openf1(start: int, end: int, races: pd.DataFrame, drivers_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build race-level and driver-level summaries from OpenF1 raw sessions."""
    race_rows: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    for year in range(max(int(start), config.OPENF1_START_SEASON), int(end) + 1):
        meetings, sessions = _load_openf1_year(year)
        meetings_by_key = {m.get("meeting_key"): m for m in meetings}
        for session in sessions:
            if str(session.get("session_name", "")) != "Race" or session.get("is_cancelled"):
                continue
            session_key = session.get("session_key")
            race = _match_openf1_race(session, races)
            if race is None:
                continue
            session_dir = config.OPENF1_RAW_DIR / str(year) / str(session_key)
            if not session_dir.exists():
                continue
            meeting = meetings_by_key.get(session.get("meeting_key"), {}) or {}

            def load(name: str) -> list[dict[str, Any]]:
                payload = load_json(session_dir / f"{name}.json", default={}) or {}
                return unwrap(payload) or []

            weather = load("weather")
            laps = load("laps")
            pit = load("pit")
            stints = load("stints")
            position = load("position")
            race_control = load("race_control")
            driver_map = _openf1_driver_map(year, int(session_key), drivers_ref)

            air = [r.get("air_temperature") for r in weather if r.get("air_temperature") is not None]
            track = [r.get("track_temperature") for r in weather if r.get("track_temperature") is not None]
            humidity = [r.get("humidity") for r in weather if r.get("humidity") is not None]
            wind = [r.get("wind_speed") for r in weather if r.get("wind_speed") is not None]
            rain = [r.get("rainfall") for r in weather if r.get("rainfall") is not None]
            sc_messages = [
                str(r.get("message", ""))
                for r in race_control
                if "SAFETY CAR" in str(r.get("message", "")).upper()
            ]
            red_flags = [
                str(r.get("message", ""))
                for r in race_control
                if "RED FLAG" in str(r.get("message", "")).upper()
            ]

            race_rows.append(
                {
                    "race_id": race["race_id"],
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "openf1_session_key": int(session_key),
                    "openf1_meeting_key": session.get("meeting_key"),
                    "openf1_circuit_key": session.get("circuit_key"),
                    "openf1_circuit_short_name": session.get("circuit_short_name"),
                    "openf1_circuit_type": meeting.get("circuit_type"),
                    "openf1_location": session.get("location"),
                    "openf1_country": session.get("country_name"),
                    "openf1_session_date": session.get("date_start"),
                    "weather_air_temp_mean": float(np.nanmean(air)) if air else np.nan,
                    "weather_track_temp_mean": float(np.nanmean(track)) if track else np.nan,
                    "weather_humidity_mean": float(np.nanmean(humidity)) if humidity else np.nan,
                    "weather_wind_mean": float(np.nanmean(wind)) if wind else np.nan,
                    "weather_rainfall_mean": float(np.nanmean(rain)) if rain else np.nan,
                    "weather_rainfall_max": float(np.nanmax(rain)) if rain else np.nan,
                    "race_control_safety_car_messages": len(sc_messages),
                    "race_control_red_flags": len(red_flags),
                    "openf1_has_weather": int(bool(weather)),
                    "openf1_has_laps": int(bool(laps)),
                    "openf1_has_pit": int(bool(pit)),
                    "openf1_has_stints": int(bool(stints)),
                    "openf1_source": "openf1",
                }
            )

            # Driver-level aggregation.
            by_driver: dict[int, dict[str, Any]] = {}
            for row in laps:
                number = row.get("driver_number")
                if number is None:
                    continue
                entry = by_driver.setdefault(int(number), {"lap_times": [], "laps": 0, "compounds": set()})
                duration = row.get("lap_duration")
                if duration is not None:
                    entry["lap_times"].append(float(duration))
                entry["laps"] += 1
                if row.get("compound"):
                    entry["compounds"].add(str(row["compound"]))
            for row in pit:
                number = row.get("driver_number")
                if number is None:
                    continue
                entry = by_driver.setdefault(int(number), {"lap_times": [], "laps": 0, "compounds": set(), "pit_durations": []})
                if row.get("pit_duration") is not None:
                    entry.setdefault("pit_durations", []).append(float(row["pit_duration"]))
            for row in stints:
                number = row.get("driver_number")
                if number is None:
                    continue
                entry = by_driver.setdefault(int(number), {"lap_times": [], "laps": 0, "compounds": set(), "stint_lengths": []})
                start_lap = row.get("lap_start")
                end_lap = row.get("lap_end")
                if start_lap is not None and end_lap is not None:
                    entry.setdefault("stint_lengths", []).append(float(end_lap) - float(start_lap) + 1)
                if row.get("compound"):
                    entry["compounds"].add(str(row["compound"]))
            positions_by_driver: dict[int, list[int]] = {}
            for row in position:
                number = row.get("driver_number")
                pos = row.get("position")
                if number is not None and pos is not None:
                    positions_by_driver.setdefault(int(number), []).append(int(pos))
            for number, entry in by_driver.items():
                driver_id = driver_map.get(number)
                lap_times = entry.get("lap_times", [])
                pit_durations = entry.get("pit_durations", [])
                stint_lengths = entry.get("stint_lengths", [])
                pos_list = positions_by_driver.get(number, [])
                driver_rows.append(
                    {
                        "race_id": race["race_id"],
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "driver_id": driver_id,
                        "openf1_driver_number": number,
                        "openf1_laps_completed": entry.get("laps", 0),
                        "openf1_mean_lap_seconds": float(np.nanmean(lap_times)) if lap_times else np.nan,
                        "openf1_lap_std_seconds": float(np.nanstd(lap_times)) if lap_times else np.nan,
                        "openf1_best_lap_seconds": float(np.nanmin(lap_times)) if lap_times else np.nan,
                        "openf1_pit_stops": len(pit_durations),
                        "openf1_mean_pit_seconds": float(np.nanmean(pit_durations)) if pit_durations else np.nan,
                        "openf1_mean_stint_laps": float(np.nanmean(stint_lengths)) if stint_lengths else np.nan,
                        "openf1_compounds": ",".join(sorted(entry.get("compounds", set()))),
                        "openf1_start_position": pos_list[0] if pos_list else np.nan,
                        "openf1_end_position": pos_list[-1] if pos_list else np.nan,
                        "openf1_position_changes": (pos_list[0] - pos_list[-1]) if len(pos_list) >= 2 else np.nan,
                        "openf1_source": "openf1",
                    }
                )
    race_df = pd.DataFrame(race_rows).drop_duplicates(subset=["race_id"], keep="last") if race_rows else pd.DataFrame()
    driver_df = pd.DataFrame(driver_rows).drop_duplicates(subset=["race_id", "driver_id"], keep="last") if driver_rows else pd.DataFrame()
    return race_df, driver_df


def extract_fastf1_qualifying(start: int, end: int, drivers_ref: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(max(int(start), config.FASTF1_START_SEASON), int(end) + 1):
        for path in sorted((config.FASTF1_RAW_DIR / str(year)).glob("*_Q/qualifying_summary.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty:
                continue
            df["season"] = int(year)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    code_to_id = {}
    if not drivers_ref.empty:
        code_to_id = {
            str(code).upper(): driver_id
            for code, driver_id in zip(drivers_ref["driver_code"], drivers_ref["driver_id"])
            if code and not pd.isna(code)
        }
    df["driver_id"] = df.get("Driver", pd.Series(dtype=object)).astype(str).str.upper().map(code_to_id)
    df["race_id"] = [
        make_race_id(season, round_number)
        for season, round_number in zip(df["season"], df["round"])
    ]
    pole = df.groupby("race_id")["LapTimeSeconds"].transform("min")
    df["fastf1_best_lap_seconds"] = df.get("LapTimeSeconds")
    df["fastf1_gap_to_pole_seconds"] = df["fastf1_best_lap_seconds"] - pole
    df["fastf1_gap_to_pole_pct"] = np.where(pole > 0, df["fastf1_gap_to_pole_seconds"] / pole * 100.0, np.nan)
    rename = {
        "Sector1Time": "fastf1_sector1_seconds",
        "Sector2Time": "fastf1_sector2_seconds",
        "Sector3Time": "fastf1_sector3_seconds",
        "Compound": "fastf1_compound",
        "Team": "fastf1_team",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [
        c
        for c in (
            "race_id", "season", "round", "driver_id", "fastf1_best_lap_seconds",
            "fastf1_gap_to_pole_seconds", "fastf1_gap_to_pole_pct",
            "fastf1_sector1_seconds", "fastf1_sector2_seconds", "fastf1_sector3_seconds",
            "fastf1_compound", "fastf1_team",
        )
        if c in df.columns
    ]
    return df[keep].drop_duplicates(subset=["race_id", "driver_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, Any]:
    """IQR-based outlier counts.  Outliers are flagged, not deleted."""
    summary: dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        summary[column] = {
            "count": int(len(series)),
            "outliers": int(((series < low) | (series > high)).sum()),
            "lower_bound": float(low),
            "upper_bound": float(high),
        }
    return summary


def missing_value_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        column: {
            "missing": int(df[column].isna().sum()),
            "missing_pct": round(float(df[column].isna().mean() * 100), 2),
            "dtype": str(df[column].dtype),
        }
        for column in df.columns
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_processed_tables(
    start: int = config.SEASON_START,
    end: int = config.SEASON_END,
    *,
    include_openf1: bool = True,
    include_fastf1: bool = True,
) -> dict[str, pd.DataFrame]:
    ensure_dirs(config.PROCESSED_DIR, config.METADATA_DIR)
    LOG.info("Building processed tables for %s-%s", start, end)
    drivers = extract_drivers()
    constructors = extract_constructors()
    circuits = extract_circuits()
    status = extract_status()
    races = extract_races(start, end)
    results = extract_results(start, end)
    qualifying = extract_qualifying(start, end)
    sprints = extract_sprints(start, end)
    driver_standings, constructor_standings = extract_standings(start, end)

    if results.empty:
        raise RuntimeError(
            "No race results found. Run scripts/collect_data.py before building datasets."
        )

    # Attach reference names where the result payload is incomplete.
    if not drivers.empty:
        results = results.merge(
            drivers[["driver_id", "driver_code", "driver_name", "date_of_birth", "nationality"]].rename(
                columns={
                    "driver_code": "driver_code_ref",
                    "driver_name": "driver_name_ref",
                    "date_of_birth": "date_of_birth_ref",
                    "nationality": "nationality_ref",
                }
            ),
            on="driver_id",
            how="left",
        )
        results["driver_code"] = results["driver_code"].fillna(results["driver_code_ref"])
        results["driver_name"] = results["driver_name"].fillna(results["driver_name_ref"])
        results["driver_dob"] = results["driver_dob"].fillna(results["date_of_birth_ref"])
        results = results.drop(columns=[c for c in ("driver_code_ref", "driver_name_ref", "date_of_birth_ref", "nationality_ref") if c in results.columns])
    if not constructors.empty:
        results = results.merge(
            constructors[["constructor_id", "constructor_name"]].rename(
                columns={"constructor_name": "constructor_name_ref"}
            ),
            on="constructor_id",
            how="left",
        )
        results["constructor_name"] = results["constructor_name"].fillna(results["constructor_name_ref"])
        results = results.drop(columns=["constructor_name_ref"])

    # Sprint points become part of the total weekend points.
    if not sprints.empty:
        sprint_points = sprints.groupby(["race_id", "driver_id"])["sprint_points"].sum()
        results = results.merge(
            sprint_points.rename("sprint_points_actual").reset_index(),
            on=["race_id", "driver_id"],
            how="left",
        )
        results["sprint_points"] = results["sprint_points_actual"].fillna(0.0)
        results = results.drop(columns=["sprint_points_actual"])
        results["total_points"] = results["race_points"].fillna(0.0) + results["sprint_points"].fillna(0.0)

    openf1_race = pd.DataFrame()
    openf1_driver = pd.DataFrame()
    if include_openf1:
        openf1_race, openf1_driver = extract_openf1(start, end, races, drivers)
        LOG.info("OpenF1 summaries: %s races, %s driver rows", len(openf1_race), len(openf1_driver))
    fastf1_quali = pd.DataFrame()
    if include_fastf1:
        fastf1_quali = extract_fastf1_qualifying(start, end, drivers)
        LOG.info("FastF1 qualifying summary rows: %s", len(fastf1_quali))

    tables = {
        "races": races,
        "results": results,
        "qualifying": qualifying,
        "sprints": sprints,
        "drivers": drivers,
        "constructors": constructors,
        "circuits": circuits,
        "status": status,
        "driver_standings": driver_standings,
        "constructor_standings": constructor_standings,
        "openf1_race_summary": openf1_race,
        "openf1_driver_race_summary": openf1_driver,
        "fastf1_qualifying_summary": fastf1_quali,
    }
    for name, df in tables.items():
        path = config.PROCESSED_TABLES.get(name, config.PROCESSED_DIR / f"{name}.parquet")
        save_parquet(df, path)

    metadata = {
        "built_at": utc_now_iso(),
        "requested_seasons": [int(start), int(end)],
        "tables": {
            name: {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "missing": missing_value_report(df),
            }
            for name, df in tables.items()
        },
        "outliers": {
            "results": detect_outliers(results, ["finish_position", "points", "laps", "grid_position"]),
            "qualifying": detect_outliers(qualifying, ["quali_best_seconds", "quali_gap_to_pole_seconds"]),
        },
        "notes": [
            "Missing values are preserved; they are not filled with future or synthetic information.",
            "finish_position is the classified position, with unclassified cars placed after the field.",
            "Actual points come from the data source; scoring-rule functions are only used for predictions.",
        ],
    }
    save_json(config.METADATA_DIR / "preprocessing_report.json", metadata)
    LOG.info("Processed tables written to %s", config.PROCESSED_DIR)
    return tables


def load_processed_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, path in config.PROCESSED_TABLES.items():
        tables[name] = load_parquet(path)
    # Optional tables that may not be in older config versions.
    for name in ("openf1_driver_race_summary",):
        path = config.PROCESSED_DIR / f"{name}.parquet"
        if name not in tables:
            tables[name] = load_parquet(path)
    return tables
