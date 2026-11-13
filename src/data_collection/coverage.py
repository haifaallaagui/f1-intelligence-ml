"""Data coverage reporting by source and season."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.utils import load_json, save_json, utc_now_iso, ensure_dirs


def _race_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    return (payload.get("RaceTable", {}) or {}).get("Races", []) or []


def jolpica_coverage(start: int = config.SEASON_START, end: int = config.SEASON_END) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(int(start), int(end) + 1):
        season_dir = config.JOLPICA_RAW_DIR / str(season)
        schedule = load_json(season_dir / "schedule.json", default={}) or {}
        results = load_json(season_dir / "results.json", default={}) or {}
        qualifying = load_json(season_dir / "qualifying.json", default={}) or {}
        sprint = load_json(season_dir / "sprint.json", default={}) or {}
        schedule_data = schedule.get("data", schedule)
        results_data = results.get("data", results)
        qualifying_data = qualifying.get("data", qualifying)
        sprint_data = sprint.get("data", sprint)
        sched_races = _race_rows(schedule_data)
        result_races = _race_rows(results_data)
        quali_races = _race_rows(qualifying_data)
        sprint_races = _race_rows(sprint_data)
        rows.append(
            {
                "season": season,
                "races_scheduled": len(sched_races),
                "races_with_results": sum(1 for r in result_races if r.get("Results")),
                "result_rows": sum(len(r.get("Results", []) or []) for r in result_races),
                "races_with_qualifying": sum(1 for r in quali_races if r.get("QualifyingResults")),
                "qualifying_rows": sum(len(r.get("QualifyingResults", []) or []) for r in quali_races),
                "races_with_sprint": sum(1 for r in sprint_races if r.get("SprintResults")),
                "sprint_rows": sum(len(r.get("SprintResults", []) or []) for r in sprint_races),
                "jolpica_collected_at": (schedule or {}).get("collected_at"),
            }
        )
    return pd.DataFrame(rows)


def openf1_coverage(start: int = config.OPENF1_START_SEASON, end: int = config.SEASON_END) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(int(start), int(end) + 1):
        year_dir = config.OPENF1_RAW_DIR / str(year)
        sessions_payload = load_json(year_dir / "sessions.json", default={}) or {}
        sessions = sessions_payload.get("data", sessions_payload) or []
        race_sessions = [
            s for s in sessions if str(s.get("session_name", "")) == "Race" and not s.get("is_cancelled")
        ]
        endpoint_counts: dict[str, int] = {}
        for session in race_sessions:
            session_dir = year_dir / str(session.get("session_key"))
            if not session_dir.exists():
                continue
            for path in session_dir.glob("*.json"):
                endpoint_counts[path.stem] = endpoint_counts.get(path.stem, 0) + 1
        rows.append(
            {
                "season": year,
                "openf1_race_sessions": len(race_sessions),
                "openf1_sessions_with_data": sum(
                    1
                    for s in race_sessions
                    if (year_dir / str(s.get("session_key"))).exists()
                ),
                **{f"openf1_{k}_sessions": v for k, v in sorted(endpoint_counts.items())},
            }
        )
    return pd.DataFrame(rows)


def fastf1_coverage(start: int = config.FASTF1_START_SEASON, end: int = config.SEASON_END) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(int(start), int(end) + 1):
        year_dir = config.FASTF1_RAW_DIR / str(year)
        schedule = load_json(year_dir / "event_schedule.json", default={}) or {}
        events = schedule.get("data", schedule) or []
        session_dirs = [p for p in year_dir.glob("*_*") if p.is_dir() and p.name != "cache"]
        rows.append(
            {
                "season": year,
                "fastf1_events": len(events),
                "fastf1_session_dirs": len(session_dirs),
                "fastf1_qualifying_summaries": len(list(year_dir.glob("*_Q/qualifying_summary.parquet"))),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_report(
    start: int = config.SEASON_START,
    end: int = config.SEASON_END,
    *,
    write: bool = True,
) -> dict[str, Any]:
    ensure_dirs(config.REPORTS_DIR, config.METADATA_DIR)
    jolpica = jolpica_coverage(start, end)
    openf1 = openf1_coverage(max(start, config.OPENF1_START_SEASON), end)
    fastf1 = fastf1_coverage(max(start, config.FASTF1_START_SEASON), end)
    if openf1.empty:
        openf1 = pd.DataFrame({"season": pd.Series(dtype="int64")})
    if fastf1.empty:
        fastf1 = pd.DataFrame({"season": pd.Series(dtype="int64")})
    merged = jolpica.merge(openf1, on="season", how="outer").merge(fastf1, on="season", how="outer")
    merged = merged.sort_values("season").reset_index(drop=True)
    payload = {
        "generated_at": utc_now_iso(),
        "requested_seasons": [int(start), int(end)],
        "coverage": merged.to_dict(orient="records"),
        "notes": {
            "jolpica": "Ergast-compatible results/qualifying/standings; missing endpoints are recorded as unavailable.",
            "openf1": "Session data available from 2023 onwards; race-session endpoints listed per season.",
            "fastf1": "Optional; event schedules are cheap, detailed sessions are opt-in.",
        },
    }
    if write:
        save_json(config.REPORTS_DIR / "data_coverage.json", payload)
        save_json(config.METADATA_DIR / "coverage.json", payload)
        (config.REPORTS_DIR / "data_coverage.md").write_text(
            render_coverage_markdown(merged, payload), encoding="utf-8"
        )
    return payload


def render_coverage_markdown(df: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Data coverage by source and season",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "| Season | Scheduled | Results | Result rows | Qualifying | Quali rows | Sprint | Sprint rows | OpenF1 races | FastF1 events |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        def val(key: str) -> str:
            value = row.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return "—"
            return str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(value)

        lines.append(
            "| {season} | {sched} | {res} | {rrows} | {qual} | {qrows} | {spr} | {srows} | {of1} | {ff1} |".format(
                season=val("season"),
                sched=val("races_scheduled"),
                res=val("races_with_results"),
                rrows=val("result_rows"),
                qual=val("races_with_qualifying"),
                qrows=val("qualifying_rows"),
                spr=val("races_with_sprint"),
                srows=val("sprint_rows"),
                of1=val("openf1_race_sessions"),
                ff1=val("fastf1_events"),
            )
        )
    lines += [
        "",
        "Notes:",
        "- Jolpica is the primary historical source; absent seasons/endpoints are never filled with synthetic data.",
        "- OpenF1 coverage starts in 2023. The collector records which session endpoints are available.",
        "- FastF1 session loading is optional because it downloads large timing/telemetry datasets.",
        "- Known Jolpica qualifying gaps: 2001-2002 expose no qualifying; 1996-2000 are partial. Results, grids and standings remain complete from 1995 onward.",
        "- The 2026 calendar lists 23 rounds; only completed races appear in the results columns. Unraced rounds are not fabricated.",
    ]
    return "\n".join(lines)
