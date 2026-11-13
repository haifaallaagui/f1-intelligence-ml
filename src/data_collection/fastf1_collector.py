"""Optional FastF1 collector.

FastF1 is used where it adds information that Jolpica/OpenF1 do not expose in a
convenient form: circuit geometry, detailed qualifying pace, sector times, tyre
compounds, lap-time consistency and pit-stop data.

Session loading downloads a substantial amount of timing/telemetry data, so it
is opt-in (`--fastf1-sessions`).  Event schedules are cheap and are collected by
default when FastF1 is installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import config
from src.data_collection.base import BaseCollector
from src.utils import get_logger, save_parquet, utc_now_iso

FASTF1_CACHE_DIR = config.FASTF1_RAW_DIR / "cache"


class FastF1Collector(BaseCollector):
    source_name = "fastf1"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or config.FASTF1_RAW_DIR)
        FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fastf1():
        try:
            import fastf1

            fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
            return fastf1
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "FastF1 is not installed. Install it with `pip install fastf1` "
                "or run the pipeline without --fastf1-sessions."
            ) from exc

    # ------------------------------------------------------------------
    # Schedules (cheap)
    # ------------------------------------------------------------------
    def collect_event_schedules(
        self,
        start: int = config.FASTF1_START_SEASON,
        end: int = config.SEASON_END,
        *,
        force: bool = False,
    ) -> int:
        fastf1 = self._fastf1()
        written = 0
        for year in range(int(start), int(end) + 1):
            path = self.raw_dir / str(year) / "event_schedule.json"
            if path.exists() and not force:
                continue
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                records: list[dict[str, Any]] = []
                for _, row in schedule.iterrows():
                    record: dict[str, Any] = {}
                    for key, value in row.items():
                        if isinstance(value, (pd.Timestamp,)):
                            record[str(key)] = value.isoformat() if pd.notna(value) else None
                        elif value is None or (isinstance(value, float) and pd.isna(value)):
                            record[str(key)] = None
                        else:
                            try:
                                record[str(key)] = value.item()  # numpy scalar
                            except Exception:
                                record[str(key)] = value
                    records.append(record)
                self.write_json_payload(
                    path,
                    records,
                    endpoint="fastf1.get_event_schedule",
                    metadata={"year": year, "rows": len(records)},
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"{year}/event_schedule", exc)
                self.log.warning("FastF1 schedule %s unavailable: %s", year, exc)
        self.save_manifest()
        return written

    # ------------------------------------------------------------------
    # Session-level data (expensive, opt-in)
    # ------------------------------------------------------------------
    def collect_session(
        self,
        season: int,
        round_number: int,
        session_name: str = "Q",
        *,
        telemetry: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        fastf1 = self._fastf1()
        out_dir = self.raw_dir / str(season) / f"{int(round_number):02d}_{session_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "session.json"
        if meta_path.exists() and not force:
            return {"status": "cached", "path": str(out_dir)}

        try:
            session = fastf1.get_session(int(season), int(round_number), session_name)
            session.load(laps=True, telemetry=telemetry, weather=True, messages=False)
            laps = session.laps.copy()
            if not laps.empty:
                # Keep only columns that are useful and serialisable.
                preferred = [
                    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime",
                    "Sector1Time", "Sector2Time", "Sector3Time", "Compound",
                    "TyreLife", "FreshTyre", "Stint", "PitInTime", "PitOutTime",
                    "TrackStatus", "IsAccurate", "Position", "SpeedST",
                ]
                keep = [c for c in preferred if c in laps.columns]
                laps = laps[keep]
                for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitInTime", "PitOutTime"):
                    if col in laps.columns:
                        laps[col] = laps[col].dt.total_seconds()
                save_parquet(laps, out_dir / "laps.parquet")

            weather = getattr(session, "weather_data", None)
            if weather is not None and not weather.empty:
                save_parquet(weather.reset_index(), out_dir / "weather.parquet")

            results = getattr(session, "results", None)
            if results is not None and not results.empty:
                save_parquet(results.reset_index(), out_dir / "results.parquet")

            circuit_info = None
            try:
                ci = session.get_circuit_info()
                circuit_info = {
                    "rotation": float(getattr(ci, "rotation", 0.0) or 0.0),
                    "corners": ci.corners.to_dict(orient="records") if hasattr(ci, "corners") else [],
                    "marshal_sectors": ci.marshal_sectors.to_dict(orient="records")
                    if hasattr(ci, "marshal_sectors")
                    else [],
                }
            except Exception:
                circuit_info = None

            self.write_json_payload(
                meta_path,
                {
                    "season": season,
                    "round": round_number,
                    "session_name": session_name,
                    "event": str(getattr(session, "event", "")),
                    "circuit_info": circuit_info,
                    "has_laps": not laps.empty,
                    "collected_at": utc_now_iso(),
                },
                endpoint="fastf1.session",
                metadata={"season": season, "round": round_number, "session": session_name},
            )
            return {"status": "collected", "path": str(out_dir)}
        except Exception as exc:  # noqa: BLE001
            self.record_error(f"{season}/{round_number}/{session_name}", exc)
            self.log.warning(
                "FastF1 session %s round %s %s unavailable: %s",
                season,
                round_number,
                session_name,
                exc,
            )
            return {"status": "error", "error": str(exc)}

    def collect_qualifying_pace(
        self,
        seasons: Iterable[int],
        *,
        rounds: Iterable[int] | None = None,
        max_events: int | None = None,
        force: bool = False,
    ) -> int:
        """Load qualifying sessions and persist a compact pace summary."""
        fastf1 = self._fastf1()
        written = 0
        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(int(season), include_testing=False)
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"{season}/schedule_for_quali", exc)
                continue
            allowed_rounds = set(int(r) for r in rounds) if rounds else None
            events = schedule
            if allowed_rounds is not None:
                events = schedule[schedule["RoundNumber"].astype(int).isin(allowed_rounds)]
            if max_events is not None:
                events = events.head(int(max_events))
            for _, event in events.iterrows():
                round_number = int(event["RoundNumber"])
                out_dir = self.raw_dir / str(season) / f"{round_number:02d}_Q"
                out_path = out_dir / "qualifying_summary.parquet"
                if out_path.exists() and not force:
                    continue
                result = self.collect_session(season, round_number, "Q", telemetry=False, force=force)
                if result.get("status") == "error":
                    continue
                try:
                    session = fastf1.get_session(int(season), round_number, "Q")
                    session.load(laps=True, telemetry=False, weather=False, messages=False)
                    laps = session.laps.copy()
                    if laps.empty:
                        continue
                    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
                    best = (
                        laps.dropna(subset=["LapTimeSeconds"])
                        .sort_values("LapTimeSeconds")
                        .groupby("Driver", as_index=False)
                        .first()
                    )
                    keep = [
                        c
                        for c in (
                            "Driver", "DriverNumber", "Team", "LapTimeSeconds",
                            "Sector1Time", "Sector2Time", "Sector3Time", "Compound",
                        )
                        if c in best.columns
                    ]
                    best = best[keep]
                    for col in ("Sector1Time", "Sector2Time", "Sector3Time"):
                        if col in best.columns:
                            best[col] = best[col].dt.total_seconds()
                    best["season"] = int(season)
                    best["round"] = round_number
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_parquet(best, out_path)
                    self.record_file(
                        out_path,
                        {
                            "source": self.source_name,
                            "endpoint": "fastf1.qualifying_summary",
                            "season": int(season),
                            "round": round_number,
                            "rows": len(best),
                        },
                    )
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    self.record_error(f"{season}/{round_number}/Q_summary", exc)
        self.save_manifest()
        return written
