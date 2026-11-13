"""OpenF1 collector.

OpenF1 provides recent, session-level data (2023+): weather, laps, stints,
pit stops, positions, intervals, race control messages, drivers and telemetry.
Telemetry (`car_data`/`location`) is intentionally opt-in because it is very
large; the core ML features only need aggregated race/session data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import config
import requests
from src.data_collection.base import BaseCollector
from src.utils import HttpClient, get_logger, load_json, save_json, utc_now_iso


DEFAULT_ENDPOINTS = (
    "weather",
    "drivers",
    "position",
    "intervals",
    "pit",
    "stints",
    "laps",
    "race_control",
)
TELEMETRY_ENDPOINTS = ("car_data", "location")


@dataclass
class OpenF1Summary:
    seasons: list[int]
    sessions_collected: int
    files_written: int
    errors: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": "openf1",
            "seasons": self.seasons,
            "sessions_collected": self.sessions_collected,
            "files_written": self.files_written,
            "errors": self.errors,
        }


class OpenF1Collector(BaseCollector):
    source_name = "openf1"

    def __init__(self, raw_dir: Path | None = None, http: HttpClient | None = None):
        super().__init__(raw_dir or config.OPENF1_RAW_DIR, http=http)
        if self.http.base_url in ("", None):
            self.http.base_url = config.OPENF1_BASE_URL

    def _get_list(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            payload = self.http.get_json(f"/{endpoint}", params=params or {})
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                # OpenF1 returns 404 for endpoints that were not recorded for
                # a given (often older) session.  That is "no coverage", not a
                # failure, so treat it as an empty result.
                return []
            raise
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "detail" in payload:
            raise ValueError(f"OpenF1 error for {endpoint}: {payload['detail']}")
        return []

    def collect(
        self,
        start: int = config.OPENF1_START_SEASON,
        end: int = config.SEASON_END,
        *,
        update: bool = False,
        force: bool = False,
        session_names: Iterable[str] = ("Race",),
        endpoints: Iterable[str] = DEFAULT_ENDPOINTS,
        include_circuit_info: bool = True,
        max_sessions: int | None = None,
    ) -> OpenF1Summary:
        seasons = list(range(int(start), int(end) + 1))
        written = 0
        errors_before = len(self.manifest.get("errors", []))
        sessions_collected = 0
        session_names = tuple(session_names)
        endpoints = tuple(endpoints)

        for year in seasons:
            year_dir = self.raw_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            try:
                meetings = self._get_list("meetings", {"year": year})
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"{year}/meetings", exc)
                self.log.warning("OpenF1 meetings %s unavailable: %s", year, exc)
                continue
            meetings_path = year_dir / "meetings.json"
            if force or update or not meetings_path.exists():
                self.write_json_payload(
                    meetings_path,
                    meetings,
                    endpoint="/meetings",
                    metadata={"year": year, "rows": len(meetings)},
                )
                written += 1

            try:
                sessions = self._get_list("sessions", {"year": year})
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"{year}/sessions", exc)
                self.log.warning("OpenF1 sessions %s unavailable: %s", year, exc)
                continue
            sessions_path = year_dir / "sessions.json"
            if force or update or not sessions_path.exists():
                self.write_json_payload(
                    sessions_path,
                    sessions,
                    endpoint="/sessions",
                    metadata={"year": year, "rows": len(sessions)},
                )
                written += 1

            if include_circuit_info:
                written += self._collect_circuit_info(year, meetings, force=force)

            selected = [
                s
                for s in sessions
                if str(s.get("session_name", "")) in session_names
                and not s.get("is_cancelled", False)
            ]
            if max_sessions is not None:
                selected = selected[: int(max_sessions)]
            for session in selected:
                session_key = session.get("session_key")
                if session_key is None:
                    continue
                session_dir = year_dir / str(session_key)
                session_dir.mkdir(parents=True, exist_ok=True)
                for endpoint in endpoints:
                    path = session_dir / f"{endpoint}.json"
                    # ``update`` refreshes the meeting/session lists so new
                    # sessions are discovered, but it must not re-download
                    # session payloads that are already on disk.
                    if path.exists() and not force:
                        continue
                    try:
                        rows = self._get_list(endpoint, {"session_key": session_key})
                        self.write_json_payload(
                            path,
                            rows,
                            endpoint=f"/{endpoint}",
                            metadata={
                                "year": year,
                                "session_key": session_key,
                                "session_name": session.get("session_name"),
                                "endpoint": endpoint,
                                "rows": len(rows),
                                "circuit_short_name": session.get("circuit_short_name"),
                                "date_start": session.get("date_start"),
                            },
                        )
                        written += 1
                    except Exception as exc:  # noqa: BLE001
                        self.record_error(f"{year}/{session_key}/{endpoint}", exc)
                        self.log.warning(
                            "OpenF1 %s/%s/%s unavailable: %s", year, session_key, endpoint, exc
                        )
                sessions_collected += 1
            self.save_manifest()
            self.log.info("OpenF1 year %s processed", year)

        self.save_manifest()
        errors = len(self.manifest.get("errors", [])) - errors_before
        return OpenF1Summary(
            seasons=seasons,
            sessions_collected=sessions_collected,
            files_written=written,
            errors=max(0, errors),
        )

    def _collect_circuit_info(self, year: int, meetings: list[dict[str, Any]], *, force: bool = False) -> int:
        written = 0
        info_dir = self.raw_dir / str(year) / "circuit_info"
        info_dir.mkdir(parents=True, exist_ok=True)
        seen: set[tuple[Any, Any]] = set()
        for meeting in meetings:
            key = (meeting.get("circuit_key"), year)
            if key in seen or meeting.get("circuit_key") is None:
                continue
            seen.add(key)
            path = info_dir / f"{meeting.get('circuit_key')}.json"
            if path.exists() and not force:
                continue
            url = meeting.get("circuit_info_url")
            if not url:
                continue
            try:
                payload = self.http.get_json(url)
                self.write_json_payload(
                    path,
                    payload,
                    endpoint=url,
                    metadata={
                        "year": year,
                        "circuit_key": meeting.get("circuit_key"),
                        "circuit_short_name": meeting.get("circuit_short_name"),
                        "source": "openf1/multiviewer",
                    },
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"{year}/circuit_info/{key[0]}", exc)
        return written
