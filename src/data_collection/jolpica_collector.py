"""Jolpica-F1 (Ergast-compatible) collector.

Coverage notes
--------------
* Race results / qualifying / standings are available from 1950 onwards in the
  Ergast schema.  This project requests 1995-2026 and stores exactly what the
  API returns; missing seasons are recorded as unavailable rather than filled.
* Sprint results exist only for sprint weekends (2021+ in this dataset).
* Reference tables (drivers, constructors, circuits, status) are global.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import config
from src.data_collection.base import BaseCollector
from src.utils import HttpClient, get_logger, load_json, utc_now_iso


@dataclass
class CollectSummary:
    source: str
    seasons_requested: list[int]
    seasons_with_races: list[int]
    files_written: int
    errors: int
    latest_race: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "seasons_requested": self.seasons_requested,
            "seasons_with_races": self.seasons_with_races,
            "files_written": self.files_written,
            "errors": self.errors,
            "latest_race": self.latest_race,
        }


class JolpicaCollector(BaseCollector):
    source_name = "jolpica"
    PAGE_SIZE = 100
    SEASON_ENDPOINTS = (
        ("schedule", ""),
        ("results", "results"),
        ("qualifying", "qualifying"),
        ("driverstandings", "driverstandings"),
        ("constructorstandings", "constructorstandings"),
    )
    REFERENCE_ENDPOINTS = {
        "drivers": "drivers",
        "constructors": "constructors",
        "circuits": "circuits",
        "status": "status",
    }

    def __init__(self, raw_dir: Path | None = None, http: HttpClient | None = None):
        super().__init__(raw_dir or config.JOLPICA_RAW_DIR, http=http)
        if self.http.base_url in ("", None):
            self.http.base_url = config.JOLPICA_BASE_URL

    # ------------------------------------------------------------------
    # Pagination / merging
    # ------------------------------------------------------------------
    def _fetch_mrdata(self, path: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        offset = 0
        total = None
        while True:
            query = dict(params or {})
            query.update({"limit": self.PAGE_SIZE, "offset": offset})
            payload = self.http.get_json(path, params=query)
            mrdata = (payload or {}).get("MRData", {})
            if not isinstance(mrdata, dict):
                raise ValueError(f"Unexpected Jolpica payload for {path}")
            pages.append(mrdata)
            if total is None:
                total = int(mrdata.get("total", 0) or 0)
            offset += self.PAGE_SIZE
            if offset >= total or not mrdata:
                break
        return pages

    @staticmethod
    def _race_key(race: Mapping[str, Any]) -> tuple[int, int]:
        return int(race.get("season", 0) or 0), int(race.get("round", 0) or 0)

    def _merge_races(
        self,
        pages: Iterable[dict[str, Any]],
        result_key: str | None = None,
        id_key: str | None = None,
    ) -> dict[str, Any]:
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        first: dict[str, Any] | None = None
        total = 0
        for page in pages:
            first = first or page
            table = page.get("RaceTable", {})
            for race in table.get("Races", []) or []:
                key = self._race_key(race)
                target = merged.setdefault(key, dict(race))
                if result_key:
                    existing = target.get(result_key, []) or []
                    seen = {
                        (item.get(id_key) if id_key else item.get("number"))
                        for item in existing
                    }
                    for item in race.get(result_key, []) or []:
                        ident = item.get(id_key) if id_key else item.get("number")
                        if ident in seen:
                            # Merge partial fields rather than duplicating.
                            for current in existing:
                                if (current.get(id_key) if id_key else current.get("number")) == ident:
                                    current.update({k: v for k, v in item.items() if v not in (None, "", [])})
                                    break
                        else:
                            existing.append(item)
                            seen.add(ident)
                    target[result_key] = existing
                else:
                    target.update({k: v for k, v in race.items() if v not in (None, "", [])})
                total += 1
        races = [merged[key] for key in sorted(merged)]
        base = dict(first or {})
        base["RaceTable"] = {"Races": races}
        base["total"] = str(len(races))
        return base

    def _fetch_race_endpoint(
        self,
        season: int,
        endpoint: str,
        result_key: str,
        rounds: list[int],
    ) -> tuple[dict[str, Any], int]:
        """Fetch a race-level endpoint without losing races split by pagination.

        Jolpica paginates *result rows*, but every page restarts at the first
        result of a race.  A race that straddles a page boundary is therefore
        returned truncated on both pages and can never be completed by
        ``offset`` pagination alone (e.g. 1 finisher instead of 22).  When the
        merged season contains fewer rows than the advertised total we
        re-fetch each round individually, which is exact.
        """
        pages = self._fetch_mrdata(f"/{season}/{endpoint}")
        merged = self._merge_races(pages, result_key, "driverId")
        expected = int(pages[0].get("total", 0) or 0) if pages else 0
        rows = sum(
            len(race.get(result_key, []) or [])
            for race in merged["RaceTable"]["Races"]
        )
        if rounds and rows < expected:
            per_round: list[dict[str, Any]] = []
            for round_number in rounds:
                per_round.extend(
                    self._fetch_mrdata(f"/{season}/{round_number}/{endpoint}")
                )
            merged = self._merge_races(per_round, result_key, "driverId")
            pages = per_round
        return merged, len(pages)

    def _merge_standings(self, pages: Iterable[dict[str, Any]], key: str, id_key: str) -> dict[str, Any]:
        lists: dict[tuple, dict[str, Any]] = {}
        first = None
        for page in pages:
            first = first or page
            table = page.get("StandingsTable", {})
            for standings_list in table.get("StandingsLists", []) or []:
                key_tuple = (
                    standings_list.get("season"),
                    standings_list.get("round"),
                )
                target = lists.setdefault(key_tuple, dict(standings_list))
                existing = target.get(key, []) or []
                seen = {item.get(id_key) for item in existing}
                for item in standings_list.get(key, []) or []:
                    if item.get(id_key) not in seen:
                        existing.append(item)
                        seen.add(item.get(id_key))
                target[key] = existing
        base = dict(first or {})
        base["StandingsTable"] = {"StandingsLists": [lists[k] for k in sorted(lists)]}
        return base

    def _merge_reference(self, pages: Iterable[dict[str, Any]], table: str, key: str, id_key: str) -> dict[str, Any]:
        items: dict[str, dict[str, Any]] = {}
        first = None
        for page in pages:
            first = first or page
            for item in (page.get(table, {}) or {}).get(key, []) or []:
                items[item.get(id_key)] = item
        base = dict(first or {})
        base[table] = {key: list(items.values())}
        base["total"] = str(len(items))
        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(
        self,
        start: int = config.SEASON_START,
        end: int = config.SEASON_END,
        *,
        update: bool = False,
        force: bool = False,
        include_reference: bool = True,
        seasons: Iterable[int] | None = None,
    ) -> CollectSummary:
        seasons = list(seasons) if seasons is not None else list(range(int(start), int(end) + 1))
        written = 0
        errors_before = len(self.manifest.get("errors", []))
        seasons_with_races: list[int] = []
        latest_race: str | None = None

        for season in seasons:
            season_dir = self.raw_dir / str(season)
            season_dir.mkdir(parents=True, exist_ok=True)
            is_recent = season >= max(config.SEASON_START, 2024)
            refresh = force or (update and is_recent) or not (season_dir / "schedule.json").exists()

            endpoints = list(self.SEASON_ENDPOINTS)
            if season >= 2021:
                endpoints.append(("sprint", "sprint"))

            cached_schedule = self.read_json_payload(season_dir / "schedule.json") or {}
            season_rounds = [
                int(race["round"])
                for race in (cached_schedule.get("RaceTable", {}) or {}).get("Races", []) or []
                if race.get("round") is not None
            ]

            for name, endpoint in endpoints:
                path = season_dir / f"{name}.json"
                if path.exists() and not refresh:
                    data = self.read_json_payload(path)
                    if name == "schedule" and data:
                        races = (data.get("RaceTable", {}) or {}).get("Races", [])
                        if races:
                            seasons_with_races.append(season)
                    continue
                url_path = f"/{season}" if endpoint == "" else f"/{season}/{endpoint}"
                try:
                    page_count = 0
                    if name == "schedule":
                        pages = self._fetch_mrdata(url_path)
                        merged = self._merge_races(pages)
                        page_count = len(pages)
                        season_rounds = [
                            int(race["round"])
                            for race in (merged.get("RaceTable", {}) or {}).get("Races", []) or []
                            if race.get("round") is not None
                        ]
                    elif name in {"results", "qualifying", "sprint"}:
                        result_key = {
                            "results": "Results",
                            "qualifying": "QualifyingResults",
                            "sprint": "SprintResults",
                        }[name]
                        merged, page_count = self._fetch_race_endpoint(
                            season, name, result_key, season_rounds
                        )
                    else:
                        pages = self._fetch_mrdata(url_path)
                        page_count = len(pages)
                        if name == "driverstandings":
                            merged = self._merge_standings(pages, "DriverStandings", "driverId")
                        elif name == "constructorstandings":
                            merged = self._merge_standings(pages, "ConstructorStandings", "constructorId")
                        else:
                            merged = dict(pages[0]) if pages else {}
                    self.write_json_payload(
                        path,
                        merged,
                        endpoint=url_path,
                        metadata={
                            "season": season,
                            "endpoint": name,
                            "total": merged.get("total", 0),
                            "pages": page_count,
                            "available": bool(merged.get("RaceTable", {}).get("Races"))
                            if name in {"schedule", "results", "qualifying", "sprint"}
                            else bool(merged),
                        },
                    )
                    written += 1
                    if name == "schedule":
                        races = (merged.get("RaceTable", {}) or {}).get("Races", [])
                        if races:
                            seasons_with_races.append(season)
                            last = races[-1]
                            latest_race = f"{last.get('season')}_{int(last.get('round', 0)):02d}"
                except Exception as exc:  # noqa: BLE001 - record and continue
                    self.record_error(f"{season}/{name}", exc)
                    self.log.warning("Jolpica %s/%s unavailable: %s", season, name, exc)
                    # Persist an explicit empty result so downstream code can
                    # distinguish "not available" from "never attempted".
                    if not path.exists() or force:
                        empty = {"RaceTable": {"Races": []}, "total": "0"}
                        self.write_json_payload(
                            path,
                            empty,
                            endpoint=url_path,
                            metadata={"season": season, "endpoint": name, "available": False},
                        )
            self.save_manifest()
            self.log.info("Jolpica season %s processed", season)

        if include_reference:
            written += self.collect_reference(force=force)

        self.save_manifest()
        errors = len(self.manifest.get("errors", [])) - errors_before
        return CollectSummary(
            source=self.source_name,
            seasons_requested=seasons,
            seasons_with_races=sorted(set(seasons_with_races)),
            files_written=written,
            errors=max(0, errors),
            latest_race=latest_race,
        )

    def collect_reference(self, *, force: bool = False) -> int:
        ref_dir = self.raw_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for name, endpoint in self.REFERENCE_ENDPOINTS.items():
            path = ref_dir / f"{name}.json"
            if path.exists() and not force:
                continue
            try:
                pages = self._fetch_mrdata(f"/{endpoint}")
                if name == "drivers":
                    merged = self._merge_reference(pages, "DriverTable", "Drivers", "driverId")
                elif name == "constructors":
                    merged = self._merge_reference(pages, "ConstructorTable", "Constructors", "constructorId")
                elif name == "circuits":
                    merged = self._merge_reference(pages, "CircuitTable", "Circuits", "circuitId")
                else:
                    merged = self._merge_reference(pages, "StatusTable", "Status", "statusId")
                self.write_json_payload(path, merged, endpoint=f"/{endpoint}", metadata={"endpoint": name})
                written += 1
            except Exception as exc:  # noqa: BLE001
                self.record_error(f"reference/{name}", exc)
                self.log.warning("Jolpica reference %s unavailable: %s", name, exc)
        self.save_manifest()
        return written

    # ------------------------------------------------------------------
    # Update detection
    # ------------------------------------------------------------------
    def detect_updates(self, season: int | None = None) -> dict[str, Any]:
        season = int(season or config.SEASON_END)
        path = self.raw_dir / str(season) / "schedule.json"
        stored = self.read_json_payload(path) or {}
        stored_races = (stored.get("RaceTable", {}) or {}).get("Races", []) or []
        stored_rounds = {int(r.get("round", 0)) for r in stored_races}
        try:
            pages = self._fetch_mrdata(f"/{season}")
            current = self._merge_races(pages)
            current_races = (current.get("RaceTable", {}) or {}).get("Races", []) or []
            current_rounds = {int(r.get("round", 0)) for r in current_races}
            # A race is considered complete when the results endpoint has rows.
            results_path = self.raw_dir / str(season) / "results.json"
            results = self.read_json_payload(results_path) or {}
            result_rounds = {
                int(r.get("round", 0))
                for r in (results.get("RaceTable", {}) or {}).get("Races", []) or []
                if r.get("Results")
            }
            new_rounds = sorted(current_rounds - result_rounds)
            changed = bool(current_rounds - stored_rounds) or bool(new_rounds)
            return {
                "season": season,
                "changed": changed,
                "new_schedule_rounds": sorted(current_rounds - stored_rounds),
                "rounds_without_results": new_rounds,
                "stored_rounds": sorted(stored_rounds),
                "current_rounds": sorted(current_rounds),
            }
        except Exception as exc:  # noqa: BLE001
            return {"season": season, "changed": False, "error": str(exc)}
