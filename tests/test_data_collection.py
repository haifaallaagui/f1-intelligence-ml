"""Collector behaviour with fully mocked external services.

No test performs a network request.  Fake sessions return the same payload
shapes as Jolpica/OpenF1, and FastF1 is replaced by a tiny in-memory module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.data_collection.base import BaseCollector
from src.data_collection.fastf1_collector import FastF1Collector
from src.data_collection.jolpica_collector import JolpicaCollector
from src.data_collection.openf1_collector import OpenF1Collector
from src.utils import HttpClient, RateLimiter


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self
            )


class FakeSession:
    """Minimal ``requests.Session`` stand-in that routes by URL suffix."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        return self.handler(url, params)


def _http(session: FakeSession, base_url: str = "", max_retries: int = 2) -> HttpClient:
    return HttpClient(
        base_url=base_url,
        session=session,
        max_retries=max_retries,
        rate_limiter=RateLimiter(0.0),
    )


def _jolpica_handler(total_races: int = 150):
    def handler(url: str, params: dict) -> FakeResponse:
        path = url.split("/ergast/f1")[-1]
        if path == "/2016":
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 100))
            races = [
                {
                    "season": "2016",
                    "round": str(index + 1),
                    "raceName": f"Test Grand Prix {index + 1}",
                    "Circuit": {"circuitId": f"circuit_{index + 1}"},
                }
                for index in range(total_races)
            ]
            return FakeResponse(
                200,
                {
                    "MRData": {
                        "total": str(total_races),
                        "RaceTable": {"Races": races[offset : offset + limit]},
                    }
                },
            )
        return FakeResponse(
            200, {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
        )

    return handler


# ---------------------------------------------------------------------------
# Jolpica
# ---------------------------------------------------------------------------
def test_jolpica_collector_paginates_merges_and_writes_manifest(tmp_path):
    session = FakeSession(_jolpica_handler(total_races=150))
    collector = JolpicaCollector(raw_dir=tmp_path, http=_http(session))

    summary = collector.collect(seasons=[2016], include_reference=False)

    schedule = collector.read_json_payload(tmp_path / "2016" / "schedule.json")
    races = schedule["RaceTable"]["Races"]
    assert len(races) == 150
    assert [int(race["round"]) for race in races] == list(range(1, 151))
    assert summary.seasons_with_races == [2016]
    assert summary.errors == 0
    assert summary.latest_race == "2016_150"
    # Two pages were requested for the schedule (100 + 50).
    schedule_calls = [
        params
        for url, params in session.calls
        if url.endswith("/2016") and params.get("offset") in (0, 100)
    ]
    assert len(schedule_calls) == 2

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "jolpica"
    assert any(key.endswith("2016/schedule.json") for key in manifest["files"])


def test_jolpica_collector_skips_existing_season_files(tmp_path):
    session = FakeSession(_jolpica_handler(total_races=2))
    collector = JolpicaCollector(raw_dir=tmp_path, http=_http(session))
    season_dir = tmp_path / "2016"
    season_dir.mkdir(parents=True)
    existing = {
        "source": "jolpica",
        "data": {
            "RaceTable": {
                "Races": [{"season": "2016", "round": "1", "raceName": "Cached"}]
            }
        },
    }
    (season_dir / "schedule.json").write_text(json.dumps(existing), encoding="utf-8")

    collector.collect(seasons=[2016], include_reference=False)

    schedule = collector.read_json_payload(season_dir / "schedule.json")
    assert schedule["RaceTable"]["Races"][0]["raceName"] == "Cached"
    assert not any(url.endswith("/2016") and params.get("offset") == 0 for url, params in session.calls)


def test_jolpica_detect_updates_finds_new_rounds(tmp_path):
    session = FakeSession(_jolpica_handler(total_races=5))
    collector = JolpicaCollector(raw_dir=tmp_path, http=_http(session))
    season_dir = tmp_path / "2016"
    season_dir.mkdir(parents=True)
    (season_dir / "schedule.json").write_text(
        json.dumps(
            {
                "data": {
                    "RaceTable": {
                        "Races": [
                            {"season": "2016", "round": "1"},
                            {"season": "2016", "round": "2"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (season_dir / "results.json").write_text(
        json.dumps(
            {
                "data": {
                    "RaceTable": {
                        "Races": [
                            {"season": "2016", "round": "1", "Results": [{"driverId": "a"}]},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    info = collector.detect_updates(2016)

    assert info["changed"] is True
    assert info["new_schedule_rounds"] == [3, 4, 5]
    assert info["rounds_without_results"] == [2, 3, 4, 5]


def test_base_collector_payload_roundtrip(tmp_path):
    collector = BaseCollector(raw_dir=tmp_path)
    path = tmp_path / "nested" / "payload.json"
    collector.write_json_payload(
        path, [{"a": 1}], endpoint="/test", metadata={"rows": 1}
    )
    assert collector.read_json_payload(path) == [{"a": 1}]
    assert collector.file_metadata(path)["endpoint"] == "/test"
    collector.record_error("boom", ValueError("bad"))
    assert collector.manifest["errors"][-1]["context"] == "boom"
    collector.save_manifest()
    assert (tmp_path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# OpenF1
# ---------------------------------------------------------------------------
def _openf1_handler(missing_endpoint: str = "pit"):
    def handler(url: str, params: dict) -> FakeResponse:
        endpoint = url.rstrip("/").split("/")[-1]
        if endpoint == missing_endpoint:
            return FakeResponse(404, {"detail": "Not Found"})
        if endpoint == "meetings":
            return FakeResponse(
                200,
                [
                    {
                        "meeting_key": 1,
                        "circuit_key": 10,
                        "circuit_short_name": "Test",
                        "circuit_info_url": None,
                    }
                ],
            )
        if endpoint == "sessions":
            return FakeResponse(
                200,
                [
                    {
                        "session_key": 111,
                        "session_name": "Race",
                        "circuit_short_name": "Test",
                        "date_start": "2024-03-02T15:00:00",
                    },
                    {
                        "session_key": 112,
                        "session_name": "Qualifying",
                        "circuit_short_name": "Test",
                    },
                ],
            )
        if endpoint == "weather":
            return FakeResponse(200, [{"air_temperature": 24.5, "rainfall": 0}])
        return FakeResponse(200, [])

    return handler


def test_openf1_collector_selects_race_sessions_and_treats_404_as_empty(tmp_path):
    session = FakeSession(_openf1_handler())
    collector = OpenF1Collector(raw_dir=tmp_path, http=_http(session))

    summary = collector.collect(
        start=2024,
        end=2024,
        endpoints=("weather", "pit"),
        max_sessions=1,
        include_circuit_info=False,
    )

    weather = collector.read_json_payload(tmp_path / "2024" / "111" / "weather.json")
    pit = collector.read_json_payload(tmp_path / "2024" / "111" / "pit.json")
    assert weather == [{"air_temperature": 24.5, "rainfall": 0}]
    assert pit == []
    assert not (tmp_path / "2024" / "112").exists()
    assert summary.sessions_collected == 1
    assert summary.errors == 0
    # meetings, sessions, weather and the empty pit payload were written.
    assert summary.files_written == 4


def test_openf1_collector_records_transient_errors(tmp_path):
    def handler(url: str, params: dict) -> FakeResponse:
        endpoint = url.rstrip("/").split("/")[-1]
        if endpoint == "meetings":
            return FakeResponse(200, [])
        if endpoint == "sessions":
            return FakeResponse(
                200,
                [{"session_key": 5, "session_name": "Race", "circuit_short_name": "X"}],
            )
        return FakeResponse(503, {"detail": "unavailable"})

    session = FakeSession(handler)
    collector = OpenF1Collector(raw_dir=tmp_path, http=_http(session, max_retries=1))

    summary = collector.collect(
        start=2024,
        end=2024,
        endpoints=("weather",),
        include_circuit_info=False,
    )

    assert summary.errors >= 1
    assert not (tmp_path / "2024" / "5" / "weather.json").exists()
    assert any("weather" in entry["context"] for entry in collector.manifest["errors"])


# ---------------------------------------------------------------------------
# FastF1 (mocked module)
# ---------------------------------------------------------------------------
class _FakeCache:
    @staticmethod
    def enable_cache(path):
        return None


class _FakeFastF1:
    Cache = _FakeCache

    @staticmethod
    def get_event_schedule(year, include_testing=False):
        return pd.DataFrame(
            [
                {
                    "RoundNumber": 1,
                    "EventName": "Mock Grand Prix",
                    "EventDate": pd.Timestamp("2024-03-03"),
                    "Location": "Mock City",
                }
            ]
        )


def test_fastf1_collector_writes_serialisable_schedule(tmp_path, monkeypatch):
    collector = FastF1Collector(raw_dir=tmp_path)
    monkeypatch.setattr(
        FastF1Collector, "_fastf1", staticmethod(lambda: _FakeFastF1)
    )

    written = collector.collect_event_schedules(start=2024, end=2024)

    assert written == 1
    payload = collector.read_json_payload(tmp_path / "2024" / "event_schedule.json")
    assert payload[0]["EventName"] == "Mock Grand Prix"
    assert payload[0]["EventDate"] == "2024-03-03T00:00:00"


def test_fastf1_collector_reports_missing_dependency(tmp_path, monkeypatch):
    collector = FastF1Collector(raw_dir=tmp_path)
    monkeypatch.setattr(
        FastF1Collector,
        "_fastf1",
        staticmethod(
            lambda: (_ for _ in ()).throw(
                RuntimeError("FastF1 is not installed")
            )
        ),
    )
    with pytest.raises(RuntimeError, match="not installed"):
        collector.collect_event_schedules(start=2024, end=2024)
