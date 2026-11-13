"""Update pipeline: interval gating and hash-based retraining."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
import src.automation.update_pipeline as update_module
from src.automation.update_pipeline import UpdatePipeline, _parse_time


def _iso(delta_hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=delta_hours)).replace(
        microsecond=0
    ).isoformat()


def test_parse_time_handles_z_and_invalid_values():
    assert _parse_time("2024-01-01T00:00:00Z") is not None
    assert _parse_time("not-a-date") is None
    assert _parse_time(None) is None


def test_is_due_respects_interval(tmp_path):
    state_file = tmp_path / "update_state.json"
    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)
    assert pipeline.is_due() is True

    state_file.write_text(json.dumps({"last_run": _iso(1)}), encoding="utf-8")
    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)
    assert pipeline.is_due() is False

    state_file.write_text(json.dumps({"last_run": _iso(7)}), encoding="utf-8")
    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)
    assert pipeline.is_due() is True


def test_run_skips_when_not_due(tmp_path, monkeypatch):
    state_file = tmp_path / "update_state.json"
    state_file.write_text(json.dumps({"last_run": _iso(0)}), encoding="utf-8")
    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)

    def explode(*args, **kwargs):
        raise AssertionError("collector must not run when the interval has not elapsed")

    monkeypatch.setattr(update_module, "JolpicaCollector", explode)
    result = pipeline.run()

    assert result["status"] == "skipped"
    assert result["reason"] == "interval"


class _FakeSummary:
    latest_race = "2026_10"

    def as_dict(self):
        return {"source": "jolpica", "latest_race": self.latest_race}


class _FakeCollector:
    def __init__(self, *args, **kwargs):
        pass

    def detect_updates(self, season):
        return {"season": season, "changed": False}

    def collect(self, *args, **kwargs):
        return _FakeSummary()


class _FakeEngineer:
    def __init__(self, tables=None):
        self.tables = tables or {}

    def build(self):
        return pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1]})


def _patch_pipeline(monkeypatch, dataset_hash: str, calls: list):
    monkeypatch.setattr(update_module, "JolpicaCollector", _FakeCollector)
    monkeypatch.setattr(update_module, "build_processed_tables", lambda *a, **k: {})
    monkeypatch.setattr(update_module, "FeatureEngineer", _FakeEngineer)
    real_load_json = update_module.load_json

    def fake_load_json(path, default=None):
        # Only the dataset metadata is stubbed; the pipeline still reads its own
        # state file from disk.
        if str(path) == str(config.DATASET_METADATA_FILE):
            return {"dataset_hash": dataset_hash}
        return real_load_json(path, default=default)

    monkeypatch.setattr(update_module, "load_json", fake_load_json)

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {"models": {}, "skipped": []}

    monkeypatch.setattr(update_module, "train_all", fake_train)


def test_run_skips_retraining_when_dataset_hash_is_unchanged(tmp_path, monkeypatch):
    state_file = tmp_path / "update_state.json"
    state_file.write_text(
        json.dumps({"last_run": _iso(48), "dataset_hash": "same-hash"}),
        encoding="utf-8",
    )
    calls: list = []
    _patch_pipeline(monkeypatch, "same-hash", calls)

    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)
    result = pipeline.run()

    assert result["status"] == "ok"
    assert result["retrained"] is False
    assert result["steps"]["training"]["skipped"] is True
    assert result["steps"]["dataset"]["changed"] is False
    assert calls == []
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["dataset_hash"] == "same-hash"
    assert saved["latest_race"] == "2026_10"


def test_run_retrains_when_dataset_hash_changes(tmp_path, monkeypatch):
    state_file = tmp_path / "update_state.json"
    state_file.write_text(
        json.dumps({"last_run": _iso(48), "dataset_hash": "old-hash"}),
        encoding="utf-8",
    )
    calls: list = []
    _patch_pipeline(monkeypatch, "new-hash", calls)

    pipeline = UpdatePipeline(state_file=state_file, interval_hours=6)
    result = pipeline.run()

    assert result["retrained"] is True
    assert result["steps"]["dataset"]["changed"] is True
    assert len(calls) == 1
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["dataset_hash"] == "new-hash"
    assert saved["retrained"] is True
