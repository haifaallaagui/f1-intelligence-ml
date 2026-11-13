"""Automatic data update and retraining pipeline.

The pipeline is intentionally conservative:

* it only refreshes the recent seasons (plus missing reference data);
* it compares dataset hashes and skips retraining when nothing changed;
* it records every run in ``data/metadata/update_state.json``.

The frequency is controlled by ``F1_UPDATE_INTERVAL_HOURS`` (default 6h).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import config
from src.data_collection.jolpica_collector import JolpicaCollector
from src.features.feature_engineering import FeatureEngineer, load_feature_dataset
from src.preprocessing.cleaning import build_processed_tables
from src.training.train import train_all
from src.utils import get_logger, load_json, save_json, utc_now_iso

LOG = get_logger("automation.update")


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class UpdatePipeline:
    def __init__(self, state_file=None, interval_hours: float | None = None):
        self.state_file = state_file or config.UPDATE_STATE_FILE
        self.interval = timedelta(
            hours=float(interval_hours if interval_hours is not None else config.UPDATE_INTERVAL_HOURS)
        )
        self.state = load_json(self.state_file, default={}) or {}

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        last = _parse_time(self.state.get("last_run"))
        if last is None:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last) >= self.interval

    def run(
        self,
        *,
        force: bool = False,
        retrain: bool = True,
        n_iter: int | None = None,
        skip_openf1: bool = False,
        skip_fastf1: bool = False,
    ) -> dict[str, Any]:
        if not force and not self.is_due():
            LOG.info("Update not due yet (interval %s)", self.interval)
            return {"status": "skipped", "reason": "interval", "state": self.state}

        previous_hash = self.state.get("dataset_hash")
        result: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "status": "ok",
            "steps": {},
        }

        collector = JolpicaCollector()
        update_info = collector.detect_updates(config.SEASON_END)
        result["steps"]["update_detection"] = update_info
        recent_start = max(config.SEASON_START, config.SEASON_END - 2)
        summary = collector.collect(
            start=recent_start,
            end=config.SEASON_END,
            update=True,
            force=False,
        )
        result["steps"]["jolpica"] = summary.as_dict()

        tables = build_processed_tables(
            config.SEASON_START,
            config.SEASON_END,
            include_openf1=not skip_openf1,
            include_fastf1=not skip_fastf1,
        )
        result["steps"]["processed_tables"] = {
            name: int(len(frame)) for name, frame in tables.items()
        }

        engineer = FeatureEngineer(tables=tables)
        dataset, constructor_dataset = engineer.build()
        current_hash = load_json(config.DATASET_METADATA_FILE, default={}).get("dataset_hash")
        result["steps"]["dataset"] = {
            "rows": int(len(dataset)),
            "constructor_rows": int(len(constructor_dataset)),
            "dataset_hash": current_hash,
            "changed": bool(previous_hash and current_hash != previous_hash),
        }

        if retrain and (force or current_hash != previous_hash):
            LOG.info("Dataset changed; retraining models")
            training = train_all(force=False, n_iter=n_iter)
            result["steps"]["training"] = training
            result["retrained"] = True
        else:
            result["steps"]["training"] = {
                "skipped": True,
                "reason": "dataset unchanged" if current_hash == previous_hash else "retrain disabled",
            }
            result["retrained"] = False

        result["finished_at"] = utc_now_iso()
        result["dataset_hash"] = current_hash
        result["latest_race"] = summary.latest_race
        result["status"] = "ok"
        self.state = {
            "last_run": result["finished_at"],
            "dataset_hash": current_hash,
            "latest_race": summary.latest_race,
            "retrained": result["retrained"],
            "last_result": {
                "status": result["status"],
                "retrained": result["retrained"],
                "latest_race": summary.latest_race,
            },
        }
        save_json(self.state_file, self.state)
        return result
