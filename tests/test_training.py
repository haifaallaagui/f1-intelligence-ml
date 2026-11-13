"""Training orchestration, hyper-parameter sampling and the model registry."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
import src.training.train as train_module
from src.preprocessing.preprocessing import DatasetSplit
from src.training.model_selection import (
    CLASSIFIER_GRIDS,
    CandidateResult,
    best_candidate,
    candidate_parameter_sets,
    tune_task,
)
from src.training.registry import get_model_entry, list_models, register_model
from tests.conftest import FEATURE_COLUMNS, make_driver_frame


# ---------------------------------------------------------------------------
# Hyper-parameter sampling and candidate selection
# ---------------------------------------------------------------------------
def test_candidate_parameter_sets_always_include_baseline():
    rng = np.random.default_rng(0)
    sets = candidate_parameter_sets("winner", "logistic_regression", 2, rng)
    assert len(sets) == 2
    assert sets[0] == CLASSIFIER_GRIDS["logistic_regression"][0]

    everything = candidate_parameter_sets("winner", "logistic_regression", 99, rng)
    assert everything == [dict(item) for item in CLASSIFIER_GRIDS["logistic_regression"]]


def test_candidate_parameter_sets_for_unknown_algorithm_is_empty():
    rng = np.random.default_rng(0)
    assert candidate_parameter_sets("winner", "nope", 3, rng) == []


def test_best_candidate_ignores_failures():
    failed = CandidateResult(
        task="winner",
        algorithm="a",
        params={},
        metrics={},
        score=float("-inf"),
        tiebreak=float("-inf"),
        error="boom",
    )
    good = CandidateResult(
        task="winner",
        algorithm="b",
        params={},
        metrics={"log_loss": 0.4},
        score=-0.4,
        tiebreak=0.8,
    )
    assert best_candidate([failed, good]) is good
    assert best_candidate([failed]) is None
    assert good.summary()["algorithm"] == "b"


def test_tune_task_scores_on_validation_window(driver_frame):
    train = driver_frame[driver_frame["season"] <= 2017]
    validation = driver_frame[driver_frame["season"] == 2018]
    results = tune_task(
        "winner",
        train,
        validation,
        FEATURE_COLUMNS,
        ["logistic_regression"],
        mode="after_qualifying",
        n_iter=2,
    )
    assert results
    assert all(result.error is None for result in results)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(score) for score in scores)


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------
def test_task_helpers_map_tasks_to_targets_and_algorithms():
    assert train_module._task_target("winner") == "win"
    assert train_module._task_target("classification") == "finish_position"
    assert train_module._task_algorithms("classification") == list(config.RANKING_ALGORITHMS)
    assert train_module._task_algorithms("winner") == list(config.CLASSIFICATION_ALGORITHMS)

    frame = pd.DataFrame(
        {
            "finish_position": [1.0, np.nan],
            "total_points": [25.0, 18.0],
            "team_total_points": [40.0, 20.0],
            "team_best_finish": [1.0, 3.0],
            "win": [1, 0],
        }
    )
    assert len(train_module._filter_task_rows("driver_performance", frame)) == 1
    assert len(train_module._filter_task_rows("constructor_performance", frame)) == 2
    assert len(train_module._filter_task_rows("winner", frame)) == 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_roundtrip(tmp_path):
    registry_path = tmp_path / "registry.json"
    bundle = tmp_path / "winner" / "best.joblib"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"not-a-real-model")

    entry = register_model(
        "winner",
        "after_qualifying",
        algorithm="logistic_regression",
        version="v1",
        bundle_path=bundle,
        metadata={"dataset_hash": "h1"},
        registry_path=registry_path,
    )
    assert entry["algorithm"] == "logistic_regression"

    loaded = get_model_entry("winner", "after_qualifying", registry_path=registry_path)
    assert loaded["metadata"]["dataset_hash"] == "h1"
    assert len(list_models(registry_path=registry_path)) == 1
    assert get_model_entry("podium", "after_qualifying", registry_path=registry_path) is None

    register_model(
        "podium",
        "before_qualifying",
        algorithm="xgboost",
        version="v2",
        bundle_path=bundle,
        metadata={"dataset_hash": "h2"},
        registry_path=registry_path,
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v2"
    assert set(payload["models"]) == {"winner/after_qualifying", "podium/before_qualifying"}


# ---------------------------------------------------------------------------
# train_all integration (patched paths and dataset loaders)
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_training(tmp_path, monkeypatch):
    dataset = make_driver_frame()
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps({"dataset_hash": "hash-1"}), encoding="utf-8")

    monkeypatch.setattr(config, "ensure_directories", lambda: None)
    monkeypatch.setattr(config, "TASK_DIRECTORIES", {
        task: tmp_path / "models" / task for task in config.TASK_DIRECTORIES
    })
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DATASET_METADATA_FILE", metadata_path)
    monkeypatch.setattr(config, "CLASSIFICATION_ALGORITHMS", ["logistic_regression"])
    monkeypatch.setattr(train_module, "load_feature_dataset", lambda: dataset)
    monkeypatch.setattr(train_module, "load_constructor_dataset", lambda: pd.DataFrame())
    monkeypatch.setattr(train_module, "get_feature_columns", lambda mode="after_qualifying": list(FEATURE_COLUMNS))
    monkeypatch.setattr(
        train_module,
        "chronological_split",
        lambda frame: DatasetSplit(
            train=frame[frame["season"] <= 2017],
            validation=frame[frame["season"] == 2018],
            test=frame.iloc[0:0],
        ),
    )
    return tmp_path, metadata_path


def test_train_all_writes_bundle_comparison_and_registry(patched_training):
    tmp_path, _ = patched_training

    summary = train_module.train_all(
        tasks=["winner"], modes=["after_qualifying"], n_iter=1
    )

    key = "winner/after_qualifying"
    assert key in summary["models"]
    assert summary["models"][key]["algorithm"] == "logistic_regression"
    assert summary["models"][key]["candidate_count"] == 1

    bundle = tmp_path / "models" / "winner" / "after_qualifying" / "best.joblib"
    assert bundle.exists()
    assert bundle.with_suffix(".metadata.json").exists()
    comparison = tmp_path / "models" / "winner" / "after_qualifying" / "comparison.json"
    assert json.loads(comparison.read_text(encoding="utf-8"))["selected"]["algorithm"] == "logistic_regression"
    assert (tmp_path / "models" / "winner" / "after_qualifying" / "feature_importance.csv").exists()
    assert (tmp_path / "reports" / "training_summary.json").exists()

    entry = get_model_entry("winner", "after_qualifying")
    assert entry is not None
    assert entry["metadata"]["dataset_hash"] == "hash-1"


def test_train_all_skips_unchanged_dataset(patched_training):
    tmp_path, _ = patched_training
    train_module.train_all(tasks=["winner"], modes=["after_qualifying"], n_iter=1)

    second = train_module.train_all(
        tasks=["winner"], modes=["after_qualifying"], n_iter=1
    )

    assert "winner/after_qualifying" in second["skipped"]
    assert second["models"] == {}


def test_train_all_retrains_when_dataset_hash_changes(patched_training):
    tmp_path, metadata_path = patched_training
    train_module.train_all(tasks=["winner"], modes=["after_qualifying"], n_iter=1)

    metadata_path.write_text(json.dumps({"dataset_hash": "hash-2"}), encoding="utf-8")
    third = train_module.train_all(
        tasks=["winner"], modes=["after_qualifying"], n_iter=1
    )

    assert "winner/after_qualifying" in third["models"]
    entry = get_model_entry("winner", "after_qualifying")
    assert entry["metadata"]["dataset_hash"] == "hash-2"
