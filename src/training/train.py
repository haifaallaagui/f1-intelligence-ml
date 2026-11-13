"""Training orchestration for every prediction task and prediction mode."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import config
from src.features.feature_engineering import (
    get_constructor_feature_columns,
    get_feature_columns,
    load_constructor_dataset,
    load_feature_dataset,
)
from src.models.base import load_bundle
from src.models.classification_model import ClassificationModel
from src.models.constructor_model import ConstructorPerformanceModel
from src.models.driver_model import DriverPerformanceModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from src.preprocessing.preprocessing import chronological_split
from src.training.model_selection import (
    CandidateResult,
    best_candidate,
    tune_task,
)
from src.training.registry import get_model_entry, register_model
from src.utils import ensure_dirs, get_logger, load_json, save_json, utc_now_iso

LOG = get_logger("training.train")

DRIVER_TASKS = ["winner", "podium", "classification", "driver_performance"]
CONSTRUCTOR_TASK = "constructor_performance"
DEFAULT_MODES = ["before_qualifying", "after_qualifying"]


def _task_algorithms(task: str) -> list[str]:
    if task in {"winner", "podium"}:
        return list(config.CLASSIFICATION_ALGORITHMS)
    if task == "classification":
        return list(config.RANKING_ALGORITHMS)
    return list(config.REGRESSION_ALGORITHMS)


def _task_target(task: str) -> str:
    return {
        "winner": "win",
        "podium": "podium",
        "classification": "finish_position",
        "driver_performance": "finish_position",
        "constructor_performance": "team_total_points",
    }[task]


def _filter_task_rows(task: str, df: pd.DataFrame) -> pd.DataFrame:
    target = _task_target(task)
    subset = df[df[target].notna()].copy()
    if task == "driver_performance":
        subset = subset[subset["total_points"].notna()]
    if task == "constructor_performance":
        subset = subset[subset["team_best_finish"].notna()]
    return subset


def _refit(
    task: str,
    best: CandidateResult,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    mode: str,
    *,
    temperature: float | None = None,
):
    combined = pd.concat([train_df, validation_df], ignore_index=True)
    if task == "winner":
        return WinnerModel(best.algorithm, best.params, mode, list(feature_columns)).fit(
            combined, feature_columns
        )
    if task == "podium":
        return PodiumModel(best.algorithm, best.params, mode, list(feature_columns)).fit(
            combined, feature_columns
        )
    if task == "classification":
        model = ClassificationModel(
            best.algorithm, best.params, mode, list(feature_columns)
        ).fit(combined, feature_columns)
        if temperature is not None:
            model.temperature = float(temperature)
        return model
    if task == "driver_performance":
        return DriverPerformanceModel(
            best.algorithm, best.params, mode, list(feature_columns)
        ).fit(combined, feature_columns)
    if task == "constructor_performance":
        return ConstructorPerformanceModel(
            best.algorithm, best.params, list(feature_columns)
        ).fit(combined, feature_columns)
    raise ValueError(task)


def _classification_winner_log_loss(
    model: ClassificationModel,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    temperature: float,
) -> float:
    matrices = model.position_probability_matrix(
        validation_df,
        feature_columns,
        n_samples=600,
        temperature=temperature,
        max_positions=20,
    )
    probabilities: list[float] = []
    for race_id, group in validation_df.groupby("race_id", sort=False):
        matrix = matrices.get(str(race_id))
        if matrix is None or "position_1" not in matrix.columns:
            probabilities.extend([np.nan] * len(group))
            continue
        p1 = matrix["position_1"]
        probabilities.extend([float(p1.get(str(driver), np.nan)) for driver in group["driver_id"]])
    scored = validation_df.assign(_p1=probabilities).dropna(subset=["_p1", "win"])
    if scored.empty:
        return float("inf")
    return float(
        log_loss(
            scored["win"].astype(int),
            np.clip(scored["_p1"].to_numpy(dtype=float), 1e-9, 1 - 1e-9),
            labels=[0, 1],
        )
    )


def _tune_temperature(
    model: ClassificationModel,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for temperature in (0.4, 0.6, 0.8, 1.0, 1.2):
        scores[str(temperature)] = _classification_winner_log_loss(
            model, validation_df, feature_columns, temperature
        )
    best = min(scores, key=scores.get)
    return float(best), scores


def _bundle_path(task: str, mode: str) -> Path:
    if task == CONSTRUCTOR_TASK:
        return config.TASK_DIRECTORIES[task] / "best.joblib"
    return config.TASK_DIRECTORIES[task] / mode / "best.joblib"


def _comparison_path(task: str, mode: str) -> Path:
    if task == CONSTRUCTOR_TASK:
        return config.TASK_DIRECTORIES[task] / "comparison.json"
    return config.TASK_DIRECTORIES[task] / mode / "comparison.json"


def _importance_path(task: str, mode: str) -> Path:
    if task == CONSTRUCTOR_TASK:
        return config.TASK_DIRECTORIES[task] / "feature_importance.csv"
    return config.TASK_DIRECTORIES[task] / mode / "feature_importance.csv"


def _model_is_current(task: str, mode: str, dataset_hash: str | None) -> bool:
    entry = get_model_entry(task, mode)
    if not entry or not dataset_hash:
        return False
    if not Path(entry.get("bundle", "")).exists():
        return False
    return (entry.get("metadata", {}) or {}).get("dataset_hash") == dataset_hash


def _existing_model_summary(task: str, mode: str) -> dict[str, Any] | None:
    """Describe a model that is already current without retraining it."""
    entry = get_model_entry(task, mode)
    if not entry:
        return None
    metadata = entry.get("metadata", {}) or {}
    params = metadata.get("params")
    bundle = entry.get("bundle")
    if not params and bundle and Path(bundle).exists():
        try:
            params = (load_bundle(Path(bundle)).get("metadata", {}) or {}).get("params")
        except Exception:  # noqa: BLE001 - a readable summary must not fail
            params = None
    return {
        "algorithm": entry.get("algorithm"),
        "params": params or {},
        "validation_metrics": metadata.get("validation_metrics", {}),
        "validation_score": metadata.get("validation_score"),
        "candidate_count": metadata.get("candidate_count"),
        "temperature": metadata.get("temperature"),
        "bundle": bundle,
        "version": entry.get("version"),
        "skipped": True,
    }


def train_all(
    *,
    tasks: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    n_iter: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config.ensure_directories()
    tasks = list(tasks or DRIVER_TASKS + [CONSTRUCTOR_TASK])
    modes = [m for m in (modes or DEFAULT_MODES) if m in {"before_qualifying", "after_qualifying"}]
    if not modes:
        modes = list(DEFAULT_MODES)
    n_iter = int(n_iter if n_iter is not None else config.N_TUNING_ITER)

    dataset = load_feature_dataset()
    constructor_dataset = load_constructor_dataset()
    if dataset.empty:
        raise RuntimeError("Driver feature dataset is empty; run scripts/build_dataset.py first")
    metadata = load_json(config.DATASET_METADATA_FILE, default={}) or {}
    dataset_hash = metadata.get("dataset_hash")
    version = datetime.now(timezone.utc).strftime(config.MODEL_VERSION_FORMAT)

    split = chronological_split(dataset)
    constructor_split = chronological_split(constructor_dataset) if not constructor_dataset.empty else None

    summary: dict[str, Any] = {
        "version": version,
        "started_at": utc_now_iso(),
        "dataset_hash": dataset_hash,
        "n_iter": n_iter,
        "models": {},
        "skipped": [],
    }

    for mode in modes:
        feature_columns = get_feature_columns(mode)
        for task in DRIVER_TASKS:
            if task not in tasks:
                continue
            key = f"{task}/{mode}"
            if not force and _model_is_current(task, mode, dataset_hash):
                LOG.info("Skipping %s: dataset hash unchanged", key)
                summary["skipped"].append(key)
                existing = _existing_model_summary(task, mode)
                if existing is not None:
                    summary["models"][key] = existing
                continue
            train_df = _filter_task_rows(task, split.train)
            validation_df = _filter_task_rows(task, split.validation)
            if train_df.empty or validation_df.empty:
                LOG.warning("Skipping %s: empty train/validation window", key)
                continue
            algorithms = _task_algorithms(task)
            LOG.info("Tuning %s with %s algorithms", key, len(algorithms))
            candidates = tune_task(
                task,
                train_df,
                validation_df,
                feature_columns,
                algorithms,
                mode=mode,
                n_iter=n_iter,
            )
            best = best_candidate(candidates)
            if best is None:
                LOG.error("No valid candidate for %s", key)
                continue
            temperature = None
            temperature_scores = None
            if task == "classification":
                temperature, temperature_scores = _tune_temperature(
                    best.model, validation_df, feature_columns
                )
            model = _refit(
                task,
                best,
                train_df,
                validation_df,
                feature_columns,
                mode,
                temperature=temperature,
            )
            bundle = _bundle_path(task, mode)
            extra = {
                "version": version,
                "dataset_hash": dataset_hash,
                "params": dict(best.params),
                "train_seasons": list(config.TRAIN_SEASONS),
                "validation_seasons": list(config.VALIDATION_SEASONS),
                "test_seasons": list(config.TEST_SEASONS),
                "n_train": int(len(train_df)),
                "n_validation": int(len(validation_df)),
                "validation_metrics": best.metrics,
                "validation_score": best.score,
                "candidate_count": len(candidates),
                "temperature": temperature,
                "temperature_selection": temperature_scores,
                "selected_at": utc_now_iso(),
            }
            model.save(bundle, extra)
            importance = model.feature_importance()
            ensure_dirs(_importance_path(task, mode).parent)
            importance.to_csv(_importance_path(task, mode), index=False)
            comparison = [item.summary() for item in candidates]
            save_json(
                _comparison_path(task, mode),
                {
                    "task": task,
                    "mode": mode,
                    "version": version,
                    "selected": best.summary(),
                    "candidates": comparison,
                },
            )
            register_model(
                task,
                mode,
                algorithm=best.algorithm,
                version=version,
                bundle_path=bundle,
                metadata=extra,
            )
            summary["models"][key] = {
                "algorithm": best.algorithm,
                "params": best.params,
                "validation_metrics": best.metrics,
                "validation_score": best.score,
                "candidate_count": len(candidates),
                "temperature": temperature,
                "bundle": str(bundle),
            }
            LOG.info(
                "Trained %s -> %s (validation score %.4f)",
                key,
                best.algorithm,
                best.score,
            )

    if CONSTRUCTOR_TASK in tasks and constructor_split is not None:
        mode = "any"
        key = CONSTRUCTOR_TASK
        if not force and _model_is_current(CONSTRUCTOR_TASK, mode, dataset_hash):
            LOG.info("Skipping %s: dataset hash unchanged", key)
            summary["skipped"].append(key)
            existing = _existing_model_summary(CONSTRUCTOR_TASK, mode)
            if existing is not None:
                summary["models"][key] = existing
        else:
            feature_columns = get_constructor_feature_columns()
            train_df = _filter_task_rows(CONSTRUCTOR_TASK, constructor_split.train)
            validation_df = _filter_task_rows(CONSTRUCTOR_TASK, constructor_split.validation)
            if not train_df.empty and not validation_df.empty:
                candidates = tune_task(
                    CONSTRUCTOR_TASK,
                    train_df,
                    validation_df,
                    feature_columns,
                    _task_algorithms(CONSTRUCTOR_TASK),
                    mode=mode,
                    n_iter=n_iter,
                )
                best = best_candidate(candidates)
                if best is not None:
                    model = _refit(
                        CONSTRUCTOR_TASK,
                        best,
                        train_df,
                        validation_df,
                        feature_columns,
                        mode,
                    )
                    bundle = _bundle_path(CONSTRUCTOR_TASK, mode)
                    extra = {
                        "version": version,
                        "dataset_hash": dataset_hash,
                        "params": dict(best.params),
                        "train_seasons": list(config.TRAIN_SEASONS),
                        "validation_seasons": list(config.VALIDATION_SEASONS),
                        "test_seasons": list(config.TEST_SEASONS),
                        "n_train": int(len(train_df)),
                        "n_validation": int(len(validation_df)),
                        "validation_metrics": best.metrics,
                        "validation_score": best.score,
                        "candidate_count": len(candidates),
                        "selected_at": utc_now_iso(),
                    }
                    model.save(bundle, extra)
                    importance = model.feature_importance()
                    ensure_dirs(_importance_path(CONSTRUCTOR_TASK, mode).parent)
                    importance.to_csv(_importance_path(CONSTRUCTOR_TASK, mode), index=False)
                    save_json(
                        _comparison_path(CONSTRUCTOR_TASK, mode),
                        {
                            "task": CONSTRUCTOR_TASK,
                            "mode": mode,
                            "version": version,
                            "selected": best.summary(),
                            "candidates": [item.summary() for item in candidates],
                        },
                    )
                    register_model(
                        CONSTRUCTOR_TASK,
                        mode,
                        algorithm=best.algorithm,
                        version=version,
                        bundle_path=bundle,
                        metadata=extra,
                    )
                    summary["models"][key] = {
                        "algorithm": best.algorithm,
                        "params": best.params,
                        "validation_metrics": best.metrics,
                        "validation_score": best.score,
                        "candidate_count": len(candidates),
                        "bundle": str(bundle),
                    }
                    LOG.info(
                        "Trained %s -> %s (validation score %.4f)",
                        key,
                        best.algorithm,
                        best.score,
                    )

    summary["finished_at"] = utc_now_iso()
    save_json(config.REPORTS_DIR / "training_summary.json", summary)
    return summary
