"""Hyper-parameter search and model comparison.

The search is deliberately small and deterministic.  Every candidate is fitted
on the training window only and scored on the validation window; the test
window is never touched here.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

import config
from src.models.classification_model import ClassificationModel
from src.models.constructor_model import ConstructorPerformanceModel
from src.models.driver_model import DriverPerformanceModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from src.training.metrics import (
    classification_metrics,
    constructor_performance_metrics,
    driver_performance_metrics,
    podium_metrics,
    winner_metrics,
)
from src.utils import get_logger

LOG = get_logger("training.model_selection")

CLASSIFIER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [
        {"C": 0.05},
        {"C": 0.2},
        {"C": 1.0},
        {"C": 5.0},
    ],
    "random_forest": [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        {"learning_rate": 0.06, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        {"learning_rate": 0.1, "max_iter": 250, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    ],
    "xgboost": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.9},
    ],
    "lightgbm": [
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
        {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 10},
        {"n_estimators": 250, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 30},
    ],
    "catboost": [
        {"iterations": 300, "depth": 4, "learning_rate": 0.03},
        {"iterations": 300, "depth": 6, "learning_rate": 0.05},
        {"iterations": 400, "depth": 6, "learning_rate": 0.03},
        {"iterations": 250, "depth": 8, "learning_rate": 0.05},
    ],
}

RANKER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "xgboost_ranker": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.9},
    ],
    "lightgbm_ranker": [
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
        {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 10},
        {"n_estimators": 250, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 30},
    ],
}

REGRESSOR_GRIDS: dict[str, list[dict[str, Any]]] = {
    "hist_gradient_boosting": CLASSIFIER_GRIDS["hist_gradient_boosting"],
    "random_forest": CLASSIFIER_GRIDS["random_forest"],
    "xgboost": CLASSIFIER_GRIDS["xgboost"],
    "lightgbm": CLASSIFIER_GRIDS["lightgbm"],
    "catboost": CLASSIFIER_GRIDS["catboost"],
}


@dataclass
class CandidateResult:
    task: str
    algorithm: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float
    tiebreak: float
    model: Any = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "algorithm": self.algorithm,
            "params": self.params,
            "metrics": self.metrics,
            "score": self.score,
            "tiebreak": self.tiebreak,
            "error": self.error,
        }


def _grid_for(task: str, algorithm: str) -> list[dict[str, Any]]:
    if task == "classification":
        return RANKER_GRIDS.get(algorithm) or REGRESSOR_GRIDS.get(algorithm) or []
    if task in {"driver_performance", "constructor_performance"}:
        return REGRESSOR_GRIDS.get(algorithm) or []
    return CLASSIFIER_GRIDS.get(algorithm) or []


def candidate_parameter_sets(
    task: str, algorithm: str, n_iter: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    grid = _grid_for(task, algorithm)
    if not grid:
        return []
    if n_iter >= len(grid):
        return [dict(item) for item in grid]
    # Always include the first (baseline) configuration, then sample the rest.
    indices = [0] + sorted(
        rng.choice(np.arange(1, len(grid)), size=max(0, n_iter - 1), replace=False).tolist()
    )
    return [dict(grid[i]) for i in indices]


def fit_candidate(
    task: str,
    algorithm: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    mode: str,
) -> CandidateResult | None:
    try:
        if task == "winner":
            model = WinnerModel(algorithm, dict(params), mode, list(feature_columns)).fit(
                train_df, feature_columns
            )
            predictions = model.predict_race(validation_df, feature_columns)
            metrics = winner_metrics(predictions)
            score = -float(metrics["log_loss"])
            tiebreak = float(metrics["accuracy"])
        elif task == "podium":
            model = PodiumModel(algorithm, dict(params), mode, list(feature_columns)).fit(
                train_df, feature_columns
            )
            predictions = model.predict_race(validation_df, feature_columns)
            metrics = podium_metrics(predictions)
            score = -float(metrics["log_loss"])
            tiebreak = float(metrics["f1"])
        elif task == "classification":
            model = ClassificationModel(algorithm, dict(params), mode, list(feature_columns)).fit(
                train_df, feature_columns
            )
            predictions = model.predict_race(validation_df, feature_columns)
            metrics = classification_metrics(predictions)
            score = -float(metrics["mae"])
            tiebreak = float(metrics.get("spearman", 0.0) or 0.0)
        elif task == "driver_performance":
            model = DriverPerformanceModel(
                algorithm, dict(params), mode, list(feature_columns)
            ).fit(train_df, feature_columns)
            predictions = model.predict(validation_df, feature_columns)
            metrics = driver_performance_metrics(predictions)
            score = -float(metrics["finish_mae"])
            tiebreak = -float(metrics["points_mae"])
        elif task == "constructor_performance":
            model = ConstructorPerformanceModel(
                algorithm, dict(params), list(feature_columns)
            ).fit(train_df, feature_columns)
            predictions = model.predict(validation_df, feature_columns)
            metrics = constructor_performance_metrics(predictions)
            score = -float(metrics["points_mae"])
            tiebreak = -float(metrics["best_finish_mae"])
        else:
            raise ValueError(f"Unknown task {task}")
        if not np.isfinite(score):
            return None
        return CandidateResult(
            task=task,
            algorithm=algorithm,
            params=dict(params),
            metrics=metrics,
            score=score,
            tiebreak=tiebreak if np.isfinite(tiebreak) else 0.0,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - a missing optional dependency should not stop the search
        LOG.warning("Candidate %s/%s failed: %s", task, algorithm, exc)
        return CandidateResult(
            task=task,
            algorithm=algorithm,
            params=dict(params),
            metrics={},
            score=float("-inf"),
            tiebreak=float("-inf"),
            error=str(exc),
        )


def tune_task(
    task: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    algorithms: Sequence[str],
    *,
    mode: str = "after_qualifying",
    n_iter: int = 3,
    seed: int = config.RANDOM_SEED,
) -> list[CandidateResult]:
    rng = np.random.default_rng(seed)
    results: list[CandidateResult] = []
    for algorithm in algorithms:
        for params in candidate_parameter_sets(task, algorithm, n_iter, rng):
            LOG.info("Fitting %s candidate %s %s", task, algorithm, params)
            result = fit_candidate(
                task, algorithm, params, train_df, validation_df, feature_columns, mode
            )
            if result is not None:
                results.append(result)
    results.sort(key=lambda item: (item.score, item.tiebreak), reverse=True)
    return results


def best_candidate(results: Sequence[CandidateResult]) -> CandidateResult | None:
    valid = [item for item in results if item.error is None and np.isfinite(item.score)]
    return valid[0] if valid else None
