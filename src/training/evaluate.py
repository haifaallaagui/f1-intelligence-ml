"""Evaluation of trained models and simple baselines on the chronological test window."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import config
from src.features.feature_engineering import (
    get_constructor_feature_columns,
    get_feature_columns,
    load_constructor_dataset,
    load_feature_dataset,
)
from src.models.classification_model import ClassificationModel
from src.models.constructor_model import ConstructorPerformanceModel
from src.models.driver_model import DriverPerformanceModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from src.preprocessing.preprocessing import chronological_split
from src.training.metrics import (
    classification_metrics,
    constructor_performance_metrics,
    driver_performance_metrics,
    podium_metrics,
    probability_from_score,
    winner_metrics,
)
from src.training.registry import get_model_entry
from src.utils import ensure_dirs, get_logger, load_json, save_json, utc_now_iso

LOG = get_logger("training.evaluate")

DRIVER_TASKS = ["winner", "podium", "classification", "driver_performance"]
MODES = ["before_qualifying", "after_qualifying"]


def _default_bundle(task: str, mode: str) -> Path:
    if task == "constructor_performance":
        return config.TASK_DIRECTORIES[task] / "best.joblib"
    return config.TASK_DIRECTORIES[task] / mode / "best.joblib"


def _bundle_for(task: str, mode: str) -> Path:
    entry = get_model_entry(task, mode)
    if entry and entry.get("bundle"):
        path = Path(entry["bundle"])
        if path.exists():
            return path
    return _default_bundle(task, mode)


def load_task_model(task: str, mode: str):
    path = _bundle_for(task, mode)
    if not path.exists():
        return None
    if task == "winner":
        return WinnerModel.load(path)
    if task == "podium":
        return PodiumModel.load(path)
    if task == "classification":
        return ClassificationModel.load(path)
    if task == "driver_performance":
        return DriverPerformanceModel.load(path)
    if task == "constructor_performance":
        return ConstructorPerformanceModel.load(path)
    raise ValueError(task)


def _safe_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _score_to_probability(df: pd.DataFrame, score: pd.Series, *, podium: bool = False) -> pd.Series:
    working = df.copy()
    working["_baseline_score"] = score.fillna(0.0)
    probability = probability_from_score(working, "_baseline_score")
    if podium:
        probability = (probability * 3.0).clip(0.0, 1.0)
    return probability


def _apply_baseline(df: pd.DataFrame, score: pd.Series, task: str) -> pd.DataFrame:
    out = df.copy()
    out["_baseline_score"] = score.fillna(0.0)
    if task == "winner":
        out["winner_probability"] = _score_to_probability(out, out["_baseline_score"])
    elif task == "podium":
        out["podium_probability"] = _score_to_probability(
            out, out["_baseline_score"], podium=True
        )
    elif task == "classification":
        out["predicted_position"] = (
            out.groupby("race_id")["_baseline_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    return out


def _baselines(df: pd.DataFrame, task: str, mode: str) -> dict[str, dict[str, Any]]:
    candidates: dict[str, pd.Series] = {
        "championship_order": -_safe_column(df, "driver_championship_position"),
        "recent_form": -_safe_column(df, "driver_last5_avg_finish"),
        "career_form": -_safe_column(df, "driver_career_avg_finish"),
    }
    if mode == "after_qualifying":
        candidates["grid_order"] = -_safe_column(df, "grid_position")
        candidates["qualifying_order"] = -_safe_column(df, "quali_position")
    out: dict[str, dict[str, Any]] = {}
    for name, score in candidates.items():
        if score.isna().all():
            continue
        predicted = _apply_baseline(df, score, task)
        if task == "winner":
            out[name] = winner_metrics(predicted)
        elif task == "podium":
            out[name] = podium_metrics(predicted)
        elif task == "classification":
            out[name] = classification_metrics(predicted)
    return out


def _evaluate_driver_task(
    task: str, mode: str, test_df: pd.DataFrame
) -> dict[str, Any] | None:
    model = load_task_model(task, mode)
    if model is None:
        LOG.warning("No model found for %s/%s", task, mode)
        return None
    feature_columns = get_feature_columns(mode)
    if task == "winner":
        predictions = model.predict_race(test_df, feature_columns)
        metrics = winner_metrics(predictions)
    elif task == "podium":
        predictions = model.predict_race(test_df, feature_columns)
        metrics = podium_metrics(predictions)
    elif task == "classification":
        predictions = model.predict_race(test_df, feature_columns)
        metrics = classification_metrics(predictions)
    elif task == "driver_performance":
        predictions = model.predict(test_df, feature_columns)
        metrics = driver_performance_metrics(predictions)
    else:
        return None
    return {
        "metrics": metrics,
        "baselines": _baselines(test_df, task, mode),
        "n_rows": int(len(test_df)),
        "algorithm": getattr(model, "algorithm", None),
        "bundle": str(_bundle_for(task, mode)),
    }


def _evaluate_constructor(test_df: pd.DataFrame) -> dict[str, Any] | None:
    model = load_task_model("constructor_performance", "any")
    if model is None:
        LOG.warning("No constructor model found")
        return None
    feature_columns = get_constructor_feature_columns()
    predictions = model.predict(test_df, feature_columns)
    metrics = constructor_performance_metrics(predictions)
    # Simple baselines: current constructor championship position and recent form.
    baselines = {}
    for name, score in {
        "championship_order": -_safe_column(test_df, "constructor_championship_position"),
        "recent_form": -_safe_column(test_df, "constructor_last5_avg_points"),
    }.items():
        if score.isna().all():
            continue
        working = test_df.copy()
        working["expected_team_points"] = score.fillna(0.0)
        working["expected_team_best_finish"] = -score.fillna(0.0)
        baselines[name] = constructor_performance_metrics(working)
    return {
        "metrics": metrics,
        "baselines": baselines,
        "n_rows": int(len(test_df)),
        "algorithm": getattr(model, "algorithm", None),
        "bundle": str(_bundle_for("constructor_performance", "any")),
    }


def _feature_importance(task: str, mode: str, top_n: int = 15) -> list[dict[str, Any]]:
    path = (
        config.TASK_DIRECTORIES[task] / "feature_importance.csv"
        if task == "constructor_performance"
        else config.TASK_DIRECTORIES[task] / mode / "feature_importance.csv"
    )
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path).head(top_n)
        return frame.to_dict("records")
    except Exception:
        return []


def _validation_comparison(task: str, mode: str) -> dict[str, Any] | None:
    path = (
        config.TASK_DIRECTORIES[task] / "comparison.json"
        if task == "constructor_performance"
        else config.TASK_DIRECTORIES[task] / mode / "comparison.json"
    )
    return load_json(path, default=None)


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def evaluate_all() -> dict[str, Any]:
    config.ensure_directories()
    dataset = load_feature_dataset()
    constructor_dataset = load_constructor_dataset()
    if dataset.empty:
        raise RuntimeError("Driver feature dataset is empty; build the dataset first")
    split = chronological_split(dataset)
    test = split.test
    if test.empty:
        raise RuntimeError("Test window is empty; check config.TEST_SEASONS")
    constructor_test = (
        chronological_split(constructor_dataset).test if not constructor_dataset.empty else pd.DataFrame()
    )

    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "test_seasons": list(config.TEST_SEASONS),
        "dataset_rows": int(len(dataset)),
        "test_rows": int(len(test)),
        "tasks": {},
        "model_comparison": {},
        "feature_importance": {},
    }

    for mode in MODES:
        for task in DRIVER_TASKS:
            key = f"{task}/{mode}"
            LOG.info("Evaluating %s", key)
            result = _evaluate_driver_task(task, mode, test)
            if result is not None:
                report["tasks"][key] = result
                report["model_comparison"][key] = _validation_comparison(task, mode)
                report["feature_importance"][key] = _feature_importance(task, mode)

    constructor_result = _evaluate_constructor(constructor_test) if not constructor_test.empty else None
    if constructor_result is not None:
        report["tasks"]["constructor_performance/any"] = constructor_result
        report["model_comparison"]["constructor_performance/any"] = _validation_comparison(
            "constructor_performance", "any"
        )
        report["feature_importance"]["constructor_performance/any"] = _feature_importance(
            "constructor_performance", "any"
        )

    save_json(config.REPORTS_DIR / "evaluation_report.json", report)
    _write_markdown(report)
    return report


def _write_markdown(report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# F1 Intelligence ML - Evaluation Report\n")
    lines.append(f"Generated: {report['generated_at']}\n")
    lines.append(
        f"Test window: {report['test_seasons'][0]}-{report['test_seasons'][1]} "
        f"({report['test_rows']} driver-race rows).\n"
    )
    lines.append(
        "All models are trained on 1995-2018, tuned on 2019-2021 and evaluated "
        "on 2022-2026.  The test window is never used for model selection.\n"
    )

    task_titles = {
        "winner": "Winner prediction",
        "podium": "Podium prediction",
        "classification": "Full classification (P1-P20)",
        "driver_performance": "Driver performance",
        "constructor_performance": "Constructor performance",
    }
    for key, result in report["tasks"].items():
        task, mode = key.split("/")
        lines.append(f"\n## {task_titles.get(task, task)} - {mode}\n")
        lines.append(f"Selected model: `{result.get('algorithm')}`\n")
        metrics = result.get("metrics", {})
        if task == "winner":
            columns = ["log_loss", "brier", "accuracy", "top3_hit_rate", "n_races"]
        elif task == "podium":
            columns = ["log_loss", "brier", "precision", "recall", "f1", "exact_set_accuracy", "n_races"]
        elif task == "classification":
            columns = ["mae", "mae_classified", "rmse", "r2", "spearman", "ndcg_at_10", "ranking_accuracy", "top1_accuracy", "podium_precision_at_3"]
        elif task == "driver_performance":
            columns = ["finish_mae", "finish_rmse", "finish_r2", "points_mae", "points_rmse", "points_r2"]
        else:
            columns = ["points_mae", "points_rmse", "points_r2", "best_finish_mae", "best_finish_rmse", "best_finish_r2"]
        lines.append(_markdown_table([metrics], columns))

        baselines = result.get("baselines", {}) or {}
        if baselines:
            lines.append("### Baselines\n")
            baseline_rows = []
            for name, values in baselines.items():
                row = {"baseline": name}
                row.update({column: values.get(column) for column in columns})
                baseline_rows.append(row)
            lines.append(_markdown_table(baseline_rows, ["baseline"] + columns))

        importance = report.get("feature_importance", {}).get(key, [])
        if importance:
            lines.append("### Top features\n")
            lines.append(
                _markdown_table(
                    [{"feature": item.get("feature"), "importance": item.get("importance")} for item in importance[:10]],
                    ["feature", "importance"],
                )
            )

    lines.append("\n## Notes and limitations\n")
    lines.append("- Historical aggregates are updated only after each race (leakage-safe).")
    lines.append("- Mode 1 (before qualifying) never uses qualifying, grid or sprint columns.")
    lines.append("- Mode 2 (after qualifying) uses the actual grid when it is available, otherwise a flagged qualifying-order proxy.")
    lines.append("- OpenF1 coverage begins in 2023 and FastF1 timing enrichment is optional; missing values remain missing.")
    lines.append("- Championship projections use Monte-Carlo Plackett-Luce sampling and do not simulate the fastest-lap bonus point.")
    lines.append("- Metrics are reported on the chronological test window; the baseline comparison is the honest reference point.")
    (config.REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
