"""Evaluation metrics for every prediction task.

All functions accept a driver-race (or constructor-race) frame that already
contains both the predictions and the observed outcomes.  They never invent
labels: rows with a missing target are excluded from the corresponding metric.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocessing import ranking_relevance


def _clip_probabilities(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(list(values), dtype=float), 1e-9, 1.0 - 1e-9)


def _classified_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for officially classified cars.

    The feature dataset stores ``classified_position`` (numeric, NaN for
    retired/unclassified cars).  Older callers and tests may instead provide a
    boolean ``classified`` column.  Both layouts are supported so metrics stay
    honest about which cars actually received a classification.
    """
    if "classified_position" in df.columns:
        return pd.to_numeric(df["classified_position"], errors="coerce").notna()
    if "classified" in df.columns:
        return df["classified"].fillna(False).astype(bool)
    return pd.Series(True, index=df.index, dtype=bool)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    true, pred = true[mask], pred[mask]
    r2 = float(r2_score(true, pred)) if len(true) > 1 and np.std(true) > 0 else float("nan")
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": r2,
        "n": int(mask.sum()),
    }


def winner_metrics(df: pd.DataFrame, probability_column: str = "winner_probability") -> dict[str, Any]:
    data = df[df["win"].notna()].copy()
    if data.empty:
        return {}
    data[probability_column] = data[probability_column].fillna(0.0)
    y = data["win"].astype(int).to_numpy()
    p = _clip_probabilities(data[probability_column])
    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "n_rows": int(len(data)),
    }
    race_hits = []
    top3_hits = []
    for _, group in data.groupby("race_id", sort=False):
        group = group.sort_values(probability_column, ascending=False)
        race_hits.append(float(group.iloc[0]["win"] == 1))
        top3 = set(group.head(3)["driver_id"].astype(str))
        actual = set(group[group["win"] == 1]["driver_id"].astype(str))
        top3_hits.append(float(bool(top3 & actual)))
    metrics["accuracy"] = float(np.mean(race_hits)) if race_hits else float("nan")
    metrics["top3_hit_rate"] = float(np.mean(top3_hits)) if top3_hits else float("nan")
    metrics["n_races"] = int(len(race_hits))
    return metrics


def podium_metrics(df: pd.DataFrame, probability_column: str = "podium_probability") -> dict[str, Any]:
    data = df[df["podium"].notna()].copy()
    if data.empty:
        return {}
    data[probability_column] = data[probability_column].fillna(0.0)
    y = data["podium"].astype(int).to_numpy()
    p = _clip_probabilities(data[probability_column])
    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "n_rows": int(len(data)),
    }
    precisions, recalls, f1s, exact = [], [], [], []
    for _, group in data.groupby("race_id", sort=False):
        predicted = set(group.sort_values(probability_column, ascending=False).head(3)["driver_id"].astype(str))
        actual = set(group[group["podium"] == 1]["driver_id"].astype(str))
        intersection = len(predicted & actual)
        precision = intersection / 3.0 if predicted else 0.0
        recall = intersection / len(actual) if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        exact.append(float(predicted == actual))
    metrics.update(
        {
            "precision": float(np.mean(precisions)) if precisions else float("nan"),
            "recall": float(np.mean(recalls)) if recalls else float("nan"),
            "f1": float(np.mean(f1s)) if f1s else float("nan"),
            "exact_set_accuracy": float(np.mean(exact)) if exact else float("nan"),
            "n_races": int(len(precisions)),
        }
    )
    return metrics


def _ndcg_at_k(relevance: np.ndarray, order: np.ndarray, k: int) -> float:
    relevance = np.asarray(relevance, dtype=float)
    ideal = np.sort(relevance)[::-1][:k]
    dcg = 0.0
    for rank, index in enumerate(order[:k]):
        dcg += (2 ** relevance[index] - 1) / np.log2(rank + 2)
    idcg = 0.0
    for rank, value in enumerate(ideal):
        idcg += (2 ** value - 1) / np.log2(rank + 2)
    return dcg / idcg if idcg > 0 else 0.0


def _pairwise_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    n = len(actual)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] == actual[j]:
                continue
            total += 1
            if (actual[i] < actual[j]) == (predicted[i] < predicted[j]):
                correct += 1
    return correct / total if total else float("nan")


def classification_metrics(
    df: pd.DataFrame,
    position_column: str = "predicted_position",
) -> dict[str, Any]:
    data = df[df["finish_position"].notna() & df[position_column].notna()].copy()
    if data.empty:
        return {}
    metrics: dict[str, Any] = {}
    all_reg = regression_metrics(data["finish_position"], data[position_column])
    metrics.update(
        {
            "mae": all_reg["mae"],
            "rmse": all_reg["rmse"],
            "r2": all_reg["r2"],
            "n_rows": all_reg["n"],
        }
    )
    classified = data[_classified_mask(data)]
    if not classified.empty:
        classified_reg = regression_metrics(classified["finish_position"], classified[position_column])
        metrics["mae_classified"] = classified_reg["mae"]
        metrics["rmse_classified"] = classified_reg["rmse"]
    else:
        metrics["mae_classified"] = float("nan")
        metrics["rmse_classified"] = float("nan")

    spearman, ndcg, pairwise, top1, top3 = [], [], [], [], []
    for _, group in data.groupby("race_id", sort=False):
        group = group.sort_values(position_column)
        classified_group = group[_classified_mask(group)]
        if len(classified_group) >= 3:
            spearman.append(
                classified_group[position_column].corr(
                    classified_group["finish_position"], method="spearman"
                )
            )
            relevance = ranking_relevance(classified_group).to_numpy()
            order = np.argsort(classified_group[position_column].to_numpy())
            ndcg.append(_ndcg_at_k(relevance, order, min(10, len(order))))
            pairwise.append(
                _pairwise_accuracy(
                    classified_group["finish_position"].to_numpy(),
                    classified_group[position_column].to_numpy(),
                )
            )
        actual_winner = group[group["win"] == 1]["driver_id"].astype(str).tolist()
        predicted_winner = group.iloc[0]["driver_id"] if not group.empty else None
        top1.append(float(bool(actual_winner) and predicted_winner in actual_winner))
        actual_podium = set(group[group["podium"] == 1]["driver_id"].astype(str))
        predicted_podium = set(group.head(3)["driver_id"].astype(str))
        top3.append(len(actual_podium & predicted_podium) / 3.0 if actual_podium else float("nan"))

    metrics.update(
        {
            "spearman": float(np.nanmean(spearman)) if spearman else float("nan"),
            "ndcg_at_10": float(np.nanmean(ndcg)) if ndcg else float("nan"),
            "ranking_accuracy": float(np.nanmean(pairwise)) if pairwise else float("nan"),
            "top1_accuracy": float(np.nanmean(top1)) if top1 else float("nan"),
            "podium_precision_at_3": float(np.nanmean(top3)) if top3 else float("nan"),
            "n_races": int(len(top1)),
        }
    )
    return metrics


def driver_performance_metrics(df: pd.DataFrame) -> dict[str, Any]:
    data = df[df["finish_position"].notna()].copy()
    if data.empty:
        return {}
    finish = regression_metrics(data["finish_position"], data["expected_finish"])
    points = regression_metrics(data["total_points"], data["expected_points"])
    return {
        "finish_mae": finish["mae"],
        "finish_rmse": finish["rmse"],
        "finish_r2": finish["r2"],
        "points_mae": points["mae"],
        "points_rmse": points["rmse"],
        "points_r2": points["r2"],
        "n_rows": finish["n"],
    }


def constructor_performance_metrics(df: pd.DataFrame) -> dict[str, Any]:
    data = df[df["team_total_points"].notna()].copy()
    if data.empty:
        return {}
    points = regression_metrics(data["team_total_points"], data["expected_team_points"])
    best = regression_metrics(data["team_best_finish"], data["expected_team_best_finish"])
    return {
        "points_mae": points["mae"],
        "points_rmse": points["rmse"],
        "points_r2": points["r2"],
        "best_finish_mae": best["mae"],
        "best_finish_rmse": best["rmse"],
        "best_finish_r2": best["r2"],
        "n_rows": points["n"],
    }


def probability_from_score(
    df: pd.DataFrame,
    score_column: str,
    *,
    group_column: str = "race_id",
    temperature: float = 1.0,
) -> pd.Series:
    """Convert a per-driver score into a normalised within-race probability."""
    out = pd.Series(0.0, index=df.index, dtype=float)
    for _, group in df.groupby(group_column, sort=False):
        scores = pd.to_numeric(group[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(scores) == 0:
            continue
        std = float(np.std(scores))
        standardised = (scores - float(np.mean(scores))) / std if std > 1e-9 else np.zeros_like(scores)
        logits = standardised / max(temperature, 1e-6)
        logits -= np.max(logits)
        exp = np.exp(logits)
        out.loc[group.index] = exp / exp.sum()
    return out
