"""Metric correctness on hand-checkable predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.metrics import (
    classification_metrics,
    constructor_performance_metrics,
    driver_performance_metrics,
    podium_metrics,
    probability_from_score,
    regression_metrics,
    winner_metrics,
)


def _race_frame(order: list[str], race_id: str = "2024_01") -> pd.DataFrame:
    n = len(order)
    rows = []
    for position, driver in enumerate(order, start=1):
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver,
                "finish_position": position,
                "classified": True,
                "win": int(position == 1),
                "podium": int(position <= 3),
                "field_size": n,
            }
        )
    return pd.DataFrame(rows)


def test_regression_metrics_perfect_and_constant():
    perfect = regression_metrics([1, 2, 3], [1, 2, 3])
    assert perfect["mae"] == 0.0
    assert perfect["rmse"] == 0.0
    assert perfect["r2"] == 1.0
    # A constant target has no defined R2; it must not be invented.
    constant = regression_metrics([1, 1, 1], [1, 1, 1])
    assert np.isnan(constant["r2"])
    empty = regression_metrics([], [])
    assert empty["n"] == 0


def test_winner_metrics_perfect_and_ranked():
    frame = _race_frame(["a", "b", "c", "d"])
    frame["winner_probability"] = [0.7, 0.1, 0.1, 0.1]
    metrics = winner_metrics(frame)
    assert metrics["accuracy"] == 1.0
    assert metrics["top3_hit_rate"] == 1.0
    assert metrics["n_races"] == 1
    assert metrics["log_loss"] < 0.5
    assert 0 <= metrics["brier"] <= 1


def test_winner_metrics_wrong_winner():
    frame = _race_frame(["a", "b", "c", "d"])
    frame["winner_probability"] = [0.1, 0.7, 0.1, 0.1]
    metrics = winner_metrics(frame)
    assert metrics["accuracy"] == 0.0
    assert metrics["top3_hit_rate"] == 1.0  # a is still in the predicted top 3


def test_podium_metrics_perfect_selection():
    frame = _race_frame(["a", "b", "c", "d", "e"])
    frame["podium_probability"] = [0.6, 0.5, 0.4, 0.1, 0.1]
    metrics = podium_metrics(frame)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["exact_set_accuracy"] == 1.0


def test_classification_metrics_perfect_ranking():
    frame = _race_frame(["a", "b", "c", "d", "e"])
    frame["predicted_position"] = [1, 2, 3, 4, 5]
    metrics = classification_metrics(frame)
    assert metrics["mae"] == 0.0
    assert metrics["mae_classified"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["ranking_accuracy"] == 1.0
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["podium_precision_at_3"] == 1.0


def test_classification_metrics_reverse_ranking():
    frame = _race_frame(["a", "b", "c", "d", "e"])
    frame["predicted_position"] = [5, 4, 3, 2, 1]
    metrics = classification_metrics(frame)
    assert metrics["spearman"] == pytest.approx(-1.0)
    assert metrics["ranking_accuracy"] == 0.0
    assert metrics["top1_accuracy"] == 0.0


def test_classification_metrics_uses_classified_position_column():
    # The real dataset stores classified_position (NaN for retired cars), not a
    # boolean "classified" column.  The metric must exclude unclassified cars
    # from the classified-only MAE without dropping them from the full MAE.
    frame = _race_frame(["a", "b", "c", "d", "e"])
    frame["predicted_position"] = [1, 2, 3, 4, 5]
    frame["classified_position"] = [1, 2, 3, np.nan, np.nan]
    metrics = classification_metrics(frame)
    assert metrics["mae"] == 0.0
    assert metrics["mae_classified"] == 0.0
    assert metrics["n_rows"] == 5


def test_driver_performance_metrics():
    frame = pd.DataFrame(
        {
            "race_id": ["r"] * 3,
            "driver_id": ["a", "b", "c"],
            "finish_position": [1, 2, 3],
            "expected_finish": [1.2, 1.9, 3.1],
            "total_points": [25, 18, 15],
            "expected_points": [24.0, 17.0, 16.0],
        }
    )
    metrics = driver_performance_metrics(frame)
    assert metrics["n_rows"] == 3
    assert metrics["finish_mae"] < 0.5
    assert metrics["points_mae"] == pytest.approx(1.0)


def test_constructor_performance_metrics():
    frame = pd.DataFrame(
        {
            "race_id": ["r", "r"],
            "constructor_id": ["x", "y"],
            "team_total_points": [40.0, 20.0],
            "expected_team_points": [38.0, 22.0],
            "team_best_finish": [1.0, 3.0],
            "expected_team_best_finish": [1.5, 3.2],
        }
    )
    metrics = constructor_performance_metrics(frame)
    assert metrics["points_mae"] == 2.0
    assert metrics["best_finish_mae"] == pytest.approx(0.35)


def test_probability_from_score_is_normalised_within_race():
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1", "r2", "r2"],
            "score": [3.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    probability = probability_from_score(frame, "score")
    assert probability.groupby(frame["race_id"]).sum().round(6).tolist() == [1.0, 1.0]
    # Higher scores receive higher probability.
    assert probability.iloc[0] > probability.iloc[1] > probability.iloc[2]
    # Equal scores split the mass evenly.
    assert np.isclose(probability.iloc[3], 0.5)
