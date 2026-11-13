"""Full-field classification prediction helpers."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.models.classification_model import ClassificationModel


def predict_classification(
    model: ClassificationModel,
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    n_samples: int = 2000,
    max_positions: int = 20,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    if model is None:
        raise RuntimeError("Classification model is not available")
    predictions = model.predict_race(features, feature_columns)
    matrices = model.position_probability_matrix(
        features,
        feature_columns,
        n_samples=n_samples,
        max_positions=max_positions,
    )
    expected = model.expected_positions(features, feature_columns, n_samples=n_samples)
    predictions = predictions.merge(
        expected, on=["race_id", "driver_id"], how="left"
    )
    return predictions, matrices, expected


def classification_summary(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for race_id, group in predictions.groupby("race_id", sort=False):
        group = group.sort_values("predicted_position")
        rows.append(
            {
                "race_id": race_id,
                "order": [
                    {
                        "position": int(row["predicted_position"]),
                        "driver_id": row["driver_id"],
                        "driver_name": row.get("driver_name"),
                        "position_score": float(row.get("position_score", 0.0)),
                    }
                    for _, row in group.iterrows()
                ],
            }
        )
    return rows
