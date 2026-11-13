"""Podium prediction helpers."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.models.podium_model import PodiumModel


def predict_podium(
    model: PodiumModel,
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model is None:
        raise RuntimeError("Podium model is not available")
    out = model.predict_race(features, feature_columns)
    selected = PodiumModel.select_podium(out)
    return out, selected


def podium_summary(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for race_id, group in predictions.groupby("race_id", sort=False):
        group = group.sort_values("podium_probability", ascending=False)
        rows.append(
            {
                "race_id": race_id,
                "predicted_podium": [
                    {
                        "driver_id": row["driver_id"],
                        "driver_name": row.get("driver_name"),
                        "probability": float(row["podium_probability"]),
                    }
                    for _, row in group.head(3).iterrows()
                ],
            }
        )
    return rows
