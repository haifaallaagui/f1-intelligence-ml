"""Winner prediction helpers."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.models.winner_model import WinnerModel


def predict_winner(
    model: WinnerModel,
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if model is None:
        raise RuntimeError("Winner model is not available")
    out = model.predict_race(features, feature_columns)
    out["winner_rank"] = out.groupby("race_id")["winner_probability"].rank(
        method="first", ascending=False
    )
    return out.sort_values(["race_id", "winner_probability"], ascending=[True, False]).reset_index(
        drop=True
    )


def winner_summary(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for race_id, group in predictions.groupby("race_id", sort=False):
        top = group.sort_values("winner_probability", ascending=False).head(5)
        rows.append(
            {
                "race_id": race_id,
                "predicted_winner": top.iloc[0]["driver_id"] if not top.empty else None,
                "winner_probability": float(top.iloc[0]["winner_probability"]) if not top.empty else None,
                "top_5": [
                    {
                        "driver_id": row["driver_id"],
                        "driver_name": row.get("driver_name"),
                        "probability": float(row["winner_probability"]),
                    }
                    for _, row in top.iterrows()
                ],
            }
        )
    return rows
