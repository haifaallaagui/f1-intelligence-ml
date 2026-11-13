"""Model explanations with SHAP and an explicit, honest fallback.

No explanation is fabricated.  If SHAP cannot explain a model (for example an
unsupported ranker objective), the module returns global feature importance and
an explicit ``method``/``note`` telling the caller that local contributions are
unavailable.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.base import RankerWrapper, model_feature_importance
from src.preprocessing.preprocessing import clean_feature_frame
from src.utils import get_logger

LOG = get_logger("explainability.shap")


def _unwrap(estimator: Any) -> tuple[Any, Any, Any]:
    """Return (imputer, scaler, core_model) for the supported estimator wrappers."""
    if isinstance(estimator, RankerWrapper):
        return estimator.imputer, None, estimator.model
    if hasattr(estimator, "named_steps"):
        return (
            estimator.named_steps.get("imputer"),
            estimator.named_steps.get("scaler"),
            estimator.named_steps.get("model"),
        )
    return None, None, estimator


def _transform_input(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    imputer, scaler, _ = _unwrap(estimator)
    values = X
    if imputer is not None:
        values = imputer.transform(values)
    else:
        values = values.fillna(0.0).to_numpy()
    if scaler is not None:
        values = scaler.transform(values)
    return np.asarray(values, dtype=float)


def _normalise_shap_values(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        chosen = values[1] if len(values) > 1 else values[0]
        return np.asarray(chosen)
    array = np.asarray(values)
    if array.ndim == 3:
        # (rows, features, classes) or (classes, rows, features)
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[0] == 2:
            array = array[1]
        else:
            array = array[:, :, 0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[-1] != n_features:
        raise ValueError(f"SHAP shape {array.shape} does not match {n_features} features")
    return array


def explain_predictions(
    estimator: Any,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_local_rows: int = 25,
    top_n_features: int = 6,
) -> dict[str, Any]:
    columns = list(feature_columns)
    X = clean_feature_frame(features, columns)
    X_imputed = _transform_input(estimator, X)
    _, _, core_model = _unwrap(estimator)
    method = None
    shap_values = None
    try:
        import shap  # type: ignore

        try:
            explainer = shap.TreeExplainer(core_model)
            raw = explainer.shap_values(X_imputed)
            shap_values = _normalise_shap_values(raw, len(columns))
            method = "shap_tree"
        except Exception:
            if hasattr(core_model, "coef_"):
                explainer = shap.LinearExplainer(core_model, X_imputed)
                raw = explainer.shap_values(X_imputed)
                shap_values = _normalise_shap_values(raw, len(columns))
                method = "shap_linear"
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - fallback is explicit
        LOG.info("SHAP unavailable for this model: %s", exc)
        importance = model_feature_importance(core_model, columns)
        if importance.empty:
            importance = model_feature_importance(estimator, columns)
        global_importance = [
            {"feature": row["feature"], "mean_abs_shap": float(row["importance"])}
            for _, row in importance.head(len(columns)).iterrows()
        ]
        return {
            "method": "feature_importance_fallback",
            "note": (
                "SHAP could not explain this model. Only global model feature importance is "
                "reported; no local contributions are invented."
            ),
            "global_importance": global_importance,
            "local": [],
        }

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_importance = [
        {"feature": columns[index], "mean_abs_shap": float(mean_abs[index])}
        for index in order
    ]

    local: list[dict[str, Any]] = []
    for row_index in range(min(int(max_local_rows), len(features))):
        contributions = shap_values[row_index]
        top_indices = np.argsort(np.abs(contributions))[::-1][: int(top_n_features)]
        local.append(
            {
                "row_index": int(row_index),
                "race_id": str(features.iloc[row_index].get("race_id")),
                "driver_id": str(features.iloc[row_index].get("driver_id")),
                "driver_name": features.iloc[row_index].get("driver_name"),
                "top_features": [
                    {
                        "feature": columns[index],
                        "contribution": float(contributions[index]),
                        "value": float(X_imputed[row_index, index]),
                    }
                    for index in top_indices
                ],
            }
        )
    return {
        "method": method,
        "note": "SHAP values are in the model's output space. Positive values push the prediction upward.",
        "global_importance": global_importance,
        "local": local,
    }
