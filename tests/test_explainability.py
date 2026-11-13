"""SHAP explanation normalisation and the explicit importance fallback."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import (
    _normalise_shap_values,
    explain_predictions,
)
from src.models.base import make_classifier
from tests.conftest import FEATURE_COLUMNS


class _TreeExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X):
        return np.ones_like(np.asarray(X), dtype=float)


class _FailingExplainer:
    def __init__(self, *args, **kwargs):
        pass

    def shap_values(self, X):
        raise RuntimeError("unsupported model")


@pytest.fixture
def fitted_classifier(driver_frame):
    estimator = make_classifier("logistic_regression", {"C": 0.5})
    estimator.fit(driver_frame[FEATURE_COLUMNS], driver_frame["win"].astype(int))
    return estimator, driver_frame


def test_normalise_shap_values_handles_lists_and_3d_arrays():
    rows, features = 4, 3
    first = np.zeros((rows, features))
    second = np.arange(rows * features, dtype=float).reshape(rows, features)
    assert np.array_equal(_normalise_shap_values([first, second], features), second)

    stacked = np.stack([first, second], axis=-1)  # (rows, features, classes)
    assert np.array_equal(_normalise_shap_values(stacked, features), second)

    class_first = np.stack([first, second], axis=0)  # (classes, rows, features)
    assert np.array_equal(_normalise_shap_values(class_first, features), second)

    flat = np.arange(features, dtype=float)
    assert _normalise_shap_values(flat, features).shape == (1, features)

    with pytest.raises(ValueError):
        _normalise_shap_values(np.zeros((rows, features + 1)), features)


def test_explain_predictions_uses_shap_when_available(fitted_classifier, monkeypatch):
    estimator, frame = fitted_classifier
    fake_shap = types.SimpleNamespace(TreeExplainer=_TreeExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    result = explain_predictions(
        estimator, frame, FEATURE_COLUMNS, max_local_rows=3, top_n_features=2
    )

    assert result["method"] == "shap_tree"
    assert len(result["global_importance"]) == len(FEATURE_COLUMNS)
    assert len(result["local"]) == 3
    assert all(len(item["top_features"]) == 2 for item in result["local"])
    assert result["local"][0]["driver_id"] == str(frame.iloc[0]["driver_id"])


def test_explain_predictions_falls_back_without_fabricating_local_values(
    fitted_classifier, monkeypatch
):
    estimator, frame = fitted_classifier
    fake_shap = types.SimpleNamespace(
        TreeExplainer=_FailingExplainer, LinearExplainer=_FailingExplainer
    )
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    result = explain_predictions(estimator, frame, FEATURE_COLUMNS)

    assert result["method"] == "feature_importance_fallback"
    assert "no local contributions are invented" in result["note"]
    assert result["local"] == []
    assert len(result["global_importance"]) == len(FEATURE_COLUMNS)
    assert all(
        set(item) == {"feature", "mean_abs_shap"} for item in result["global_importance"]
    )


def test_explain_predictions_falls_back_when_shap_is_not_installed(
    fitted_classifier, monkeypatch
):
    estimator, frame = fitted_classifier
    monkeypatch.setitem(sys.modules, "shap", None)

    result = explain_predictions(estimator, frame, FEATURE_COLUMNS)

    assert result["method"] == "feature_importance_fallback"
    assert result["local"] == []
    assert isinstance(result["global_importance"], list)
