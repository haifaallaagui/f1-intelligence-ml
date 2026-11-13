"""Model fitting, coherent probabilities and persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import (
    _contiguous_group_sizes,
    load_bundle,
    make_classifier,
    make_regressor,
    model_feature_importance,
    normalize_probabilities_by_group,
    softmax,
)
from src.models.classification_model import ClassificationModel, plackett_luce_marginals
from src.models.constructor_model import ConstructorPerformanceModel
from src.models.driver_model import DriverPerformanceModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from tests.conftest import FEATURE_COLUMNS

CLASSIFICATION_FEATURES = ["form_score", "grid_position", "quali_position", "field_size"]


@pytest.fixture
def winner_model(driver_frame):
    return WinnerModel("logistic_regression", {"C": 0.5}, "after_qualifying", FEATURE_COLUMNS).fit(
        driver_frame, FEATURE_COLUMNS
    )


def test_winner_probabilities_are_normalised_per_race(winner_model, driver_frame):
    predictions = winner_model.predict_race(driver_frame, FEATURE_COLUMNS)
    totals = predictions.groupby("race_id")["winner_probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    assert predictions["winner_probability"].between(0, 1).all()
    # Raw classifier probabilities are not forced to sum to one.
    raw_totals = predictions.groupby("race_id")["winner_probability_raw"].sum()
    assert not np.allclose(raw_totals.to_numpy(), 1.0)


def test_winner_model_requires_two_classes(driver_frame):
    frame = driver_frame.copy()
    frame["win"] = 0
    with pytest.raises(ValueError):
        WinnerModel("logistic_regression", {}, "after_qualifying", FEATURE_COLUMNS).fit(
            frame, FEATURE_COLUMNS
        )


def test_winner_model_save_and_load(tmp_path, winner_model, driver_frame):
    path = tmp_path / "winner" / "best.joblib"
    winner_model.save(path, {"dataset_hash": "abc"})
    loaded = WinnerModel.load(path)
    assert loaded.algorithm == "logistic_regression"
    assert loaded.metadata["dataset_hash"] == "abc"
    before = winner_model.predict_proba(driver_frame, FEATURE_COLUMNS)
    after = loaded.predict_proba(driver_frame, FEATURE_COLUMNS)
    assert np.allclose(before, after)


def test_winner_feature_importance(winner_model):
    importance = winner_model.feature_importance()
    assert not importance.empty
    assert set(importance.columns) == {"feature", "importance"}
    assert importance["importance"].notna().all()


def test_podium_probabilities_are_bounded(driver_frame):
    model = PodiumModel(
        "logistic_regression", {"C": 0.5}, "after_qualifying", FEATURE_COLUMNS
    ).fit(driver_frame, FEATURE_COLUMNS)
    predictions = model.predict_race(driver_frame, FEATURE_COLUMNS)
    assert predictions["podium_probability"].between(0, 1).all()
    totals = predictions.groupby("race_id")["podium_probability"].sum()
    assert (totals <= 3.0 + 1e-9).all()
    selected = PodiumModel.select_podium(predictions)
    assert (selected.groupby("race_id").size() == 3).all()


def test_classification_model_regression_scores_and_coherent_matrix(driver_frame):
    model = ClassificationModel(
        "hist_gradient_boosting",
        {"max_iter": 40, "learning_rate": 0.1},
        "after_qualifying",
        CLASSIFICATION_FEATURES,
    ).fit(driver_frame, CLASSIFICATION_FEATURES)
    predictions = model.predict_race(driver_frame, CLASSIFICATION_FEATURES)
    for _, group in predictions.groupby("race_id"):
        assert sorted(group["predicted_position"].tolist()) == list(range(1, len(group) + 1))

    matrices = model.position_probability_matrix(
        driver_frame, CLASSIFICATION_FEATURES, n_samples=300, max_positions=6
    )
    assert matrices
    for matrix in matrices.values():
        assert matrix.shape[0] == 6
        assert np.allclose(matrix.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
        assert (matrix.to_numpy() >= 0).all()

    expected = model.expected_positions(
        driver_frame, CLASSIFICATION_FEATURES, n_samples=300
    )
    assert expected["expected_position"].between(1, 6).all()


def test_classification_model_save_and_load(tmp_path, driver_frame):
    model = ClassificationModel(
        "hist_gradient_boosting",
        {"max_iter": 20},
        "before_qualifying",
        CLASSIFICATION_FEATURES,
    ).fit(driver_frame, CLASSIFICATION_FEATURES)
    path = tmp_path / "classification.joblib"
    model.save(path, {"dataset_hash": "xyz"})
    loaded = ClassificationModel.load(path)
    assert loaded.mode == "before_qualifying"
    assert loaded.temperature == model.temperature
    assert np.allclose(
        model.predict_scores(driver_frame, CLASSIFICATION_FEATURES),
        loaded.predict_scores(driver_frame, CLASSIFICATION_FEATURES),
    )


def test_plackett_luce_marginals_are_a_valid_distribution():
    marginals = plackett_luce_marginals([3.0, 1.0, 0.0], n_samples=500, temperature=1.0)
    assert marginals.shape == (3, 3)
    assert np.allclose(marginals.sum(axis=1), 1.0)
    assert np.allclose(marginals.sum(axis=0), 1.0)
    # The strongest driver is most likely to win.
    assert marginals[0, 0] > marginals[1, 0] > marginals[2, 0]


def test_plackett_luce_handles_equal_scores():
    marginals = plackett_luce_marginals([1.0, 1.0, 1.0], n_samples=400)
    assert np.allclose(marginals, 1 / 3, atol=0.05)


def test_driver_performance_model(driver_frame, tmp_path):
    features = ["form_score", "grid_position", "field_size"]
    model = DriverPerformanceModel(
        "hist_gradient_boosting", {"max_iter": 30}, "after_qualifying", features
    ).fit(driver_frame, features)
    predictions = model.predict(driver_frame, features)
    assert predictions["expected_finish"].between(1, 6).all()
    assert (predictions["expected_points"] >= 0).all()
    assert predictions["predicted_finish_position"].between(1, 6).all()

    path = tmp_path / "driver.joblib"
    model.save(path, {"dataset_hash": "d"})
    loaded = DriverPerformanceModel.load(path)
    assert np.allclose(
        predictions["expected_finish"],
        loaded.predict(driver_frame, features)["expected_finish"],
    )


def test_constructor_performance_model(constructor_frame, tmp_path):
    features = ["form_score", "constructor_championship_position", "field_size"]
    model = ConstructorPerformanceModel(
        "hist_gradient_boosting", {"max_iter": 30}, features
    ).fit(constructor_frame, features)
    predictions = model.predict(constructor_frame, features)
    assert (predictions["expected_team_points"] >= 0).all()
    assert (predictions["expected_team_best_finish"] >= 1).all()
    for _, group in predictions.groupby("race_id"):
        assert sorted(group["predicted_team_position"].tolist()) == list(
            range(1, len(group) + 1)
        )

    path = tmp_path / "constructor.joblib"
    model.save(path, {"dataset_hash": "c"})
    loaded = ConstructorPerformanceModel.load(path)
    assert np.allclose(
        predictions["expected_team_points"],
        loaded.predict(constructor_frame, features)["expected_team_points"],
    )


def test_contiguous_group_sizes():
    assert _contiguous_group_sizes(["a", "a", "b", "c", "c", "c"]).tolist() == [2, 1, 3]
    assert _contiguous_group_sizes([]).tolist() == []


def test_probability_helpers():
    frame = pd.DataFrame(
        {"race_id": ["a", "a", "b", "b"], "p": [1.0, 3.0, 2.0, 2.0]}
    )
    normalised = normalize_probabilities_by_group(frame, "p")
    assert np.allclose(normalised.tolist(), [0.25, 0.75, 0.5, 0.5])
    assert np.isclose(softmax([0.0, 0.0]).sum(), 1.0)
    assert softmax([1.0, 0.0])[0] > softmax([1.0, 0.0])[1]


def test_factories_and_bundle_helpers(tmp_path):
    classifier = make_classifier("logistic_regression")
    assert hasattr(classifier, "predict_proba")
    regressor = make_regressor("hist_gradient_boosting")
    assert hasattr(regressor, "predict")
    with pytest.raises(ValueError):
        make_classifier("does_not_exist")
    with pytest.raises(ValueError):
        make_regressor("does_not_exist")

    from src.models.base import save_bundle

    path = tmp_path / "bundle.joblib"
    save_bundle(path, {"x": 1}, {"task": "test"})
    assert load_bundle(path)["metadata"]["task"] == "test"
    assert path.with_suffix(".metadata.json").exists()
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")


def test_model_feature_importance_for_pipeline(winner_model):
    importance = model_feature_importance(winner_model.model, FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert importance["importance"].is_monotonic_decreasing


def test_model_feature_importance_maps_surviving_columns():
    # SimpleImputer drops all-NaN columns, so the estimator's importance vector
    # is shorter than the requested feature list.  Names must follow the
    # surviving columns instead of discarding the whole importance table.
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "all_nan": [np.nan] * 5,
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = make_regressor("random_forest", {"n_estimators": 10})
    model.fit(X, y)

    importance = model_feature_importance(model, ["a", "all_nan", "c"])

    assert set(importance["feature"]) == {"a", "c"}
    assert len(importance) == 2
    assert importance["importance"].is_monotonic_decreasing
