"""Prediction helpers for winner, podium, full classification and championships."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.championship_model import ChampionshipModel, RaceSimulationInput
from src.models.classification_model import ClassificationModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from src.prediction.championship_prediction import project_championship
from src.prediction.classification_prediction import (
    classification_summary,
    predict_classification,
)
from src.prediction.podium_prediction import podium_summary, predict_podium
from src.prediction.winner_prediction import predict_winner, winner_summary
from tests.conftest import FEATURE_COLUMNS

CLASSIFICATION_FEATURES = ["form_score", "grid_position", "quali_position", "field_size"]


@pytest.fixture
def winner_model(driver_frame):
    return WinnerModel(
        "logistic_regression", {"C": 0.5}, "after_qualifying", FEATURE_COLUMNS
    ).fit(driver_frame, FEATURE_COLUMNS)


@pytest.fixture
def podium_model(driver_frame):
    return PodiumModel(
        "logistic_regression", {"C": 0.5}, "after_qualifying", FEATURE_COLUMNS
    ).fit(driver_frame, FEATURE_COLUMNS)


@pytest.fixture
def classification_model(driver_frame):
    return ClassificationModel(
        "hist_gradient_boosting",
        {"max_iter": 30, "learning_rate": 0.1},
        "after_qualifying",
        CLASSIFICATION_FEATURES,
    ).fit(driver_frame, CLASSIFICATION_FEATURES)


def test_predict_winner_ranks_each_race(winner_model, driver_frame):
    predictions = predict_winner(winner_model, driver_frame, FEATURE_COLUMNS)

    assert predictions.groupby("race_id")["winner_rank"].min().eq(1).all()
    assert predictions["winner_probability"].between(0, 1).all()
    for _, group in predictions.groupby("race_id"):
        assert group["winner_rank"].tolist() == list(range(1, len(group) + 1))
        assert np.isclose(group["winner_probability"].sum(), 1.0)

    summary = winner_summary(predictions)
    assert len(summary) == predictions["race_id"].nunique()
    assert all(len(item["top_5"]) == 5 for item in summary)
    assert summary[0]["predicted_winner"] == predictions.iloc[0]["driver_id"]


def test_predict_winner_requires_model(driver_frame):
    with pytest.raises(RuntimeError):
        predict_winner(None, driver_frame, FEATURE_COLUMNS)


def test_predict_podium_selects_three_per_race(podium_model, driver_frame):
    predictions, selected = predict_podium(podium_model, driver_frame, FEATURE_COLUMNS)

    assert predictions["podium_probability"].between(0, 1).all()
    assert (selected.groupby("race_id").size() == 3).all()
    for _, group in selected.groupby("race_id"):
        assert group["podium_probability"].is_monotonic_decreasing

    summary = podium_summary(predictions)
    assert all(len(item["predicted_podium"]) == 3 for item in summary)


def test_predict_podium_requires_model(driver_frame):
    with pytest.raises(RuntimeError):
        predict_podium(None, driver_frame, FEATURE_COLUMNS)


def test_predict_classification_returns_order_and_matrices(classification_model, driver_frame):
    predictions, matrices, expected = predict_classification(
        classification_model,
        driver_frame,
        CLASSIFICATION_FEATURES,
        n_samples=250,
        max_positions=6,
    )

    assert set(matrices) == set(driver_frame["race_id"].astype(str).unique())
    assert "expected_position" in predictions.columns
    assert predictions["expected_position"].between(1, 6).all()
    for _, group in predictions.groupby("race_id"):
        assert sorted(group["predicted_position"].tolist()) == list(range(1, len(group) + 1))

    summary = classification_summary(predictions)
    assert len(summary) == predictions["race_id"].nunique()
    assert [row["position"] for row in summary[0]["order"]] == list(
        range(1, len(summary[0]["order"]) + 1)
    )
    assert len(expected) == len(predictions)


def test_predict_classification_requires_model(driver_frame):
    with pytest.raises(RuntimeError):
        predict_classification(None, driver_frame, CLASSIFICATION_FEATURES)


def test_championship_simulation_produces_a_distribution():
    races = [
        RaceSimulationInput(
            race_id="2024_01",
            driver_ids=["a", "b", "c"],
            scores=[2.0, 1.0, 0.0],
            driver_constructors={"a": "t1", "b": "t1", "c": "t2"},
        ),
        RaceSimulationInput(
            race_id="2024_02",
            driver_ids=["a", "b", "c"],
            scores=[1.5, 1.0, 0.5],
            sprint=True,
            driver_constructors={"a": "t1", "b": "t1", "c": "t2"},
        ),
    ]
    model = ChampionshipModel(n_simulations=300, seed=7)
    result = model.simulate(
        2024,
        races,
        current_driver_points={"a": 10.0, "b": 5.0, "c": 3.0},
        current_constructor_points={"t1": 15.0, "t2": 3.0},
        driver_names={"a": "A", "b": "B", "c": "C"},
        constructor_names={"t1": "Team 1", "t2": "Team 2"},
    )

    drivers = result["drivers"]
    constructors = result["constructors"]
    assert np.isclose(drivers["championship_probability"].sum(), 1.0)
    assert np.isclose(constructors["championship_probability"].sum(), 1.0)
    assert drivers.iloc[0]["championship_probability"] >= drivers.iloc[-1]["championship_probability"]
    assert (drivers["expected_final_points"] >= drivers["current_points"]).all()
    assert result["metadata"]["n_simulations"] == 300
    assert result["metadata"]["sprint_races"] == ["2024_02"]


class _FakeEngineer:
    """Just enough surface for the completed-season branch of the projection."""

    def __init__(self):
        self.results = pd.DataFrame(
            [
                {
                    "race_id": "2024_01",
                    "season": 2024,
                    "driver_id": "a",
                    "driver_name": "A",
                    "constructor_id": "t1",
                    "constructor_name": "Team 1",
                    "total_points": 25.0,
                },
                {
                    "race_id": "2024_01",
                    "season": 2024,
                    "driver_id": "b",
                    "driver_name": "B",
                    "constructor_id": "t1",
                    "constructor_name": "Team 1",
                    "total_points": 18.0,
                },
                {
                    "race_id": "2024_01",
                    "season": 2024,
                    "driver_id": "c",
                    "driver_name": "C",
                    "constructor_id": "t2",
                    "constructor_name": "Team 2",
                    "total_points": 15.0,
                },
            ]
        )
        self.races = pd.DataFrame(
            [{"race_id": "2024_01", "season": 2024, "_order": 0}]
        )
        self.race_order = {"2024_01": 0}

    def build_prediction_features(self, race_id, mode="after_qualifying"):
        raise AssertionError("No remaining race should request features")


def test_complete_season_projection_uses_actual_points():
    engineer = _FakeEngineer()
    projection = project_championship(
        engineer,
        classification_model=None,
        feature_columns=[],
        season=2024,
        mode="after_qualifying",
    )

    metadata = projection["metadata"]
    assert metadata["complete_season"] is True
    assert metadata["n_simulations"] == 0
    assert metadata["races_simulated"] == []

    drivers = projection["drivers"]
    assert drivers.iloc[0]["driver_id"] == "a"
    assert drivers.iloc[0]["championship_probability"] == 1.0
    assert drivers["championship_probability"].sum() == 1.0
    assert projection["constructors"].iloc[0]["constructor_id"] == "t1"
