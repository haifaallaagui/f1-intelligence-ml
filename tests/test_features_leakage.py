"""Leakage-safe feature engineering and prediction-time feature parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.features.feature_engineering import (
    TARGET_COLUMNS,
    FeatureEngineer,
    get_constructor_feature_columns,
    get_feature_columns,
)


@pytest.fixture
def built(tmp_path, monkeypatch, processed_tables):
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "METADATA_DIR", tmp_path / "metadata")
    monkeypatch.setattr(config, "RACE_FEATURES_FILE", tmp_path / "datasets" / "race.parquet")
    monkeypatch.setattr(
        config, "CONSTRUCTOR_FEATURES_FILE", tmp_path / "datasets" / "constructor.parquet"
    )
    monkeypatch.setattr(
        config, "DATASET_METADATA_FILE", tmp_path / "datasets" / "metadata.json"
    )
    engineer = FeatureEngineer(tables=processed_tables)
    driver_df, constructor_df = engineer.build()
    return engineer, driver_df, constructor_df


def test_dataset_shape_and_constructor_aggregation(built):
    engineer, driver_df, constructor_df = built
    # 2 seasons x 3 races x 4 drivers.
    assert len(driver_df) == 24
    # 2 seasons x 3 races x 2 constructors.
    assert len(constructor_df) == 12
    assert set(driver_df["season"]) == {2016, 2017}
    assert constructor_df["team_entries"].sum() == 24
    assert (constructor_df["team_entries"] == 2).all()


def test_metadata_has_no_target_leakage(built):
    _, driver_df, _ = built
    from src.utils import load_json

    metadata = load_json(config.DATASET_METADATA_FILE)
    checks = metadata["leakage_checks"]
    assert checks["targets_not_in_base_features"] == []
    assert checks["targets_not_in_post_features"] == []
    assert checks["targets_not_in_constructor_features"] == []
    mode1 = set(metadata["feature_columns_mode_1"])
    mode2 = set(metadata["feature_columns_mode_2"])
    assert not (mode1 & set(TARGET_COLUMNS))
    assert not (mode2 & set(TARGET_COLUMNS))
    # Mode 1 must not contain any post-qualifying column.
    assert not (mode1 & set(metadata["feature_columns_post_quali"]))
    # Mode 2 is a strict superset of mode 1.
    assert mode1 <= mode2
    # All-NaN features are documented and excluded.
    dropped = metadata["dropped_all_nan_features"]
    assert isinstance(dropped, dict)
    for column in dropped["driver"]:
        assert column not in mode2
    assert metadata["rows"] == len(driver_df)


def test_features_only_use_strictly_prior_races(built):
    _, driver_df, _ = built
    first = driver_df[driver_df["race_id"] == "2016_01"].set_index("driver_id")
    second = driver_df[driver_df["race_id"] == "2016_02"].set_index("driver_id")
    # Before round 1 nobody has started.
    assert (first["driver_career_starts"] == 0).all()
    assert (first["driver_career_points"] == 0).all()
    # Before round 2 every driver has exactly one prior start and only round-1
    # points, proving the target race did not update the state first.
    assert (second["driver_career_starts"] == 1).all()
    assert second.loc["driver_a", "driver_career_points"] == 10.0
    assert second.loc["driver_b", "driver_career_points"] == 8.0
    # Season points reset at the start of a new season.
    new_season = driver_df[driver_df["race_id"] == "2017_01"].set_index("driver_id")
    assert (new_season["driver_season_points"] == 0).all()


def test_mode1_and_mode2_column_sets(built):
    from src.utils import load_json

    metadata = load_json(config.DATASET_METADATA_FILE)
    mode1 = get_feature_columns("before_qualifying")
    mode2 = get_feature_columns("after_qualifying")
    assert "grid_position" not in mode1
    assert "quali_position" not in mode1
    assert "grid_position" in mode2
    assert "quali_position" in mode2
    assert len(mode1) == len(metadata["feature_columns_mode_1"])
    assert len(mode2) == len(metadata["feature_columns_mode_2"])


def test_prediction_features_match_training_features_without_leakage(built):
    engineer, driver_df, _ = built
    features, info = engineer.build_prediction_features(
        "2016_02", mode="before_qualifying"
    )
    assert info["has_actual_results"] is True
    assert info["entries_source"] == "actual_results"
    # Post-qualifying columns exist but must be entirely missing in mode 1.
    post_columns = [
        column
        for column in features.columns
        if column in {
            "quali_position",
            "grid_position",
            "grid_position_norm",
            "sprint_position_current",
        }
    ]
    assert post_columns
    assert features[post_columns].isna().all().all()

    columns = get_feature_columns("before_qualifying")
    expected = driver_df[driver_df["race_id"] == "2016_02"].set_index("driver_id")[columns]
    actual = features.set_index("driver_id")[columns]
    expected = expected.sort_index()
    actual = actual.sort_index()
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False)


def test_mode2_uses_actual_grid_and_qualifying(built):
    engineer, _, _ = built
    features, info = engineer.build_prediction_features(
        "2016_01", mode="after_qualifying"
    )
    assert info["has_qualifying"] is True
    indexed = features.set_index("driver_id")
    # driver_d started 4th after a two-place penalty but qualified 4th.
    assert indexed.loc["driver_d", "grid_position"] == 6.0
    assert indexed.loc["driver_d", "quali_position"] == 4.0
    assert indexed.loc["driver_d", "grid_penalty"] == 2.0
    assert indexed.loc["driver_d", "grid_position_estimated"] == 0.0
    # Driver A starts from pole.
    assert indexed.loc["driver_a", "grid_position"] == 1.0
    assert indexed.loc["driver_a", "quali_position"] == 1.0


def test_prediction_features_reject_unknown_race(built):
    engineer, _, _ = built
    with pytest.raises(KeyError):
        engineer.build_prediction_features("2030_01", mode="before_qualifying")


def test_constructor_feature_columns_are_available(built):
    columns = get_constructor_feature_columns()
    assert columns
    assert "team_total_points" not in columns
    assert "team_best_finish" not in columns
    assert any(column.startswith("team_avg_driver_") for column in columns)


def test_feature_engineer_requires_processed_tables():
    with pytest.raises(RuntimeError):
        FeatureEngineer(tables={"races": pd.DataFrame(), "results": pd.DataFrame()})
