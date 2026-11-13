"""Preprocessing, splitting and leakage-guard behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocessing import (
    add_missing_indicators,
    assert_no_leakage,
    chronological_split,
    clean_feature_frame,
    make_xy,
    ranking_relevance,
    relative_position_target,
    season_weights,
    walk_forward_folds,
)


def test_chronological_split_uses_explicit_windows(driver_frame):
    split = chronological_split(
        driver_frame,
        train_seasons=(2016, 2016),
        validation_seasons=(2017, 2017),
        test_seasons=(2018, 2018),
    )
    assert set(split.train["season"]) == {2016}
    assert set(split.validation["season"]) == {2017}
    assert set(split.test["season"]) == {2018}
    # No temporal overlap.
    assert split.train["season"].max() < split.validation["season"].min()
    assert split.validation["season"].max() < split.test["season"].min()


def test_chronological_split_rejects_empty_windows(driver_frame):
    with pytest.raises(ValueError):
        chronological_split(
            driver_frame,
            train_seasons=(1990, 1991),
            validation_seasons=(2017, 2017),
            test_seasons=(2018, 2018),
        )


def test_walk_forward_folds_expand_without_looking_ahead(driver_frame):
    folds = walk_forward_folds(driver_frame, min_train_seasons=1)
    assert folds
    for train, validation in folds:
        assert train["season"].max() < validation["season"].min()
    # The first fold trains on the earliest season only.
    assert set(folds[0][0]["season"]) == {2016}


def test_assert_no_leakage_flags_target_columns():
    with pytest.raises(ValueError, match="Leakage guard"):
        assert_no_leakage(["form_score", "finish_position"])
    # Historical aggregates are allowed; the guard only rejects exact targets.
    assert_no_leakage(["form_score", "driver_last5_avg_finish"])


def test_clean_feature_frame_coerces_and_removes_infinities():
    frame = pd.DataFrame(
        {
            "a": ["1.5", "bad", None],
            "b": [1.0, np.inf, -np.inf],
        }
    )
    cleaned = clean_feature_frame(frame, ["a", "b"])
    assert cleaned["a"].tolist()[0] == 1.5
    assert np.isnan(cleaned["a"].tolist()[1])
    assert cleaned["b"].isna().sum() == 2


def test_missing_indicators_only_for_columns_with_missing_values():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    out = add_missing_indicators(frame, ["a", "b"])
    assert "a_missing" in out.columns
    assert "b_missing" not in out.columns
    assert out["a_missing"].tolist() == [0, 1]


def test_make_xy_drops_missing_targets_and_keeps_groups(driver_frame):
    frame = driver_frame.copy()
    frame.loc[frame.index[:3], "win"] = np.nan
    X, y, groups = make_xy(frame, ["form_score", "grid_position"], "win")
    assert len(X) == len(y) == len(groups) == len(frame) - 3
    assert y.notna().all()
    assert set(groups.unique()) == set(frame["race_id"].unique())


def test_make_xy_missing_target_raises(driver_frame):
    with pytest.raises(KeyError):
        make_xy(driver_frame, ["form_score"], "not_a_target")


def test_season_weights_favour_recent_seasons(driver_frame):
    weights = season_weights(driver_frame, decay=0.5)
    frame = driver_frame.assign(weight=weights)
    early = frame[frame["season"] == 2016]["weight"].iloc[0]
    late = frame[frame["season"] == 2018]["weight"].iloc[0]
    assert late > early


def test_relative_and_ranking_targets(driver_frame):
    relative = relative_position_target(driver_frame)
    assert relative.between(0, 1).all()
    relevance = ranking_relevance(driver_frame, max_relevance=20)
    assert relevance.min() >= 0
    assert relevance.max() <= 20
    # A better finish always gets a higher relevance.
    ordered = driver_frame.assign(relevance=relevance).sort_values("finish_position")
    assert ordered["relevance"].is_monotonic_decreasing
