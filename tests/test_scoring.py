"""Season-aware scoring rules."""
from __future__ import annotations

import pytest

from src.scoring import (
    expected_points_from_distribution,
    fastest_lap_point_available,
    is_double_points_race,
    points_for_position,
    race_points_for_order,
    race_points_system,
    scoring_rules_description,
    sprint_points_system,
)


def test_race_points_system_eras():
    assert race_points_system(1995) == (10, 6, 4, 3, 2, 1)
    assert race_points_system(2002) == (10, 6, 4, 3, 2, 1)
    assert race_points_system(2003) == (10, 8, 6, 5, 4, 3, 2, 1)
    assert race_points_system(2009) == (10, 8, 6, 5, 4, 3, 2, 1)
    assert race_points_system(2010) == (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
    assert race_points_system(2026) == (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


def test_sprint_points_system_eras():
    assert sprint_points_system(2020) == ()
    assert sprint_points_system(2021) == (3, 2, 1)
    assert sprint_points_system(2022) == (8, 7, 6, 5, 4, 3, 2, 1)


@pytest.mark.parametrize(
    "season, position, expected",
    [
        (1995, 1, 10.0),
        (1995, 6, 1.0),
        (1995, 7, 0.0),
        (2005, 8, 1.0),
        (2015, 1, 25.0),
        (2015, 10, 1.0),
        (2015, 11, 0.0),
        (2015, 0, 0.0),
        (2015, None, 0.0),
    ],
)
def test_points_for_position(season, position, expected):
    assert points_for_position(season, position) == expected


def test_fastest_lap_bonus_window_and_top_ten_only():
    assert fastest_lap_point_available(2018) is False
    assert fastest_lap_point_available(2019) is True
    assert fastest_lap_point_available(2024) is True
    assert fastest_lap_point_available(2025) is False
    assert points_for_position(2020, 1, fastest_lap=True) == 26.0
    assert points_for_position(2020, 10, fastest_lap=True) == 2.0
    assert points_for_position(2020, 11, fastest_lap=True) == 0.0
    # Sprint races never award the fastest-lap bonus (2021: P1 sprint = 3 pts).
    assert points_for_position(2021, 1, sprint=True, fastest_lap=True) == 3.0


def test_sprint_and_double_points():
    assert points_for_position(2023, 1, sprint=True) == 8.0
    assert points_for_position(2021, 3, sprint=True) == 1.0
    assert is_double_points_race(2014, 19) is True
    assert is_double_points_race(2014, 18) is False
    assert points_for_position(2014, 1, double_points=True) == 50.0


def test_race_points_for_order_matches_position_rules():
    order = ["a", "b", "c"]
    points = race_points_for_order(2024, order, fastest_lap_driver="c")
    assert points["a"] == 25.0
    assert points["b"] == 18.0
    # Third place + fastest lap (2024 rule).
    assert points["c"] == 16.0


def test_expected_points_from_distribution():
    # Half probability of winning, half of finishing second.
    assert expected_points_from_distribution(2024, [0.5, 0.5]) == 21.5
    assert expected_points_from_distribution(2024, [1.0], sprint=True) == 8.0
    assert expected_points_from_distribution(2024, [1.0], double_points=True) == 50.0


def test_scoring_rules_description_is_serialisable():
    description = scoring_rules_description(2021)
    assert description["season"] == 2021
    assert description["sprint_points"] == [3, 2, 1]
    assert description["fastest_lap_point"] is True
