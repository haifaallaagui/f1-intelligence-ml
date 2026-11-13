"""Season-aware Formula 1 scoring rules.

Only rules that are documented public regulations are encoded here.  The
functions are used to *predict* points; actual historical points always come
from the data source and are never overwritten.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# Race points by era.  The 2010-2025 system is also used for 2026.
RACE_POINTS_TOP6 = (10, 6, 4, 3, 2, 1)  # 1995-2002
RACE_POINTS_TOP8 = (10, 8, 6, 5, 4, 3, 2, 1)  # 2003-2009
RACE_POINTS_TOP10 = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)  # 2010+
SPRINT_POINTS_2021 = (3, 2, 1)
SPRINT_POINTS_2022 = (8, 7, 6, 5, 4, 3, 2, 1)


def race_points_system(season: int) -> tuple[int, ...]:
    if season <= 2002:
        return RACE_POINTS_TOP6
    if season <= 2009:
        return RACE_POINTS_TOP8
    return RACE_POINTS_TOP10


def sprint_points_system(season: int) -> tuple[int, ...]:
    if season < 2021:
        return ()
    if season == 2021:
        return SPRINT_POINTS_2021
    return SPRINT_POINTS_2022


def fastest_lap_point_available(season: int) -> bool:
    # The fastest-lap bonus point existed from 2019 to 2024 inclusive.
    return 2019 <= season <= 2024


def points_for_position(
    season: int,
    position: int | None,
    *,
    sprint: bool = False,
    fastest_lap: bool = False,
    double_points: bool = False,
) -> float:
    if position is None or position <= 0:
        return 0.0
    system = sprint_points_system(season) if sprint else race_points_system(season)
    points = float(system[position - 1]) if position <= len(system) else 0.0
    if double_points and not sprint:
        points *= 2.0
    if fastest_lap and not sprint and fastest_lap_point_available(season) and position <= 10:
        points += 1.0
    return points


def is_double_points_race(season: int, round_number: int) -> bool:
    # The 2014 Abu Dhabi Grand Prix was the only double-points race.
    return int(season) == 2014 and int(round_number) == 19


def race_points_for_order(
    season: int,
    order: Sequence[str],
    *,
    sprint: bool = False,
    fastest_lap_driver: str | None = None,
    double_points: bool = False,
) -> dict[str, float]:
    return {
        driver: points_for_position(
            season,
            index + 1,
            sprint=sprint,
            fastest_lap=(fastest_lap_driver == driver),
            double_points=double_points,
        )
        for index, driver in enumerate(order)
    }


def expected_points_from_distribution(
    season: int,
    position_probabilities: Iterable[float],
    *,
    sprint: bool = False,
    double_points: bool = False,
) -> float:
    system = sprint_points_system(season) if sprint else race_points_system(season)
    total = 0.0
    for index, probability in enumerate(position_probabilities):
        if index >= len(system):
            break
        total += float(probability) * system[index] * (2 if double_points and not sprint else 1)
    return total


def scoring_rules_description(season: int) -> dict[str, object]:
    return {
        "season": int(season),
        "race_points": list(race_points_system(season)),
        "sprint_points": list(sprint_points_system(season)),
        "fastest_lap_point": fastest_lap_point_available(season),
        "double_points_race": "2014 Abu Dhabi" if season == 2014 else None,
    }
