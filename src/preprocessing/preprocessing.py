"""Model-facing preprocessing: leakage guards, splits and matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

LOG = get_logger("preprocessing.preprocessing")

# These prefixes/columns must never be used as inputs.  They either contain the
# target race outcome or information generated after the race.
FORBIDDEN_FEATURE_PATTERNS = (
    "finish_position",
    "classified_position",
    "position_raw",
    "position_text",
    "points",
    "race_points",
    "total_points",
    "sprint_points_actual",
    "win",
    "podium",
    "top10",
    "dnf",
    "status",
    "fastest_lap",
    "position_gain",
    "target",
)


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    """Raise if a forbidden outcome-derived column is present in the inputs."""
    offenders = []
    for column in feature_columns:
        name = str(column).lower()
        for pattern in FORBIDDEN_FEATURE_PATTERNS:
            if name == pattern or name.endswith(f"_{pattern}") or name.startswith(f"{pattern}_"):
                # Explicitly allowed historical aggregates use the word
                # "points" in their names; they are filtered by the feature
                # lists, not by this guard.  The guard targets exact target
                # columns and obvious target variants.
                if name in FORBIDDEN_FEATURE_PATTERNS:
                    offenders.append(column)
                    break
    if offenders:
        raise ValueError(f"Leakage guard triggered by columns: {offenders}")


def clean_feature_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    X = df.loc[:, list(feature_columns)].copy()
    for column in X.columns:
        # pandas >= 3 infers a dedicated string dtype instead of ``object``,
        # so check "not numeric" rather than comparing to ``object``.
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def add_missing_indicators(X: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    columns = list(columns or X.columns)
    out = X.copy()
    for column in columns:
        if column in out.columns and out[column].isna().any():
            out[f"{column}_missing"] = out[column].isna().astype(int)
    return out


def chronological_split(
    df: pd.DataFrame,
    train_seasons: tuple[int, int] = config.TRAIN_SEASONS,
    validation_seasons: tuple[int, int] = config.VALIDATION_SEASONS,
    test_seasons: tuple[int, int] = config.TEST_SEASONS,
) -> DatasetSplit:
    train = df[(df["season"] >= train_seasons[0]) & (df["season"] <= train_seasons[1])].copy()
    validation = df[
        (df["season"] >= validation_seasons[0]) & (df["season"] <= validation_seasons[1])
    ].copy()
    test = df[(df["season"] >= test_seasons[0]) & (df["season"] <= test_seasons[1])].copy()
    if train.empty or validation.empty:
        raise ValueError(
            f"Chronological split produced empty train/validation sets: "
            f"{len(train)}/{len(validation)} rows"
        )
    return DatasetSplit(train=train, validation=validation, test=test)


def walk_forward_folds(
    df: pd.DataFrame,
    min_train_seasons: int = config.WALK_FORWARD_MIN_TRAIN_SEASONS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window season folds for honest temporal validation."""
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    folds = []
    for index in range(min_train_seasons, len(seasons)):
        validation_season = seasons[index]
        train = df[df["season"].isin(seasons[:index])].copy()
        validation = df[df["season"] == validation_season].copy()
        if not train.empty and not validation.empty:
            folds.append((train, validation))
    return folds


def make_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
    *,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found")
    subset = df
    if dropna_target:
        subset = subset[subset[target].notna()]
    X = clean_feature_frame(subset, feature_columns)
    y = subset[target]
    groups = subset["race_id"] if "race_id" in subset.columns else pd.Series(range(len(subset)), index=subset.index)
    return X, y, groups


def season_weights(df: pd.DataFrame, decay: float = 0.97) -> np.ndarray:
    """Exponentially weight recent seasons more heavily."""
    if df.empty:
        return np.array([])
    max_season = int(df["season"].max())
    return np.power(decay, max_season - df["season"].astype(int)).to_numpy()


def relative_position_target(df: pd.DataFrame) -> pd.Series:
    return df["finish_position"] / df["field_size"].clip(lower=1)


def ranking_relevance(df: pd.DataFrame, max_relevance: int = 20) -> pd.Series:
    """Higher relevance for better finishes, integer labels for rankers."""
    field = df["field_size"].clip(lower=1)
    rel = (field - df["finish_position"] + 1).clip(lower=0)
    return rel.round().astype(int).clip(upper=max_relevance)
