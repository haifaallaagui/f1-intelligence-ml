"""Central configuration for the F1 Intelligence machine-learning component.

The module deliberately contains no driver/team names.  Everything that is
specific to a season is either fetched from a data source or derived from
historical records at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESOURCES_DIR = PROJECT_ROOT / "resources"
TESTS_DIR = PROJECT_ROOT / "tests"

JOLPICA_RAW_DIR = RAW_DIR / "jolpica"
OPENF1_RAW_DIR = RAW_DIR / "openf1"
FASTF1_RAW_DIR = RAW_DIR / "fastf1"

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
JOLPICA_BASE_URL = os.getenv("JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1")
OPENF1_BASE_URL = os.getenv("OPENF1_BASE_URL", "https://api.openf1.org/v1")
MULTIVIEWER_BASE_URL = os.getenv("MULTIVIEWER_BASE_URL", "https://api.multiviewer.app/api/v1")

# Public APIs are rate limited.  Jolpica documents a burst limit around
# 4 requests/second but in practice enforces a stricter window; the
# conservative value below keeps collection reliable without excessive 429s.
REQUEST_RATE_LIMIT_SECONDS = float(os.getenv("F1_REQUEST_RATE_LIMIT", "0.8"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("F1_REQUEST_TIMEOUT", "90"))
MAX_RETRIES = int(os.getenv("F1_MAX_RETRIES", "5"))
USER_AGENT = "f1-intelligence-ml/1.0 (+https://github.com/f1-intelligence)"

# ---------------------------------------------------------------------------
# Historical coverage
# ---------------------------------------------------------------------------
SEASON_START = int(os.getenv("F1_SEASON_START", "1995"))
SEASON_END = int(os.getenv("F1_SEASON_END", "2026"))
# OpenF1 coverage starts in 2023.  FastF1's timing data is reliable from 2018
# onwards.  These are *source coverage* limits, not claims about F1 history.
OPENF1_START_SEASON = 2023
FASTF1_START_SEASON = 2018

# ---------------------------------------------------------------------------
# Chronological evaluation windows
# ---------------------------------------------------------------------------
# Tuning uses the training window only.  Validation selects the algorithm and
# hyper-parameters.  The test window is never used for model selection.
TRAIN_SEASONS = (1995, 2018)
VALIDATION_SEASONS = (2019, 2021)
TEST_SEASONS = (2022, 2026)
WALK_FORWARD_MIN_TRAIN_SEASONS = 8

RANDOM_SEED = 42
N_TUNING_ITER = int(os.getenv("F1_TUNING_ITER", "8"))
CV_FOLDS = int(os.getenv("F1_CV_FOLDS", "3"))

# Model families compared for each task.  The list can be trimmed from the
# command line with --algorithms for quicker experiments.
CLASSIFICATION_ALGORITHMS = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]
RANKING_ALGORITHMS = ["xgboost_ranker", "lightgbm_ranker", "hist_gradient_boosting"]
REGRESSION_ALGORITHMS = ["hist_gradient_boosting", "random_forest", "xgboost", "lightgbm"]

# Number of Monte-Carlo seasons used by championship projections.
CHAMPIONSHIP_SIMULATIONS = int(os.getenv("F1_CHAMPIONSHIP_SIMULATIONS", "5000"))

# ---------------------------------------------------------------------------
# Automation
# ---------------------------------------------------------------------------
UPDATE_INTERVAL_HOURS = float(os.getenv("F1_UPDATE_INTERVAL_HOURS", "6"))
UPDATE_STATE_FILE = METADATA_DIR / "update_state.json"
REGISTRY_FILE = MODELS_DIR / "registry.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
MODEL_VERSION_FORMAT = "%Y%m%dT%H%M%SZ"

# ---------------------------------------------------------------------------
# Dataset artefacts
# ---------------------------------------------------------------------------
RACE_FEATURES_FILE = DATASETS_DIR / "race_features.parquet"
CONSTRUCTOR_FEATURES_FILE = DATASETS_DIR / "constructor_race_features.parquet"
DATASET_METADATA_FILE = DATASETS_DIR / "dataset_metadata.json"
PROCESSED_TABLES = {
    "races": PROCESSED_DIR / "races.parquet",
    "results": PROCESSED_DIR / "results.parquet",
    "qualifying": PROCESSED_DIR / "qualifying.parquet",
    "sprints": PROCESSED_DIR / "sprints.parquet",
    "drivers": PROCESSED_DIR / "drivers.parquet",
    "constructors": PROCESSED_DIR / "constructors.parquet",
    "circuits": PROCESSED_DIR / "circuits.parquet",
    "status": PROCESSED_DIR / "status.parquet",
    "driver_standings": PROCESSED_DIR / "driver_standings.parquet",
    "constructor_standings": PROCESSED_DIR / "constructor_standings.parquet",
    "openf1_race_summary": PROCESSED_DIR / "openf1_race_summary.parquet",
    "openf1_driver_race_summary": PROCESSED_DIR / "openf1_driver_race_summary.parquet",
    "fastf1_qualifying_summary": PROCESSED_DIR / "fastf1_qualifying_summary.parquet",
}

TASK_DIRECTORIES = {
    "winner": MODELS_DIR / "winner",
    "podium": MODELS_DIR / "podium",
    "classification": MODELS_DIR / "classification",
    "driver_performance": MODELS_DIR / "driver_performance",
    "constructor_performance": MODELS_DIR / "constructor_performance",
    "championship": MODELS_DIR / "championship",
}


def ensure_directories() -> None:
    """Create every directory used by the project (idempotent)."""
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        DATASETS_DIR,
        METADATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        PREDICTIONS_DIR,
        NOTEBOOKS_DIR,
        RESOURCES_DIR,
        JOLPICA_RAW_DIR,
        OPENF1_RAW_DIR,
        FASTF1_RAW_DIR,
        *TASK_DIRECTORIES.values(),
    ):
        path.mkdir(parents=True, exist_ok=True)
