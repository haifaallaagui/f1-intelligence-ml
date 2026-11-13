# F1 Intelligence ML

A complete, standalone machine-learning component for Formula 1 race prediction.
It collects real historical data (1995-2026), builds leakage-safe features,
trains and compares several model families, evaluates them on a chronological
test window, explains predictions, projects championships with Monte-Carlo
simulation, and supports automatic data updates and retraining.

## What it predicts

| Task                     | Output                                                                 |
| ------------------------ | ---------------------------------------------------------------------- |
| Winner                   | Probability for every driver, predicted winner                         |
| Podium                   | Probability of a top-3 finish, predicted podium                        |
| Full classification      | Coherent P1-P20 order, position probabilities, expected position       |
| Driver performance       | Expected finishing position and expected points                        |
| Constructor performance  | Expected team points and best finish                                   |
| Driver championship      | Championship probability, expected final points, position distribution |
| Constructor championship | Championship probability, expected final points                        |

Two prediction modes are supported:

- **Mode 1 - before qualifying**: uses only information available before the
  qualifying session. Qualifying, grid and sprint columns are never used.
- **Mode 2 - after qualifying**: additionally uses qualifying pace, grid
  position, sprint results and optional FastF1 qualifying summaries.

## Data sources and coverage

| Source                                        | Role                                                                                            | Coverage                                                                       |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) | Primary historical source: schedules, results, qualifying, sprints, standings, reference tables | 1995-2026 (whatever the API exposes; missing data is recorded, never invented) |
| [OpenF1](https://openf1.org/)                 | Recent session enrichment: weather, laps, stints, pit stops, positions, race control            | 2023-2026                                                                      |
| [FastF1](https://docs.fastf1.dev/)            | Optional detailed qualifying pace, sectors, compounds, circuit geometry                         | 2018-2026 (session loading is opt-in because it is large)                      |

The generated `reports/data_coverage.md` and `reports/data_coverage.json`
document, season by season, how many races/results/qualifying/sprint rows were
actually collected. A missing endpoint is stored as unavailable; no synthetic
result, grid or qualifying time is inserted.

## Project structure

```
f1-intelligence-ml/
  config.py                     # paths, seasons, splits, model settings
  requirements.txt
  scripts/
    collect_data.py             # collect Jolpica/OpenF1/FastF1 data
    build_dataset.py            # build processed tables + feature datasets
    train_all_models.py         # tune/compare/train every task and mode
    evaluate_all_models.py      # chronological test evaluation + baselines
    predict_race.py             # race prediction + championship projection
    update_pipeline.py          # automatic update/retrain entry point
  src/
    data_collection/            # Jolpica, OpenF1, FastF1 collectors + coverage
    preprocessing/              # cleaning, leakage guards, splits
    features/                   # driver/constructor/circuit histories, features
    models/                     # winner, podium, classification, performance, championship
    training/                   # model selection, training, evaluation, registry
    prediction/                 # prediction helpers and end-to-end engine
    explainability/             # SHAP explanations with an honest fallback
    automation/                 # update detection and retraining pipeline
  tests/                        # unit tests with mocked external APIs
  notebooks/                    # exploration and model-analysis notebooks
  reports/                      # coverage, training and evaluation reports
```

## Installation

Python 3.10+ is required (the pipeline was validated with Python 3.12).

```powershell
cd "C:\Users\haifa\Documents\Projet par défaut\f1-intelligence-ml"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Commands

The exact workflow requested:

```powershell
python scripts/collect_data.py --start 1995 --end 2026
python scripts/build_dataset.py
python scripts/train_all_models.py
python scripts/evaluate_all_models.py
python scripts/predict_race.py --race-id RACE_ID --mode after_qualifying
python scripts/collect_data.py --update
```

Useful variants:

```powershell
# Faster smoke test (Jolpica only, recent seasons)
python scripts/collect_data.py --start 2023 --end 2026 --skip-openf1 --skip-fastf1

# Automatic update: checks the configured interval, refreshes recent seasons,
# rebuilds the dataset and retrains only when the dataset hash changed
python scripts/update_pipeline.py

# Predict before qualifying
python scripts/predict_race.py --race-id 2026_06 --mode before_qualifying

# Provide a custom entry list (driver_id, constructor_id)
python scripts/predict_race.py --race-id 2026_06 --mode after_qualifying --entries entries.csv

# Train only Mode 2 with a smaller hyper-parameter search
python scripts/train_all_models.py --modes after_qualifying --n-iter 3
```

Race IDs use the form `2026_06` (season_round). The parser also accepts
`2026-06`.

## Leakage prevention

Leakage is prevented structurally, not by hoping the model behaves:

1. **Temporal state updates.** Driver, constructor and circuit histories are
   updated only _after_ the feature row for a race has been created. A race
   can never see its own result.
2. **Explicit target columns.** Finish position, points, wins, podiums, DNFs
   and related outcomes are never included in the feature lists.
3. **Post-qualifying isolation.** Qualifying, grid, sprint and FastF1 columns
   are stored separately and only added for Mode 2.
4. **Leakage guard.** `src/preprocessing/preprocessing.py` rejects forbidden
   outcome-derived feature names, and `FeatureEngineer._write_metadata()`
   asserts that targets do not appear in base or post-qualifying features.
5. **Chronological evaluation.** Train 1995-2018, validation 2019-2021, test
   2022-2026. Hyper-parameters are selected on validation only. The test
   window is never used for model selection.
6. **Actual points.** Historical points come from the data source. The
   season-aware scoring functions are used only for predictions and
   championship simulations.

## Models and evaluation

Model families compared per task:

- Winner / podium: Logistic Regression, Random Forest, HistGradientBoosting,
  XGBoost, LightGBM, CatBoost.
- Full classification: XGBoost ranker, LightGBM ranker, HistGradientBoosting
  regression on relative finishing position.
- Driver / constructor performance: HistGradientBoosting, Random Forest,
  XGBoost, LightGBM.

The full classification model produces a coherent race order. Position
probabilities are derived with a Plackett-Luce model over the model scores.
The Plackett-Luce order is sampled exactly with the Gumbel top-k trick, so the
marginal position probabilities are consistent with a valid finishing order.

Metrics reported by `scripts/evaluate_all_models.py`:

- Winner: log loss, Brier score, top-1 accuracy, top-3 hit rate.
- Podium: log loss, Brier, precision/recall/F1 for the top-3 set, exact-set
  accuracy.
- Full classification: MAE, RMSE, R2, Spearman correlation, NDCG@10, pairwise
  ranking accuracy, top-1 accuracy, podium precision@3.
- Driver performance: MAE/RMSE/R2 for finishing position and points.
- Constructor performance: MAE/RMSE/R2 for team points and best finish.
- Baselines: championship order, recent form, career form, and (Mode 2 only)
  grid order and qualifying order.

Results are written to `reports/evaluation_report.md` and
`reports/evaluation_report.json`. Model selection details are written next to
each bundle (`models/<task>/<mode>/comparison.json`).

## Explainability and uncertainty

`src/explainability/shap_explainer.py` uses SHAP TreeExplainer or
LinearExplainer when the model supports it. If SHAP cannot explain a model, the
module returns global feature importance and explicitly says that local
contributions are unavailable. It never fabricates an explanation.

Race predictions include:

- per-driver win and podium probabilities;
- a full position distribution and expected finishing position;
- expected points;
- a championship projection with champion probabilities and uncertainty
  (standard deviation and 10th/90th percentiles of final points).

## Model persistence and registry

Every selected model is saved as a joblib bundle with metadata
(`models/<task>/<mode>/best.joblib` and `best.metadata.json`). The registry
`models/registry.json` records the task, mode, algorithm, version, dataset hash
and validation metrics. `scripts/train_all_models.py` skips a model when the
dataset hash is unchanged, which avoids unnecessary retraining.

## Automatic updates

`src/automation/update_pipeline.py`:

1. checks `data/metadata/update_state.json` and the configured interval
   (`F1_UPDATE_INTERVAL_HOURS`, default 6 hours);
2. detects new rounds and refreshes the recent seasons;
3. rebuilds processed tables and feature datasets;
4. compares the new dataset hash with the previous one;
5. retrains only when the dataset actually changed (unless `--force`).

## Tests

```powershell
pytest -q
```

The test suite mocks the external HTTP APIs. It covers collection pagination
and merging, preprocessing, leakage guards, feature construction, training,
evaluation metrics and winner/podium/classification/championship prediction.

## Limitations

- OpenF1 starts in 2023 and FastF1 detailed sessions are optional, so recent
  races have more features than older ones. Missing values are imputed by the
  model pipeline; they are not filled with invented history.
- The championship simulator does not model the fastest-lap bonus point and
  uses the same strength scores for sprint and race orders when no separate
  sprint model is supplied.
- Formula 1 regulations, calendars, teams and driver line-ups change over
  time. The models learn from historical patterns; they cannot know about a
  future regulation change, a mechanical failure or a driver replacement that
  is not in the collected data.
- Predictions are probabilistic model outputs, not guarantees. They must not be
  presented as historical facts.
