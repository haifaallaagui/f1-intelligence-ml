"""End-to-end prediction engine for a single race and a season projection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import config
from src.explainability.shap_explainer import explain_predictions
from src.features.feature_engineering import (
    BASE_CONTEXT_FEATURES,
    DRIVER_AGG_FEATURES,
    FeatureEngineer,
    get_constructor_feature_columns,
    get_feature_columns,
)
from src.models.classification_model import ClassificationModel
from src.models.constructor_model import ConstructorPerformanceModel
from src.models.driver_model import DriverPerformanceModel
from src.models.podium_model import PodiumModel
from src.models.winner_model import WinnerModel
from src.prediction.championship_prediction import project_championship
from src.prediction.classification_prediction import (
    classification_summary,
    predict_classification,
)
from src.prediction.podium_prediction import podium_summary, predict_podium
from src.prediction.winner_prediction import predict_winner, winner_summary
from src.training.evaluate import load_task_model
from src.utils import ensure_dirs, get_logger, make_race_id, parse_race_id, save_json, utc_now_iso

LOG = get_logger("prediction.engine")


def _records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


class PredictionEngine:
    def __init__(self, mode: str = "after_qualifying", tables: dict[str, pd.DataFrame] | None = None):
        if mode not in {"before_qualifying", "after_qualifying"}:
            raise ValueError("mode must be 'before_qualifying' or 'after_qualifying'")
        self.mode = mode
        self.engineer = FeatureEngineer(tables=tables)
        self.feature_columns = get_feature_columns(mode)
        self.models = self._load_models(mode)

    def _load_models(self, mode: str) -> dict[str, Any]:
        return {
            "winner": load_task_model("winner", mode),
            "podium": load_task_model("podium", mode),
            "classification": load_task_model("classification", mode),
            "driver_performance": load_task_model("driver_performance", mode),
            "constructor_performance": load_task_model("constructor_performance", "any"),
        }

    # ------------------------------------------------------------------
    # Race prediction
    # ------------------------------------------------------------------
    def predict_race(
        self,
        race_id: str,
        *,
        entries: pd.DataFrame | list[dict[str, Any]] | None = None,
        include_championship: bool = True,
        n_simulations: int = config.CHAMPIONSHIP_SIMULATIONS,
        explain: bool = True,
    ) -> dict[str, Any]:
        season, round_number = parse_race_id(race_id)
        race_id = make_race_id(season, round_number)
        features, info = self.engineer.build_prediction_features(
            race_id, entries=entries, mode=self.mode
        )
        if features.empty:
            raise RuntimeError(f"No prediction features for {race_id}")

        classification_model: ClassificationModel | None = self.models.get("classification")
        if classification_model is None:
            raise RuntimeError(
                f"No classification model for mode {self.mode}; run scripts/train_all_models.py"
            )
        base, matrices, expected = predict_classification(
            classification_model, features, self.feature_columns
        )

        winner_model: WinnerModel | None = self.models.get("winner")
        if winner_model is not None:
            winner_predictions = predict_winner(winner_model, features, self.feature_columns)
            base = base.merge(
                winner_predictions[
                    ["race_id", "driver_id", "winner_probability", "winner_probability_raw", "winner_rank"]
                ],
                on=["race_id", "driver_id"],
                how="left",
            )
            base["predicted_win"] = base["winner_rank"].fillna(99).astype(int) == 1
        else:
            base["winner_probability"] = np.nan
            base["predicted_win"] = False

        podium_model: PodiumModel | None = self.models.get("podium")
        if podium_model is not None:
            podium_predictions, _ = predict_podium(podium_model, features, self.feature_columns)
            base = base.merge(
                podium_predictions[
                    ["race_id", "driver_id", "podium_probability", "podium_probability_raw"]
                ],
                on=["race_id", "driver_id"],
                how="left",
            )
            base["predicted_podium"] = (
                base.groupby("race_id")["podium_probability"].rank(method="first", ascending=False) <= 3
            )
        else:
            base["podium_probability"] = np.nan
            base["predicted_podium"] = False

        driver_model: DriverPerformanceModel | None = self.models.get("driver_performance")
        if driver_model is not None:
            driver_predictions = driver_model.predict(features, self.feature_columns)
            base = base.merge(
                driver_predictions[
                    ["race_id", "driver_id", "expected_finish", "expected_points", "predicted_finish_position"]
                ],
                on=["race_id", "driver_id"],
                how="left",
            )
        else:
            base["expected_finish"] = np.nan
            base["expected_points"] = np.nan

        base = base.sort_values(["race_id", "predicted_position"]).reset_index(drop=True)

        constructors = self._constructor_predictions(base)
        championship = None
        if include_championship:
            try:
                target_order = int(self.engineer.race_order[race_id])
                championship = project_championship(
                    self.engineer,
                    classification_model,
                    self.feature_columns,
                    season,
                    self.mode,
                    points_through_order=target_order - 1,
                    simulate_from_order=target_order,
                    n_simulations=n_simulations,
                )
            except Exception as exc:  # noqa: BLE001 - a projection must not block a race prediction
                LOG.warning("Championship projection failed for %s: %s", race_id, exc)
                championship = {"error": str(exc)}

        explanations: dict[str, Any] = {}
        if explain:
            try:
                if winner_model is not None:
                    explanations["winner"] = explain_predictions(
                        winner_model.model, features, self.feature_columns
                    )
                explanations["classification"] = explain_predictions(
                    classification_model.model, features, self.feature_columns
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Explanation failed: %s", exc)
                explanations = {"error": str(exc)}

        result = {
            "race_info": info,
            "drivers": base,
            "constructors": constructors,
            "championship": championship,
            "explanations": explanations,
            "summary": {
                "winner": winner_summary(base),
                "podium": podium_summary(base),
                "classification": classification_summary(base),
            },
            "metadata": {
                "mode": self.mode,
                "generated_at": utc_now_iso(),
                "models": {
                    task: getattr(model, "algorithm", None)
                    for task, model in self.models.items()
                    if model is not None
                },
                "n_samples_for_position_probabilities": 2000,
                "note": "Predictions are model outputs, not historical facts. Check race_info.has_actual_results.",
            },
        }
        return result

    # ------------------------------------------------------------------
    # Constructor aggregation
    # ------------------------------------------------------------------
    def _build_constructor_features(self, driver_table: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        base_columns = [
            column
            for column in driver_table.columns
            if column.startswith(("constructor_", "circuit_"))
            or column in BASE_CONTEXT_FEATURES
            or column
            in {
                "race_id",
                "season",
                "round",
                "circuit_id",
                "race_name",
                "circuit_name",
                "race_date",
                "field_size",
            }
        ]
        for (race_id, constructor_id), group in driver_table.groupby(
            ["race_id", "constructor_id"], sort=False
        ):
            row: dict[str, Any] = {
                "race_id": race_id,
                "constructor_id": constructor_id,
                "constructor_name": group["constructor_name"].iloc[0]
                if "constructor_name" in group.columns
                else constructor_id,
                "team_driver_count": float(len(group)),
            }
            for column in base_columns:
                if column in group.columns and column not in row:
                    row[column] = group[column].iloc[0]
            for column in DRIVER_AGG_FEATURES:
                if column not in group.columns:
                    continue
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"team_avg_driver_{column}"] = float(values.mean()) if values.notna().any() else np.nan
                row[f"team_max_driver_{column}"] = float(values.max()) if values.notna().any() else np.nan
                row[f"team_min_driver_{column}"] = float(values.min()) if values.notna().any() else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _constructor_predictions(self, driver_table: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (race_id, constructor_id), group in driver_table.groupby(
            ["race_id", "constructor_id"], sort=False
        ):
            win_probability = group.get("winner_probability", pd.Series(dtype=float)).fillna(0.0)
            podium_probability = group.get("podium_probability", pd.Series(dtype=float)).fillna(0.0)
            expected_points = group.get("expected_points", pd.Series(dtype=float)).fillna(0.0)
            expected_finish = group.get("expected_finish", pd.Series(dtype=float))
            rows.append(
                {
                    "race_id": race_id,
                    "constructor_id": constructor_id,
                    "constructor_name": group["constructor_name"].iloc[0]
                    if "constructor_name" in group.columns
                    else constructor_id,
                    "expected_team_points": float(expected_points.sum()),
                    "expected_team_best_finish": float(expected_finish.min())
                    if expected_finish.notna().any()
                    else np.nan,
                    "team_win_probability": float(1.0 - np.prod(1.0 - win_probability.to_numpy(dtype=float))),
                    "team_podium_probability": float(podium_probability.sum()),
                    "team_expected_finish": float(expected_finish.mean())
                    if expected_finish.notna().any()
                    else np.nan,
                    "team_driver_count": float(len(group)),
                }
            )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table["predicted_team_position"] = (
            table.groupby("race_id")["expected_team_points"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        constructor_model: ConstructorPerformanceModel | None = self.models.get("constructor_performance")
        if constructor_model is not None:
            try:
                constructor_features = self._build_constructor_features(driver_table)
                model_columns = get_constructor_feature_columns()
                for column in model_columns:
                    if column not in constructor_features.columns:
                        constructor_features[column] = np.nan
                predictions = constructor_model.predict(constructor_features, model_columns)
                table = table.merge(
                    predictions[
                        [
                            "race_id",
                            "constructor_id",
                            "expected_team_points",
                            "expected_team_best_finish",
                            "predicted_team_position",
                        ]
                    ].rename(
                        columns={
                            "expected_team_points": "model_expected_team_points",
                            "expected_team_best_finish": "model_expected_team_best_finish",
                            "predicted_team_position": "model_predicted_team_position",
                        }
                    ),
                    on=["race_id", "constructor_id"],
                    how="left",
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Constructor model prediction failed: %s", exc)
        return table.sort_values(["race_id", "predicted_team_position"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_prediction(self, result: dict[str, Any], output_dir: Path | None = None) -> Path:
        race_info = result.get("race_info", {})
        race_id = race_info.get("race_id", "race")
        mode = result.get("metadata", {}).get("mode", self.mode)
        output_dir = Path(output_dir or config.PREDICTIONS_DIR / f"{race_id}_{mode}")
        ensure_dirs(output_dir)

        drivers = result.get("drivers", pd.DataFrame())
        constructors = result.get("constructors", pd.DataFrame())
        championship = result.get("championship")
        if isinstance(drivers, pd.DataFrame):
            drivers.to_csv(output_dir / "drivers.csv", index=False)
        if isinstance(constructors, pd.DataFrame):
            constructors.to_csv(output_dir / "constructors.csv", index=False)
        if isinstance(championship, dict) and "drivers" in championship:
            championship["drivers"].to_csv(output_dir / "championship_drivers.csv", index=False)
            championship["constructors"].to_csv(
                output_dir / "championship_constructors.csv", index=False
            )

        payload = {
            "race_info": race_info,
            "drivers": _records(drivers if isinstance(drivers, pd.DataFrame) else pd.DataFrame()),
            "constructors": _records(
                constructors if isinstance(constructors, pd.DataFrame) else pd.DataFrame()
            ),
            "championship": (
                {
                    "drivers": _records(championship.get("drivers")),
                    "constructors": _records(championship.get("constructors")),
                    "metadata": championship.get("metadata"),
                }
                if isinstance(championship, dict) and "drivers" in championship
                else championship
            ),
            "explanations": result.get("explanations"),
            "summary": result.get("summary"),
            "metadata": result.get("metadata"),
        }
        save_json(output_dir / "prediction.json", payload)
        (output_dir / "prediction.md").write_text(
            self._markdown(result), encoding="utf-8"
        )
        return output_dir

    def _markdown(self, result: dict[str, Any]) -> str:
        info = result.get("race_info", {})
        lines = [f"# Race prediction: {info.get('race_name', info.get('race_id'))}\n"]
        lines.append(
            f"- Race: `{info.get('race_id')}`\n"
            f"- Mode: `{info.get('mode')}`\n"
            f"- Entries source: `{info.get('entries_source')}`\n"
            f"- Actual results available: `{info.get('has_actual_results')}`\n"
            f"- Qualifying available: `{info.get('has_qualifying')}`\n"
        )
        drivers = result.get("drivers", pd.DataFrame())
        if isinstance(drivers, pd.DataFrame) and not drivers.empty:
            lines.append("\n## Predicted classification\n")
            lines.append(
                "| Pos | Driver | Team | Win % | Podium % | Expected finish | Expected points |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
            )
            for _, row in drivers.sort_values("predicted_position").head(20).iterrows():
                lines.append(
                    f"| {int(row['predicted_position'])} | {row.get('driver_name', row['driver_id'])} | "
                    f"{row.get('constructor_name', row.get('constructor_id'))} | "
                    f"{100 * float(row.get('winner_probability', 0.0) or 0.0):.1f} | "
                    f"{100 * float(row.get('podium_probability', 0.0) or 0.0):.1f} | "
                    f"{float(row.get('expected_finish', float('nan'))):.1f} | "
                    f"{float(row.get('expected_points', float('nan'))):.1f} |\n"
                )
        constructors = result.get("constructors", pd.DataFrame())
        if isinstance(constructors, pd.DataFrame) and not constructors.empty:
            lines.append("\n## Constructor projection\n")
            lines.append(
                "| Pos | Constructor | Expected points | Win % | Podiums (expected) |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
            for _, row in constructors.sort_values("predicted_team_position").iterrows():
                lines.append(
                    f"| {int(row['predicted_team_position'])} | {row.get('constructor_name', row['constructor_id'])} | "
                    f"{float(row.get('expected_team_points', 0.0)):.1f} | "
                    f"{100 * float(row.get('team_win_probability', 0.0)):.1f} | "
                    f"{float(row.get('team_podium_probability', 0.0)):.2f} |\n"
                )
        championship = result.get("championship")
        if isinstance(championship, dict) and "drivers" in championship and not championship["drivers"].empty:
            lines.append("\n## Championship projection\n")
            lines.append(
                "| Driver | Current points | Expected final | Champion % | Top-3 % |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
            for _, row in championship["drivers"].head(10).iterrows():
                lines.append(
                    f"| {row.get('driver_name', row['driver_id'])} | "
                    f"{float(row.get('current_points', 0.0)):.1f} | "
                    f"{float(row.get('expected_final_points', 0.0)):.1f} | "
                    f"{100 * float(row.get('championship_probability', 0.0)):.1f} | "
                    f"{100 * float(row.get('top3_probability', 0.0)):.1f} |\n"
                )
        explanations = result.get("explanations", {})
        if isinstance(explanations, dict) and explanations:
            lines.append("\n## Explanations\n")
            for name, explanation in explanations.items():
                if not isinstance(explanation, dict):
                    continue
                lines.append(
                    f"- **{name}**: method `{explanation.get('method')}`. "
                    f"{explanation.get('note', '')}\n"
                )
                importance = explanation.get("global_importance", [])[:5]
                if importance:
                    lines.append(
                        "  Top global features: "
                        + ", ".join(
                            f"`{item['feature']}` ({item['mean_abs_shap']:.4f})"
                            for item in importance
                        )
                        + "\n"
                    )
        lines.append(
            "\n> Predictions are model outputs. They are not guarantees and must not be "
            "presented as historical facts.\n"
        )
        return "".join(lines)
