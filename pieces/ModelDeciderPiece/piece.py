import json
from pathlib import Path

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


TREE_MODELS = {"xgb_regressor_model", "interval_xgb_regressor_model", "eda_rule_baseline"}


class ModelDeciderPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running ModelDeciderPiece.")

        payload = input_data.payload_as_dict()

        model_type = (
            payload.get("model_type")
            or self._pick_model(payload.get("available_models"))
            or "xgb_regressor_model"
        )
        model_type = str(model_type).lower()

        # Tree-based models don't need scaling; everything else defaults to z_score.
        normalization_type = payload.get("normalization_type")
        if normalization_type is None:
            normalization_type = "none" if model_type in TREE_MODELS else "z_score"
        normalization_type = str(normalization_type).lower()

        decision = {
            "model_type": model_type,
            "normalization_type": normalization_type,
            "feature_columns": payload.get("feature_columns"),
            "target_column": payload.get("target_column", "PVOUT"),
            "problem_type": payload.get("problem_type"),
            "horizon": payload.get("horizon"),
        }

        decision_path = str(Path(self.results_path) / "decision.json")
        Path(decision_path).parent.mkdir(parents=True, exist_ok=True)
        with open(decision_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)
        self.display_result = {"file_type": "txt", "file_path": decision_path}

        artifacts = dict(decision)
        artifacts["decision_path"] = decision_path

        return OutputModel(
            message=f"ModelDeciderPiece selected model_type={model_type}, normalization_type={normalization_type}.",
            artifacts=artifacts,
        )

    @staticmethod
    def _pick_model(available_models):
        if not available_models:
            return None
        preferred = ["xgb_regressor_model", "linear_regression_model", "eda_rule_baseline"]
        for candidate in preferred:
            if candidate in available_models:
                return candidate
        return available_models[0]
