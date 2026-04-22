import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Training inputs for PVOUT error-correction model.\n"
            "Expected keys:\n"
            "- `model_type`: one of `error_correction_xgb_regressor_model`, "
            "`error_correction_residual_meta_xgb_regressor_model`, "
            "`error_correction_difficulty_weighted_xgb_regressor_model` "
            "(and lightweight built-ins `linear_regression` / `ridge_regression`).\n"
            "- `model_params`: constructor params passed to selected model.\n"
            "- `model_setup`: dict with `feature_columns`, optional `target_column`, "
            "and `pred_column` for error-correction models.\n"
            "- Input data provided by `data_path` (CSV) or `tabular_data` (list[dict] / dict-of-lists).\n"
            "- Optional `checkpoint_dir` for saved model checkpoint."
        ),
    )

    @field_validator("payload", mode="before")
    @classmethod
    def _coerce_payload(cls, value):
        if value is None:
            return "{}"
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def payload_as_dict(self) -> dict:
        try:
            parsed = json.loads(self.payload) if self.payload else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., trained corrector URI, evaluation metrics).",
    )
