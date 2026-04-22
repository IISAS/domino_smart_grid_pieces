import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Training inputs for PVOUT prediction model.\n"
            "Expected keys:\n"
            "- `model_type`: one of `linear_regression_model`, `xgb_regressor_model`, "
            "`interval_xgb_regressor_model`, `eda_rule_baseline`, `tabpfn_regressor_model`.\n"
            "- `model_params`: constructor params for selected model.\n"
            "- `model_setup`: dict with required `feature_columns` and optional `target_column`.\n"
            "- Input data via `data_path` (CSV) or `tabular_data` (list[dict] / dict-of-lists).\n"
            "- Optional `checkpoint_dir` for model checkpoint."
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
        description="Optional outputs (e.g., trained model URI, training metrics).",
    )
