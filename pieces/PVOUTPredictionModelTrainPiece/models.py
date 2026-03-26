from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
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


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., trained model URI, training metrics).",
    )
