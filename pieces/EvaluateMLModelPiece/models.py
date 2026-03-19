from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description=(
            "Arbitrary inputs.\n"
            "Supported evaluation keys (optional):\n"
            "- `evaluation_option`: 'normal' or 'errorcorrection' (default: 'normal').\n"
            "- `pred_df`: predictions DataFrame (required for evaluation).\n"
            "- `true_baseline_df`: baseline DataFrame (used for errorcorrection when `y_true` isn't provided).\n"
            "- `y_true`: array/series of true values (used for errorcorrection test split).\n"
            "- `baseline_id`: baseline pred_sequence_id (default: 1).\n"
            "- `plot`: bool (default: false; heatmaps only generated when true).\n"
        ),
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., metrics JSON, plots, report URI).",
    )
