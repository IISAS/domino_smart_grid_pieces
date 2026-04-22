import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Arbitrary inputs as JSON object.\n"
            "Supported evaluation keys (optional):\n"
            "- `evaluation_option`: 'normal' or 'errorcorrection' (default: 'normal').\n"
            "- `pred_df`: predictions DataFrame (required for evaluation).\n"
            "- `true_baseline_df`: baseline DataFrame (used for errorcorrection when `y_true` isn't provided).\n"
            "- `y_true`: array/series of true values (used for errorcorrection test split).\n"
            "- `baseline_id`: baseline pred_sequence_id (default: 1).\n"
            "- `plot`: bool (default: false; heatmaps only generated when true).\n"
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
        description="Optional outputs (e.g., metrics JSON, plots, report URI).",
    )
