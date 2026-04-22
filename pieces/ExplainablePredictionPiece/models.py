import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Inputs for generating explanations as JSON object.\n"
            "Optional keys:\n"
            "- `model`: trained model object.\n"
            "- `data`: evaluation dataset as `pd.DataFrame`, `(X, y)` tuple, or dict `{X, y?, feature_names?}`.\n"
            "- `explainability`: `{method: 'lime'|'shap', mode: 'regression'|'classification', ...}`.\n"
            "- `explain_method`: shortcut for `explainability.method`.\n"
            "- `use_diagnostic_loss`: if true, build diagnostic heatmaps from `payload['diagnostic']`.\n"
            "- `diagnostic`: dict with precomputed diagnostic arrays used by diagnostic heatmaps.\n"
            "- `x_train`: optional DataFrame used to derive `hour_of_day` for heatmaps."
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
        description="Optional outputs (e.g., explanation report URI, attribution arrays).",
    )
