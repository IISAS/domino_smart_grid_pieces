from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    evaluation_option: str = Field(
        default="normal",
        description="Evaluation mode: `normal` or `errorcorrection`.",
    )
    baseline_id: int = Field(default=1, description="Baseline horizon id.")
    plot: bool = Field(default=False, description="Whether to generate plots/heatmaps.")
    pred_df: str | None = Field(
        default=None, description="Predictions payload as JSON object."
    )
    true_baseline_df: str | None = Field(
        default=None, description="Optional baseline payload as JSON object."
    )
    y_true: str | None = Field(
        default=None, description="Optional true values as JSON array."
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_payload(cls, data):
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            merged = dict(data["payload"])
            for key, value in data.items():
                if key != "payload":
                    merged[key] = value
            return merged
        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., metrics JSON, plots, report URI).",
    )
