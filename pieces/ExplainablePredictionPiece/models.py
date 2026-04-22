from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    explain: bool = Field(default=False, description="Enable explainability run.")
    explain_method: str | None = Field(default=None, description="`lime` or `shap`.")
    use_diagnostic_loss: bool = Field(
        default=False, description="Enable diagnostic heatmap artifacts."
    )
    model: str | None = Field(
        default=None, description="Optional model payload as JSON."
    )
    data: str | None = Field(default=None, description="Optional data payload as JSON.")
    x_train: str | None = Field(
        default=None, description="Optional x_train payload as JSON."
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
        return self.model_dump(exclude_none=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., explanation report URI, attribution arrays).",
    )
