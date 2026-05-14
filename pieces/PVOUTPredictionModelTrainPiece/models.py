from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_type: str | None = Field(default=None, description="Training model type.")
    data_path: str | None = Field(default=None, description="Input CSV path.")
    csv_path: str | None = Field(default=None, description="Alias for input CSV path.")
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used for training (consumed from preprocessor/decider).",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name (defaults to `PVOUT`).",
    )
    checkpoint_dir: str | None = Field(
        default=None, description="Optional checkpoint directory."
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
    model_path: str | None = Field(
        default=None,
        description="Path to trained model checkpoint (consumable upstream → inference.model_path).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used at training time (consumable upstream → inference.feature_columns).",
    )
    target_column: str = Field(
        default="PVOUT",
        description="Target column used at training time.",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., trained model URI, training metrics).",
    )
