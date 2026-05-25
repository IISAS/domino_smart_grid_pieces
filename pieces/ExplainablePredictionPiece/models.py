from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    explain: bool = Field(default=False, description="Enable explainability run.")
    explain_method: str | None = Field(
        default=None,
        description="`lime` or `shap`. Defaults to `shap` when `explain=True`.",
    )
    use_diagnostic_loss: bool = Field(
        default=False, description="Enable diagnostic heatmap artifacts."
    )

    model_path: str | None = Field(
        default=None,
        description=(
            "Path to a trained model checkpoint produced by an upstream trainer "
            "(consumed from `PVOUTPredictionModelTrainPiece.model_path` or "
            "`PVOUTErrorCorrectionModelTrainPiece.model_path`)."
        ),
    )
    data_path: str | None = Field(
        default=None,
        description="Path to input CSV/parquet used as the explanation dataset.",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns the model expects (from preprocessor/trainer).",
    )
    target_column: str | None = Field(
        default=None,
        description="Optional target column name (informational; not required).",
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
        description="Optional outputs (e.g., explanation report URI, attribution arrays).",
    )
