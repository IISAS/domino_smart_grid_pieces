from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExplainSpec(BaseModel):
    """One trained model's explanation request."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(
        default=None,
        description="Stable identifier (used in per-model artifacts). "
        "Falls back to `model_path` basename / a generated index.",
    )
    explain: bool | None = Field(default=None, description="Enable explainability run.")
    explain_method: str | None = Field(
        default=None, description="`lime` or `shap`."
    )
    use_diagnostic_loss: bool | None = Field(
        default=None, description="Enable diagnostic heatmap artifacts."
    )
    model_path: str | None = Field(default=None, description="Model checkpoint path.")
    data_path: str | None = Field(default=None, description="Explanation dataset path.")
    feature_columns: list[str] | None = Field(
        default=None, description="Feature columns expected by the model."
    )
    target_column: str | None = Field(
        default=None, description="Optional target column name."
    )


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    explanations: list[ExplainSpec] | None = Field(
        default=None,
        description=(
            "Array of explanation requests to process in one piece run. Each entry "
            "produces its own per-model artifacts. When omitted, the scalar fields "
            "below are used as a single-entry fallback for backward compatibility."
        ),
    )
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
    model_config = ConfigDict(protected_namespaces=())

    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "`input_payload` on no-op. On run: aggregated `per_model = {model_id: {...}}` "
            "plus first-entry top-level keys for back-compat."
        ),
    )
