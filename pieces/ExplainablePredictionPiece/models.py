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


class ForecastBinding(BaseModel):
    """Structural mirror of `InferencePiece.OutputModel.forecasts[*]`.

    Lets a single upstream edge `ExplainablePrediction.forecasts ← Inference.forecasts`
    auto-populate per-model explanations without manual per-entry wiring.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(default=None)
    model_path: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    forecast_csv_path: str | None = Field(default=None)
    data_path: str | None = Field(default=None)
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = Field(default=None)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    forecasts: list[ForecastBinding] | None = Field(
        default=None,
        description=(
            "Per-model forecast entries — wire in one click from "
            "`InferencePiece.OutputModel.forecasts`. When provided, each entry's "
            "`model_path`, `data_path`, `feature_columns`, `target_column` drive "
            "the explanation run for that model. Use the `explanations` field "
            "below only to override specific entries."
        ),
    )
    explanations: list[ExplainSpec] | None = Field(
        default=None,
        description=(
            "Optional explicit per-entry overrides. Match by `model_id` to the "
            "forecast entries above. When `forecasts` is not wired, this becomes "
            "the primary input (legacy single-target mode)."
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
