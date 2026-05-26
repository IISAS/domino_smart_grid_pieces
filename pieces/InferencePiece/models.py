from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSpec(BaseModel):
    """One trained-model entry consumed by InferencePiece.

    A pipeline can wire several trainers into one Inference node by exposing
    them as a list of ModelSpec entries. Each entry is self-contained: it
    can override datetime/horizon columns, schema-strictness, etc.
    """

    model_config = ConfigDict(extra="allow")

    model_id: str | None = Field(
        default=None,
        description="Stable identifier for this model (used in forecast filenames and per-model artifacts). "
        "Falls back to the basename of `model_path` when omitted.",
    )
    mode: str | None = Field(
        default=None,
        description="Inference mode: `pvout_correction`, `price_ahead`, `price_level`.",
    )
    model_path: str | None = Field(default=None, description="Model artifact path.")
    data_path: str | None = Field(
        default=None,
        description="Path to input CSV (defaults to parent payload's data_path when missing).",
    )
    preprocessing_metadata_path: str | None = Field(
        default=None,
        description="Optional explicit metadata JSON next to the model checkpoint.",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns the model expects.",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name (for downstream metrics/aggregation).",
    )
    base_forecast_column: str | None = Field(
        default=None, description="Baseline forecast column name."
    )
    datetime_column: str | None = Field(
        default=None, description="Datetime column name."
    )
    horizon_column: str | None = Field(
        default=None, description="Horizon/id column name."
    )
    max_horizon: int | None = Field(
        default=None, description="Optional maximum horizon."
    )
    stages: list[dict] | None = Field(
        default=None,
        description="Optional nested staged inference for this entry (advanced).",
    )


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: list[ModelSpec] | None = Field(
        default=None,
        description=(
            "Array of trained models to evaluate in one Inference run. Each entry "
            "produces its own forecast CSV. When omitted, the scalar fields below are "
            "used as a single-entry fallback for backward compatibility."
        ),
    )
    mode: str | None = Field(
        default=None,
        description="Inference mode: `pvout_correction`, `price_ahead`, `price_level`.",
    )
    model_path: str | None = Field(default=None, description="Model artifact path.")
    data_path: str | None = Field(
        default=None,
        description="Path to input CSV (e.g. from preprocessor/normalization).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns the model expects (from preprocessor/decider/trainer).",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name (for downstream metrics/aggregation).",
    )
    datetime_column: str | None = Field(
        default=None, description="Datetime column name."
    )
    base_forecast_column: str | None = Field(
        default=None, description="Baseline forecast column name."
    )
    horizon_column: str | None = Field(
        default=None, description="Horizon/id column name."
    )
    max_horizon: int | None = Field(
        default=None, description="Optional maximum horizon."
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


class ForecastEntry(BaseModel):
    model_id: str = Field(description="Identifier of the model that produced this forecast.")
    model_path: str | None = Field(default=None, description="Source model checkpoint.")
    mode: str | None = Field(default=None, description="Inference mode used.")
    forecast_csv_path: str | None = Field(
        default=None,
        description="Path to forecast CSV for this model (consumable by EvaluateMLModel / aggregator).",
    )
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = Field(default=None)


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    forecasts: list[ForecastEntry] = Field(
        default_factory=list,
        description="Per-model forecast outputs. Length matches the number of input models.",
    )
    forecast_csv_path: str | None = Field(
        default=None,
        description=(
            "Back-compat scalar: path to the first entry in `forecasts`. Downstream nodes "
            "still wired to single-model inference keep working."
        ),
    )
    model_path: str | None = Field(
        default=None,
        description="Back-compat: echoed checkpoint path of the first model.",
    )
    data_path: str | None = Field(
        default=None,
        description="Echoed input data path shared across models (when uniform).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Back-compat: feature columns of the first model.",
    )
    target_column: str = Field(
        default="PVOUT",
        description="Back-compat: target column of the first model.",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "`input_payload` on no-op; `per_model` mapping `{model_id: {forecast, per_horizon, "
            "metadata, debug?}}` plus aggregate `stage_summaries` when stages are used."
        ),
    )
