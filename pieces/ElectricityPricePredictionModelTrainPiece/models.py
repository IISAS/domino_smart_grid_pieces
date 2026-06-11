from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSpec(BaseModel):
    """Typed bundle that matches `InferencePiece.ModelSpec`.

    Exposed as a single typed output so a downstream `InferencePiece.models[i]`
    entry can bind to one trainer in a single click. Defaults pick `price_level`
    mode (no baseline column) and the price target.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    model_path: str | None = Field(default=None)
    data_path: str | None = Field(default=None)
    preprocessing_metadata_path: str | None = Field(default=None)
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = Field(default=None)
    base_forecast_column: str | None = Field(default=None)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_type: str | None = Field(default=None, description="Training model type.")
    data_path: str | None = Field(default=None, description="Input CSV path.")
    csv_path: str | None = Field(default=None, description="Alias for input CSV path.")
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used for training (consumed from preprocessor/decider).",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name (defaults to `spot_price_eur_mwh`).",
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
    model_config = ConfigDict(protected_namespaces=())

    message: str = Field(description="Human-readable status message.")
    model_path: str | None = Field(
        default=None,
        description="Path to saved model checkpoint (consumable upstream → inference.model_path).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used at training time (consumable upstream → inference.feature_columns).",
    )
    target_column: str = Field(
        default="spot_price_eur_mwh",
        description="Target column used at training time.",
    )
    preprocessing_metadata_path: str | None = Field(
        default=None,
        description="Path to preprocessing_metadata.json (consumable upstream → inference.preprocessing_metadata_path).",
    )
    data_path: str | None = Field(
        default=None,
        description="Echoed input data path (consumable upstream → inference / explainable).",
    )
    model_spec: list[ModelSpec] | None = Field(
        default=None,
        description=(
            "Single-element list mirroring `InferencePiece.price_model` so the entire "
            "bundle binds in one click (`InferencePiece.price_model ← model_spec`). "
            "Defaults to `mode=price_level` and `target_column=spot_price_eur_mwh`."
        ),
    )
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., trained model URI, training metrics).",
    )
