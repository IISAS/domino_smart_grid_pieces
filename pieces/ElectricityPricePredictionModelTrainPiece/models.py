from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    data_path: str | None = Field(default=None, description="Input CSV path.")
    csv_path: str | None = Field(default=None, description="Alias for input CSV path.")
    output_dir: str | None = Field(default=None, description="Output directory.")
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used for training (consumed from preprocessor/decider).",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name (defaults to `PVOUT`).",
    )
    model_filename: str | None = Field(
        default=None, description="Model output filename."
    )
    save_enriched_csv: bool | None = Field(
        default=None, description="Save enriched CSV when target_source=okte."
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
        description="Path to saved model file (consumable upstream → inference.model_path).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns used at training time (consumable upstream → inference.feature_columns).",
    )
    target_column: str = Field(
        default="price_eur_mwh",
        description="Target column used at training time.",
    )
    preprocessing_metadata_path: str | None = Field(
        default=None,
        description="Path to preprocessing_metadata.json (consumable upstream → inference.preprocessing_metadata_path).",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "Training outputs: `model_path`, `preprocessing_metadata_path`, `train_metrics`, "
            "`train_rows`, `fallback_used`; optional `enriched_csv_path`."
        ),
    )
