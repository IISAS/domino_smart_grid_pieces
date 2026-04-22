from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_path: str | None = Field(default=None, description="Input CSV path.")
    csv_path: str | None = Field(default=None, description="Alias for input CSV path.")
    output_dir: str | None = Field(default=None, description="Output directory.")
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
        return self.model_dump(exclude_none=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "Training outputs: `model_path`, `preprocessing_metadata_path`, `train_metrics`, "
            "`train_rows`, `fallback_used`; optional `enriched_csv_path`."
        ),
    )
