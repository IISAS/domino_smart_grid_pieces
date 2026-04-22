from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str | None = Field(
        default=None,
        description="Inference mode: `pvout_correction`, `price_ahead`, `price_level`.",
    )
    model_path: str | None = Field(default=None, description="Model artifact path.")
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


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "No-op: `input_payload`. "
            "Run: `forecast` (inline_records, csv_path, columns), `per_horizon`, `metadata`, "
            "optional `debug`. Pipeline: additionally `stage_summaries`."
        ),
    )
