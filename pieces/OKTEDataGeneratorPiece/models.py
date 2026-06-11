from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    records_count: int = Field(
        default=20,
        description="Number of records to generate.",
    )
    time_step_minutes: int = Field(
        default=15,
        description="Time step between generated records in minutes.",
    )
    output_mode: str = Field(
        default="batch_sample",
        description="One of: `batch_sample`, `realtime_stream`.",
    )
    output_format: str = Field(
        default="csv",
        description="One of: `json`, `csv`.",
        validation_alias=AliasChoices(
            "output_format",
            "outputFormat",
            "Output format",
            "Output Format",
            "export_format",
            "file_format",
        ),
    )
    interval_ms: int = Field(
        default=1000,
        description="Realtime interval hint in milliseconds (only used in realtime_stream mode).",
    )
    start_at: Optional[str] = Field(
        default=None,
        description="Optional start datetime in ISO format. Defaults to current UTC time.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for reproducible output.",
    )
    timezone_offset_hours: float = Field(
        default=1.0,
        description="Timezone offset in hours applied to Date/Time columns (CET = 1.0).",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_payload(cls, data):
        if not isinstance(data, dict):
            return data

        if isinstance(data.get("payload"), dict):
            merged = dict(data["payload"])
            for key, value in data.items():
                if key != "payload":
                    merged[key] = value
            data = merged

        if not data.get("output_format"):
            for alias in ("outputFormat", "Output format", "Output Format", "export_format", "file_format"):
                if alias in data and data.get(alias) is not None:
                    data["output_format"] = data.pop(alias)
                    break

        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True, exclude_unset=True)


class OutputModel(BaseModel):
    file_path: Optional[str] = Field(default=None, title="Dataset file path")
    target_column: str = Field(
        default="spot_price_eur_mwh",
        title="Target column",
        description="Suggested target column for downstream preprocessing/training pieces.",
    )
