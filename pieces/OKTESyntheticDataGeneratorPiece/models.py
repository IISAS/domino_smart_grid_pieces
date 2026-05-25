from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

OUTPUT_FORMAT_ALIASES: dict[str, str] = {
    "json": "json",
    "csv": "csv",
}


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    output_mode: str | None = Field(
        default=None,
        description="One of: `batch_sample`, `realtime_stream`.",
    )
    output_format: str | None = Field(
        default=None,
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
    records_count: int = Field(
        default=20,
        description="Number of records to generate.",
    )
    time_step_minutes: int = Field(
        default=15,
        description="Time step between generated records in minutes.",
    )
    interval_ms: int = Field(
        default=1000,
        description="Realtime interval hint in milliseconds.",
    )
    start_at: str | None = Field(
        default=None,
        description="Optional start datetime in ISO format.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible synthetic data.",
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
            for alias in (
                "outputFormat",
                "Output format",
                "Output Format",
                "export_format",
                "file_format",
            ):
                if alias in data and data.get(alias) is not None:
                    data["output_format"] = data.pop(alias)
                    break

        output_mode = data.get("output_mode")
        if output_mode is not None:
            data["output_mode"] = str(output_mode).strip().lower()

        output_format = data.get("output_format")
        if output_format is not None:
            normalized = str(output_format).strip().lower()
            data["output_format"] = OUTPUT_FORMAT_ALIASES.get(normalized, normalized)

        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True, exclude_unset=True)


class OutputModel(BaseModel):
    file_path: Optional[str] = Field(default=None, title="Dataset file path")
