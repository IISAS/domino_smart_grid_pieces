from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_type: str | None = Field(
        default=None,
        description=(
            "Type of generated dataset: "
            "`solargis`, `microstep`, `shmu`, `okte`, `battery`, `machine` "
            "(also accepts long names like `SolarGIS Dataset`)."
        ),
    )
    output_mode: str = Field(
        default="batch_sample",
        description="Generation mode: `batch_sample` or `realtime_stream`.",
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
    timezone_offset_hours: float = Field(
        default=1.0,
        description="Timezone offset used by SolarGIS-like records.",
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


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Generated records and generation metadata.",
    )
