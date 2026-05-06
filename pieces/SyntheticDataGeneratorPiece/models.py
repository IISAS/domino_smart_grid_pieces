from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum
from typing import ClassVar, Optional


class DatasetType(str, Enum):
    solargis = "solargis"
    microstep = "microstep"
    shmu = "shmu"
    okte = "okte"
    battery = "battery"
    machine = "machine"


class OutputMode(str, Enum):
    batch_sample = "batch_sample"
    realtime_stream = "realtime_stream"


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_type: DatasetType = Field(
        default="solargis",
        title="Dataset Type",
        description="Type of generated dataset.",
    )
    output_mode: OutputMode = Field(
        default="batch_sample",
        title="Output Mode",
        description="Generation mode.",
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
        if not isinstance(data, dict):
            return data

        if isinstance(data.get("payload"), dict):
            merged = dict(data["payload"])
            for key, value in data.items():
                if key != "payload":
                    merged[key] = value
            data = merged

        dataset_type = data.get("dataset_type")
        if dataset_type is not None:
            normalized = str(dataset_type).strip().lower()
            data["dataset_type"] = cls._DATASET_TYPE_ALIASES.get(normalized, normalized)

        output_mode = data.get("output_mode")
        if output_mode is not None:
            data["output_mode"] = str(output_mode).strip().lower()

        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)


class OutputModel(BaseModel):
    file_path: Optional[str] = Field(default=None, title="Dataset file path")
