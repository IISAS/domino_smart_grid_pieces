import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Synthetic data generation config.\n"
            "Expected keys:\n"
            "- `dataset_type`: one of `solargis`, `microstep`, `shmu`, `okte`, `battery`, `machine`.\n"
            "- `output_mode`: `batch_sample` or `realtime_stream`.\n"
            "- `records_count`: number of generated rows (default 20).\n"
            "- `time_step_minutes`: step between timestamps (default 15).\n"
            "- `interval_ms`: realtime interval hint in milliseconds (default 1000).\n"
            "- Optional: `start_at` (ISO datetime), `seed` (int), `timezone_offset_hours` (number)."
        ),
    )

    @field_validator("payload", mode="before")
    @classmethod
    def _coerce_payload(cls, value):
        if value is None:
            return "{}"
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def payload_as_dict(self) -> dict:
        try:
            parsed = json.loads(self.payload) if self.payload else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Generated records and generation metadata.",
    )
