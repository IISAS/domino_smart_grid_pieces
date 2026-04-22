import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Preprocessing inputs as JSON object. Expected keys:\n"
            "- `preprocessing_option`: one of `none`, `prediction`, `correction` (default: `none`).\n"
            "- For `prediction`: provide either `dataframe` (preferred) or `data_path`, "
            "plus `save_data_path` (optional), `flag_each_day`, `preprocessor_features`, "
            "and `keep_datetime`.\n"
            "- For `correction`: provide either `dataframe` (preferred) or `data_path`, "
            "plus `save_data_path` (optional), `flag_each_day`, `preprocessor_features` "
            "and optional `test_size` / `load_all_data`."
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
        description="Optional outputs (e.g., cleaned dataset URI, dropped rows stats).",
    )
