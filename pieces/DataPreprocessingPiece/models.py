from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    preprocessing_option: str | None = Field(
        default=None,
        description="One of: `none`, `prediction`, `correction`.",
    )
    data_path: str | None = Field(default=None, description="Optional input data path.")
    save_data_path: str | None = Field(
        default=None, description="Optional output path."
    )
    test_size: float | None = Field(default=None, description="Optional split size.")
    keep_datetime: bool | None = Field(
        default=None, description="Keep datetime column if supported."
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
        description="Optional outputs (e.g., cleaned dataset URI, dropped rows stats).",
    )
