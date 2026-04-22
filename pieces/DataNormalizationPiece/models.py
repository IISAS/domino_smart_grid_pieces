from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalization_type: str | None = Field(
        default=None,
        description="Normalization type: `logaritmic`, `exponential`, `min_max`, or `z_score`.",
    )
    features: list[str] | None = Field(
        default=None,
        description="Optional list of feature/column names to normalize.",
    )
    dataframe: str | None = Field(
        default=None,
        description="Optional JSON object (or upstream object) representing input dataframe-like data.",
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
        out = self.model_dump(exclude_none=True)
        if "normalization_type" in out and "type" not in out:
            out["type"] = out["normalization_type"]
        return out

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., normalized dataset URI, fitted scaler params).",
    )
