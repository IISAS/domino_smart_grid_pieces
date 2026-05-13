from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_type: str | None = Field(
        default=None, description="Problem type for model selection."
    )
    horizon: int | None = Field(default=None, description="Prediction horizon.")
    available_models: list[str] = Field(
        default_factory=list, description="Available model identifiers."
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Echoed feature columns (so downstream trainer/inference can wire it).",
    )
    target_column: str | None = Field(
        default=None, description="Echoed target column name (default `PVOUT`)."
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
    model_type: str = Field(
        default="xgb_regressor_model",
        description="Selected model type (consumable upstream → trainer.model_type).",
    )
    normalization_type: str = Field(
        default="none",
        description="Recommended normalization (consumable upstream → normalization.normalization_type).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Echoed feature columns (consumable upstream → trainer / inference).",
    )
    target_column: str = Field(
        default="PVOUT",
        description="Echoed target column (consumable upstream → trainer.target_column).",
    )
    decision_path: str | None = Field(
        default=None, description="Path to the on-disk decision.json artifact."
    )
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., selected model id/version, decision rationale).",
    )
