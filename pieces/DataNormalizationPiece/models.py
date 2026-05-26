from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalization_type: str | None = Field(
        default="none",
        title="Normalization Type",
        description=(
            "One of: `none`, `min_max`, `z_score`, `logaritmic`, `exponential`. "
            "Use `none` (passthrough) for tree-based models like XGBoost. "
            "Usually wired upstream from `ModelDeciderPiece.Normalization Type`."
        ),
    )
    features: list[str] = Field(
        default_factory=list,
        title="Features",
        description=(
            "Optional list of column names to normalize. Empty = apply to all numeric columns. "
            "Click `+` and add column names like `GHI`, `TEMP`, `WS`."
        ),
    )
    data_path: str | None = Field(
        default=None,
        title="Data Path",
        description=(
            "Path to input CSV. Wire upstream from `DataPreprocessingPiece.Data Path` "
            "(the preprocessor's `preprocessed.csv`)."
        ),
    )
    dataframe: str | None = Field(
        default=None,
        title="Dataframe",
        description=(
            "Optional inline dataframe payload (JSON object). "
            "Leave empty when `Data Path` is provided."
        ),
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
        out = self.model_dump(exclude_none=True, exclude_defaults=True)
        if "normalization_type" in out and "type" not in out:
            out["type"] = out["normalization_type"]
        return out

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    data_path: str | None = Field(
        default=None,
        description="Path to normalized CSV (consumable upstream → trainer / inference).",
    )
    normalization_type: str = Field(
        default="none",
        description="Normalization type that was applied.",
    )
    features: list[str] = Field(
        default_factory=list,
        description="Feature columns that were normalized.",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., normalized dataset URI, fitted scaler params).",
    )
