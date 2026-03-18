from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description="Arbitrary inputs (e.g., dataframe metadata, file paths/URIs, column mappings).",
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., normalized dataset URI, fitted scaler params).",
    )
