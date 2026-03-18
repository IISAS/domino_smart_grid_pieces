from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description="Arbitrary inputs (e.g., raw dataset URI, schema, cleaning rules).",
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., cleaned dataset URI, dropped rows stats).",
    )
