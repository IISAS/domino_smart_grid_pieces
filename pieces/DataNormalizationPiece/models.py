from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description=(
            "Arbitrary inputs. For DataNormalizationPiece, expected keys include: "
            "`dataframe` (dataframe-like object), `type` (normalization type), and optional "
            "`features` (list of column names to normalize)."
        ),
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., normalized dataset URI, fitted scaler params).",
    )
