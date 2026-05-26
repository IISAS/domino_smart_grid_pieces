from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalSpec(BaseModel):
    """One model's evaluation request consumed by EvaluateMLModelPiece."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(
        default=None,
        description="Stable identifier (used in metrics filename). "
        "Falls back to `pred_df_path` basename / a generated index.",
    )
    evaluation_option: str | None = Field(
        default=None,
        description="`normal` or `errorcorrection`. Defaults to parent value.",
    )
    baseline_id: int | None = Field(default=None, description="Baseline horizon id.")
    plot: bool | None = Field(default=None, description="Generate plots/heatmaps.")
    forecast_column: str | None = Field(
        default=None, description="Predicted-value column."
    )
    target_column: str | None = Field(
        default=None, description="Ground-truth column in pred_df."
    )
    pred_df_path: str | None = Field(
        default=None,
        description="Path to predictions CSV (typically from InferencePiece forecast).",
    )
    true_baseline_df_path: str | None = Field(
        default=None,
        description="Path to true-baseline CSV for errorcorrection mode.",
    )
    pred_df: Any = Field(
        default=None, description="Inline predictions payload (JSON object/list/string)."
    )
    true_baseline_df: Any = Field(
        default=None, description="Optional baseline payload (JSON object/list/string)."
    )
    y_true: Any = Field(
        default=None, description="Optional true values (JSON array/string)."
    )


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    evaluations: list[EvalSpec] | None = Field(
        default=None,
        description=(
            "Array of evaluation requests to process in one piece run. Each entry "
            "produces its own metrics.json. When omitted, the scalar fields below are "
            "used as a single-entry fallback for backward compatibility."
        ),
    )
    evaluation_option: str = Field(
        default="normal",
        description="Evaluation mode: `normal` or `errorcorrection`.",
    )
    baseline_id: int = Field(default=1, description="Baseline horizon id.")
    plot: bool = Field(default=False, description="Whether to generate plots/heatmaps.")
    forecast_column: str = Field(
        default="final_forecast",
        title="Forecast Column",
        description=(
            "Name of the predicted-value column in pred_df. "
            "Defaults to `final_forecast` (from InferencePiece). "
            "Switch to `correction` when evaluating the raw model output in pvout_correction mode."
        ),
    )
    target_column: str = Field(
        default="PVOUT",
        title="Target Column",
        description="Name of the ground-truth column in pred_df. Default `PVOUT`.",
    )
    pred_df_path: str | None = Field(
        default=None,
        description="Path to predictions CSV (e.g. inference.forecast_csv_path).",
    )
    true_baseline_df_path: str | None = Field(
        default=None,
        description="Path to true-baseline CSV for errorcorrection mode.",
    )
    pred_df: Any = Field(
        default=None,
        description="Inline predictions payload (JSON object/list/string).",
    )
    true_baseline_df: Any = Field(
        default=None,
        description="Optional baseline payload (JSON object/list/string).",
    )
    y_true: Any = Field(
        default=None, description="Optional true values (JSON array/string)."
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
        # `exclude_unset` keeps explicitly-set defaults (e.g. evaluation_option="normal")
        # so the piece can distinguish "no input provided" from "user asked for the default".
        return self.model_dump(exclude_none=True, exclude_unset=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class MetricsEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(description="Identifier for this evaluation entry.")
    evaluation_option: str = Field(description="Evaluation mode used.")
    metrics_path: str = Field(description="Path to this entry's metrics JSON.")
    metrics: dict = Field(
        default_factory=dict, description="Computed metrics for this entry."
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    metrics: list[MetricsEntry] = Field(
        default_factory=list,
        description="Per-entry metrics. Length matches the number of evaluations.",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "`input_payload` on no-op. On run: aggregated `per_model = {model_id: metrics}` "
            "plus first-entry `metrics`, `metrics_path`, `evaluation_option` at top level for back-compat."
        ),
    )
