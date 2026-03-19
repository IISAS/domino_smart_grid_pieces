from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description=(
            "Arbitrary inputs for evaluation/explainability.\n"
            "Supported (optional) keys:\n"
            "- `model`: trained model object.\n"
            "- `data`: evaluation dataset as `pd.DataFrame`, `(X, y)` tuple, or dict `{X, y?, feature_names?}`.\n"
            "- `explainability`: `{method: 'lime'|'shap', mode: 'regression'|'classification', ...}`.\n"
            "- `explain_method`: shortcut for `explainability.method`.\n"
            "- `use_diagnostic_loss`: if true, attempts to build diagnostic heatmaps.\n"
            "- `diagnostic`: dict with precomputed diagnostic arrays (see `EvaluateMLModelPiece._maybe_build_diagnostic_heatmaps`).\n"
            "- `x_train`: optional DataFrame used to derive `hour_of_day` for heatmaps."
        ),
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., metrics JSON, plots, report URI).",
    )
