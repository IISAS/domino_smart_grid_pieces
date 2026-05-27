from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


_INHERITED_KEYS = (
    "explain",
    "explain_method",
    "explainability",
    "use_diagnostic_loss",
    "mode",
)


def _slugify_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "model"
    return slug.strip("_") or "model"


def _model_id_for(entry: dict, index: int) -> str:
    candidate = entry.get("model_id")
    if candidate:
        return _slugify_id(str(candidate))
    model_path = entry.get("model_path")
    if model_path:
        return _slugify_id(Path(str(model_path)).stem)
    return f"model_{index + 1}"


def _derive_explanations_from_forecasts(payload: dict) -> list[dict]:
    """Auto-build an explanations list from an upstream `forecasts` bind.

    When `ExplainablePredictionPiece` is wired to `InferencePiece.forecasts`
    (at the list level), each ForecastEntry carries `model_path`, `data_path`,
    `feature_columns`, and `target_column` — everything we need to run SHAP /
    LIME per model without manual per-entry configuration. Any explicit
    overrides set on the `explanations` field still take precedence per
    `model_id`.
    """
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        return []

    overrides_by_id: dict[str, dict] = {}
    explicit = payload.get("explanations") or []
    if isinstance(explicit, list):
        for entry in explicit:
            if not isinstance(entry, dict):
                continue
            key = entry.get("model_id")
            if key:
                overrides_by_id[str(key)] = dict(entry)

    derived: list[dict] = []
    for index, forecast in enumerate(forecasts):
        if not isinstance(forecast, dict):
            continue
        model_id = forecast.get("model_id") or f"model_{index + 1}"
        defaults: dict[str, Any] = {
            "model_id": model_id,
            "model_path": forecast.get("model_path"),
            "data_path": forecast.get("data_path"),
            "feature_columns": list(forecast.get("feature_columns") or []),
            "target_column": forecast.get("target_column"),
            "explain": True,
        }
        override = overrides_by_id.get(str(model_id), {})
        for k, v in override.items():
            if v is not None and v != "":
                defaults[k] = v
        derived.append(defaults)
    return derived


def _normalize_explanations(payload: dict) -> list[dict]:
    # Upstream-bound `forecasts` list takes precedence — that's the canonical
    # path (Inference → Explainable via a single edge).
    derived = _derive_explanations_from_forecasts(payload)
    if derived:
        return derived

    explanations = payload.get("explanations")
    if isinstance(explanations, list) and explanations:
        return [dict(entry or {}) for entry in explanations]
    return []


def _entry_payload_for(entry: dict, parent: dict) -> dict:
    materialized: dict[str, Any] = {}
    for key in _INHERITED_KEYS:
        if key in parent and parent[key] is not None:
            materialized[key] = parent[key]
    materialized.update({k: v for k, v in entry.items() if v is not None})
    return materialized


def _run_single_explanation(entry_payload: dict) -> dict[str, Any]:
    """Execute one explanation request and return its artifacts dict."""
    explain_cfg = entry_payload.get("explainability") or {}
    explain_method = entry_payload.get("explain_method") or explain_cfg.get("method")
    explain_enabled = bool(entry_payload.get("explain", False) or explain_method)

    model = entry_payload.get("model") or entry_payload.get("trained_model")
    data = (
        entry_payload.get("data")
        or entry_payload.get("eval_data")
        or entry_payload.get("X_y")
    )

    if model is None and entry_payload.get("model_path"):
        from .utils.loader import load_model_object

        model = load_model_object(entry_payload)

    if data is None and (
        entry_payload.get("data_path")
        or entry_payload.get("csv_path")
        or entry_payload.get("tabular_data")
        or entry_payload.get("dataframe")
    ):
        from .utils.loader import load_input_dataframe

        df = load_input_dataframe(entry_payload)
        if df is not None:
            feature_columns = entry_payload.get("feature_columns") or []
            if feature_columns:
                missing = [c for c in feature_columns if c not in df.columns]
                if missing:
                    raise ValueError(
                        f"Missing feature columns in input data: {missing}"
                    )
                df = df[list(feature_columns)]
            data = df

    artifacts: dict[str, Any] = {"input_payload": entry_payload}

    if explain_enabled:
        mode = explain_cfg.get("mode") or entry_payload.get("mode") or "regression"
        method = (explain_method or "shap").lower()
        if method not in {"lime", "shap"}:
            raise ValueError("explain_method must be 'lime' or 'shap'")

        from .utils.explainability import run_explainability

        if model is None or data is None:
            raise ValueError(
                "Explainability requires payload['model'] and payload['data']"
            )

        artifacts["explainability"] = run_explainability(
            model=model, data=data, method=method, mode=mode, cfg=explain_cfg
        )

    diag_enabled = bool(
        entry_payload.get("use_diagnostic_loss")
        or explain_cfg.get("use_diagnostic_loss")
    )
    if diag_enabled:
        from .utils.diagnostics import maybe_build_diagnostic_heatmaps

        artifacts["diagnostic_heatmaps"] = maybe_build_diagnostic_heatmaps(
            entry_payload
        )

    return artifacts


class ExplainablePredictionPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload_as_dict()
        self.logger.info("Running ExplainablePredictionPiece.")

        entries = _normalize_explanations(payload)
        if not entries:
            return OutputModel(
                message="ExplainablePredictionPiece executed (no-op).",
                artifacts={"input_payload": payload},
            )

        per_model: dict[str, dict] = {}
        used_ids: set[str] = set()
        head_artifacts: dict[str, Any] | None = None
        for index, entry in enumerate(entries):
            base_id = _model_id_for(entry, index)
            model_id = base_id
            suffix = 2
            while model_id in used_ids:
                model_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(model_id)

            entry_payload = _entry_payload_for(entry, payload)
            artifacts = _run_single_explanation(entry_payload)
            per_model[model_id] = artifacts
            if head_artifacts is None:
                head_artifacts = artifacts

        message = (
            "ExplainablePredictionPiece executed."
            if len(entries) == 1
            else f"ExplainablePredictionPiece executed for {len(entries)} models."
        )

        combined: dict[str, Any] = {"input_payload": payload, "per_model": per_model}
        if head_artifacts:
            for key, value in head_artifacts.items():
                if key != "input_payload" and key not in combined:
                    combined[key] = value

        return OutputModel(message=message, artifacts=combined)
