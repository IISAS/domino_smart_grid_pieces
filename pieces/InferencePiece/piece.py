from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import ForecastEntry, InputModel, OutputModel
from .utils.run_inference import run_inference, run_staged_inference


# Keys copied from the parent payload onto each per-model stage payload when the
# entry doesn't override them. Lets users wire shared input via top-level fields
# and only specify the differing per-model values.
_INHERITED_KEYS = (
    "input",
    "tabular_data",
    "datetime_column",
    "horizon_column",
    "base_forecast_column",
    "max_horizon",
    "strict_schema",
    "missing_fill_value",
    "per_horizon_outputs",
    "return_debug",
    "build_baseline_if_missing",
    "price_profile_path",
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


def _normalize_models(payload: dict) -> list[dict]:
    """Return a list of per-model dicts. Falls back to a single scalar entry."""
    models = payload.get("models")
    if isinstance(models, list) and models:
        return [dict(entry or {}) for entry in models]
    scalar_keys = (
        "mode",
        "model_path",
        "data_path",
        "feature_columns",
        "target_column",
        "base_forecast_column",
        "datetime_column",
        "horizon_column",
        "max_horizon",
        "stages",
        "preprocessing_metadata_path",
    )
    scalar = {k: payload[k] for k in scalar_keys if k in payload and payload[k] is not None}
    if not scalar.get("mode") and not scalar.get("stages"):
        return []
    return [scalar]


def _stage_payload_for(entry: dict, parent: dict) -> dict:
    """Compose the payload run_inference expects for a single model entry."""
    stage: dict[str, Any] = {}
    for key in _INHERITED_KEYS:
        if key in parent and parent[key] is not None:
            stage[key] = parent[key]
    stage.update({k: v for k, v in entry.items() if v is not None})
    if "data_path" not in stage and parent.get("data_path") is not None:
        stage["data_path"] = parent["data_path"]
    return stage


class InferencePiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running InferencePiece.")
        payload = input_data.payload_as_dict()

        if not payload:
            return OutputModel(
                message="InferencePiece executed (no-op).",
                artifacts={"input_payload": payload},
            )

        entries = _normalize_models(payload)
        if not entries:
            echo_model_path = payload.get("model_path")
            echo_data_path = payload.get("data_path")
            echo_feature_columns = list(payload.get("feature_columns") or [])
            echo_target_column = str(payload.get("target_column") or "PVOUT")
            return OutputModel(
                message="InferencePiece executed (no-op).",
                model_path=echo_model_path,
                data_path=echo_data_path,
                feature_columns=echo_feature_columns,
                target_column=echo_target_column,
                artifacts={"input_payload": payload},
            )

        forecasts_dir = Path(self.results_path) / "forecasts"
        forecasts_dir.mkdir(parents=True, exist_ok=True)

        per_model_artifacts: dict[str, Any] = {}
        forecast_entries: list[ForecastEntry] = []
        used_ids: set[str] = set()

        for index, entry in enumerate(entries):
            base_id = _model_id_for(entry, index)
            model_id = base_id
            suffix = 2
            while model_id in used_ids:
                model_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(model_id)

            stage_payload = _stage_payload_for(entry, payload)
            stage_payload.setdefault(
                "forecast_output_csv_path", str(forecasts_dir / f"{model_id}.csv")
            )

            if stage_payload.get("stages"):
                artifacts = run_staged_inference(stage_payload)
            elif stage_payload.get("mode"):
                artifacts = run_inference(stage_payload)
            else:
                raise ValueError(
                    f"models[{index}]: provide `mode` (or nested `stages`) for inference."
                )

            csv_path = (artifacts.get("forecast") or {}).get("csv_path")
            forecast_entries.append(
                ForecastEntry(
                    model_id=model_id,
                    model_path=stage_payload.get("model_path"),
                    mode=stage_payload.get("mode"),
                    forecast_csv_path=csv_path,
                    # Echo the INPUT data path so ExplainablePrediction can
                    # re-feed the same rows to SHAP / LIME from a single edge.
                    data_path=(
                        stage_payload.get("data_path")
                        or entry.get("data_path")
                        or payload.get("data_path")
                    ),
                    feature_columns=list(
                        stage_payload.get("feature_columns")
                        or entry.get("feature_columns")
                        or []
                    ),
                    target_column=(
                        stage_payload.get("target_column")
                        or entry.get("target_column")
                    ),
                )
            )
            per_model_artifacts[model_id] = artifacts

        head = forecast_entries[0]
        head_artifacts = per_model_artifacts[head.model_id]
        head_csv = head.forecast_csv_path
        if head_csv:
            self.display_result = {"file_type": "txt", "file_path": head_csv}

        message = (
            "InferencePiece executed."
            if len(forecast_entries) == 1
            else f"InferencePiece executed for {len(forecast_entries)} models."
        )

        return OutputModel(
            message=message,
            forecasts=forecast_entries,
            forecast_csv_path=head_csv,
            model_path=head.model_path,
            data_path=payload.get("data_path"),
            feature_columns=head.feature_columns,
            target_column=str(head.target_column or payload.get("target_column") or "PVOUT"),
            artifacts={
                "per_model": per_model_artifacts,
                # Back-compat: surface the first model's artifacts at the top level so
                # legacy downstream wiring keeps working without traversing per_model.
                **{k: v for k, v in head_artifacts.items() if k != "input_payload"},
            },
        )
