from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class ExplainablePredictionPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload or {}
        self.logger.info("Running ExplainablePredictionPiece.")

        # Explainability config
        explain_cfg = payload.get("explainability") or {}
        explain_method = payload.get("explain_method") or explain_cfg.get("method")
        explain_enabled = bool(payload.get("explain", False) or explain_method)

        model = payload.get("model") or payload.get("trained_model")
        data = payload.get("data") or payload.get("eval_data") or payload.get("X_y")

        artifacts: dict = {"input_payload": payload}

        if explain_enabled:
            mode = explain_cfg.get("mode") or payload.get("mode") or "regression"
            method = (explain_method or "").lower()
            if method not in {"lime", "shap"}:
                raise ValueError("explain_method must be 'lime' or 'shap'")

            from .utils.explainability import run_explainability

            if model is None or data is None:
                raise ValueError(
                    "Explainability requires payload['model'] and payload['data']"
                )

            explain_result = run_explainability(
                model=model, data=data, method=method, mode=mode, cfg=explain_cfg
            )
            artifacts["explainability"] = explain_result

        # Optional diagnostic heatmaps for diagnostic-weighted error correction.
        diag_enabled = bool(
            payload.get("use_diagnostic_loss") or explain_cfg.get("use_diagnostic_loss")
        )
        if diag_enabled:
            from .utils.diagnostics import maybe_build_diagnostic_heatmaps

            diag_artifacts = maybe_build_diagnostic_heatmaps(payload)
            artifacts["diagnostic_heatmaps"] = diag_artifacts

        return OutputModel(
            message="ExplainablePredictionPiece executed.",
            artifacts=artifacts,
        )
