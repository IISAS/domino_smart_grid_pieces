from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class EvaluateMLModelPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload or {}
        self.logger.info("Running EvaluateMLModelPiece.")

        # Keep this piece safe for smoke tests: only import heavy deps when needed.
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

            explain_result = self._run_explainability(
                model=model,
                data=data,
                method=method,
                mode=mode,
                cfg=explain_cfg,
            )
            artifacts["explainability"] = explain_result

        # Optional diagnostic heatmaps for diagnostic-weighted error correction.
        # We accept precomputed diagnostic arrays in payload because this repo
        # doesn't contain the full custom-loss implementation.
        diag_enabled = bool(
            payload.get("use_diagnostic_loss") or explain_cfg.get("use_diagnostic_loss")
        )
        if diag_enabled:
            diag_artifacts = self._maybe_build_diagnostic_heatmaps(
                model=model, payload=payload
            )
            artifacts["diagnostic_heatmaps"] = diag_artifacts

        if len(artifacts) == 1:
            return OutputModel(
                message="EvaluateMLModelPiece executed (no-op).",
                artifacts=artifacts,
            )

        return OutputModel(
            message="EvaluateMLModelPiece executed.",
            artifacts=artifacts,
        )

    @staticmethod
    def _run_explainability(model, data, method: str, mode: str, cfg: dict) -> dict:
        """
        Implements the provided Explainability module (LIME/SHAP dispatch) with lazy imports.
        """
        # Delegate the full implementation to utils to keep this piece readable.
        from .utils.explainability import run_explainability as _run

        return _run(model=model, data=data, method=method, mode=mode, cfg=cfg)

    @staticmethod
    def _maybe_build_diagnostic_heatmaps(model, payload: dict) -> dict:
        """
        Build diagnostic heatmaps similar to the provided notebook snippet.

        Because the underlying custom-loss helpers (derive_regime, diagnostic_weighted_grad_hess, etc.)
        are not part of this repo, this piece expects precomputed diagnostic arrays in payload:

        Required diagnostic keys (payload['diagnostic']):
        - train_horizons: 1D array-like (int)
        - train_regimes: 1D array-like (int)
        - w_i: 1D array-like (weights per sample)
        - grad_diag: 1D array-like (weighted grad per sample)
        - hess_diag: 1D array-like (weighted hess per sample)
        - y_tr_label: 1D array-like (true correction target)
        - y_tr_pred: 1D array-like (pred correction output)
        Optional unweighted (constraint-only) comparison:
        - grad_u: 1D
        - hess_u: 1D
        - apply_unweighted: bool (ignored if grad_u/hess_u missing)

        Also expects x_train with at least:
        - 'hour_of_day' or something to compute it
        """
        # Delegate the full implementation to utils to keep this piece readable.
        from .utils.diagnostics import maybe_build_diagnostic_heatmaps as _build

        return _build(payload)
