from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class EvaluateMLModelPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload or {}
        self.logger.info("Running EvaluateMLModelPiece.")

        if not payload:
            return OutputModel(
                message="EvaluateMLModelPiece template executed (no-op).",
                artifacts={"input_payload": payload},
            )

        evaluation_option = (
            payload.get("evaluation_option")
            or payload.get("evaluation_type")
            or payload.get("mode")
            or "normal"
        )
        evaluation_option = str(evaluation_option).lower()

        pred_df = (
            payload.get("pred_df")
            or payload.get("predictions_df")
            or payload.get("predictions")
            or payload.get("df")
        )

        plot = bool(payload.get("plot", False))
        baseline_id = int(payload.get("baseline_id", 1))

        # Lazy import: heavy deps only loaded when evaluation is requested.
        from .utils.error_evaluator import ErrorEvaluator

        evaluator = ErrorEvaluator()
        metrics = {}

        if pred_df is None:
            raise ValueError(
                "evaluation requires `payload['pred_df']` (or `predictions_df`/`predictions`/`df`)."
            )

        if evaluation_option == "normal":
            metrics = evaluator.evaluate(
                pred_df=pred_df,
                true_baseline_df=None,
                y_true=None,
                baseline_id=baseline_id,
                plot=plot,
            )
        elif evaluation_option in {"errorcorrection", "error_correction", "correction"}:
            y_true = payload.get("y_true")
            true_baseline_df = payload.get("true_baseline_df") or payload.get(
                "baseline_df"
            )

            if y_true is not None:
                metrics = evaluator.evaluate(
                    pred_df=pred_df,
                    y_true=y_true,
                    true_baseline_df=None,
                    baseline_id=baseline_id,
                    plot=plot,
                )
            elif true_baseline_df is not None:
                metrics = evaluator.evaluate(
                    pred_df=pred_df,
                    true_baseline_df=true_baseline_df,
                    y_true=None,
                    baseline_id=baseline_id,
                    plot=plot,
                )
            else:
                raise ValueError(
                    "errorcorrection evaluation requires either `payload['y_true']` "
                    "or `payload['true_baseline_df']`."
                )
        else:
            raise ValueError(
                "evaluation_option must be one of: normal, errorcorrection."
            )

        return OutputModel(
            message="EvaluateMLModelPiece executed.",
            artifacts={
                "input_payload": payload,
                "evaluation_option": evaluation_option,
                "metrics": metrics,
            },
        )
