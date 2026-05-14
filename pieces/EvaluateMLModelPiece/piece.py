import json
from pathlib import Path

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class EvaluateMLModelPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload_as_dict()
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

        # Allow CSV-path inputs (e.g. wired from InferencePiece.forecast_csv_path).
        if pred_df is None and payload.get("pred_df_path"):
            import pandas as pd  # type: ignore

            pred_df = pd.read_csv(payload["pred_df_path"]).to_dict(orient="records")

        plot = bool(payload.get("plot", False))
        baseline_id = int(payload.get("baseline_id", 1))
        forecast_column = str(payload.get("forecast_column") or "final_forecast")
        target_column = str(payload.get("target_column") or "PVOUT")

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
                forecast_column=forecast_column,
                target_column=target_column,
            )
        elif evaluation_option in {"errorcorrection", "error_correction", "correction"}:
            y_true = payload.get("y_true")
            true_baseline_df = payload.get("true_baseline_df") or payload.get(
                "baseline_df"
            )
            if true_baseline_df is None and payload.get("true_baseline_df_path"):
                import pandas as pd  # type: ignore

                true_baseline_df = pd.read_csv(payload["true_baseline_df_path"]).to_dict(
                    orient="records"
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

        metrics_path = str(Path(self.results_path) / "metrics.json")
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"evaluation_option": evaluation_option, "metrics": metrics}, f, indent=2, default=str)
        self.display_result = {"file_type": "txt", "file_path": metrics_path}

        return OutputModel(
            message="EvaluateMLModelPiece executed.",
            artifacts={
                "input_payload": payload,
                "evaluation_option": evaluation_option,
                "metrics": metrics,
                "metrics_path": metrics_path,
            },
        )
