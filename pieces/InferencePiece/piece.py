from pathlib import Path

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel
from .utils.run_inference import run_inference, run_staged_inference


class InferencePiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running InferencePiece.")
        payload = input_data.payload_as_dict()

        if not payload:
            return OutputModel(
                message="InferencePiece executed (no-op).",
                artifacts={"input_payload": payload},
            )

        if not payload.get("forecast_output_csv_path"):
            payload["forecast_output_csv_path"] = str(
                Path(self.results_path) / "forecast.csv"
            )

        if payload.get("stages"):
            artifacts = run_staged_inference(payload)
        elif not payload.get("mode"):
            return OutputModel(
                message="InferencePiece executed (no-op).",
                artifacts={"input_payload": payload},
            )
        else:
            artifacts = run_inference(payload)

        csv_path = (artifacts.get("forecast") or {}).get("csv_path")
        if csv_path:
            self.display_result = {"file_type": "txt", "file_path": csv_path}

        return OutputModel(
            message="InferencePiece executed.",
            artifacts=artifacts,
        )
