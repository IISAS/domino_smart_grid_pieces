from domino.base_piece import BasePiece

from .models import InputModel, OutputModel
from .utils.run_inference import run_inference, run_staged_inference


class InferencePiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running InferencePiece.")
        payload = input_data.payload or {}

        if not payload:
            return OutputModel(
                message="InferencePiece executed (no-op).",
                artifacts={"input_payload": payload},
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
        return OutputModel(
            message="InferencePiece executed.",
            artifacts=artifacts,
        )
