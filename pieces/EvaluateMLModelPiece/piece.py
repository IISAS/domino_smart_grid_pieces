from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class EvaluateMLModelPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running EvaluateMLModelPiece (template).")
        return OutputModel(
            message="EvaluateMLModelPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
