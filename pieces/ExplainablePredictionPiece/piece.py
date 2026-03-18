from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class ExplainablePredictionPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running ExplainablePredictionPiece (template).")

        return OutputModel(
            message="ExplainablePredictionPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
