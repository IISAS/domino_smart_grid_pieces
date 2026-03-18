from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class DataPreprocessingPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running DataPreprocessingPiece (template).")

        return OutputModel(
            message="DataPreprocessingPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
