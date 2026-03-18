from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class DataNormalizationPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running DataNormalizationPiece (template).")

        return OutputModel(
            message="DataNormalizationPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
