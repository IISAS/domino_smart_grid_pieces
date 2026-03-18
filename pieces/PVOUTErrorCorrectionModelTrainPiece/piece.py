from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class PVOUTErrorCorrectionModelTrainPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running PVOUTErrorCorrectionModelTrainPiece (template).")

        return OutputModel(
            message="PVOUTErrorCorrectionModelTrainPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
