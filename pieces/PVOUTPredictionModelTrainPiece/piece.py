from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class PVOUTPredictionModelTrainPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running PVOUTPredictionModelTrainPiece (template).")

        return OutputModel(
            message="PVOUTPredictionModelTrainPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
