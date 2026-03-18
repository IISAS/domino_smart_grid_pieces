from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class ElectricityPricePredictionModelTrainPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info(
            "Running ElectricityPricePredictionModelTrainPiece (template)."
        )

        return OutputModel(
            message="ElectricityPricePredictionModelTrainPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
