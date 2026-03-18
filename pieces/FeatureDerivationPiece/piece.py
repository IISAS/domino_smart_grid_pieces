from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class FeatureDerivationPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running FeatureDerivationPiece (template).")

        return OutputModel(
            message="FeatureDerivationPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
