from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class ModelDeciderPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running ModelDeciderPiece (template).")

        return OutputModel(
            message="ModelDeciderPiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload_as_dict()},
        )
