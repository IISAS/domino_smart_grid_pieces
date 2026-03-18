from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class InferencePiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running InferencePiece (template).")

        return OutputModel(
            message="InferencePiece template executed (no-op).",
            artifacts={"input_payload": input_data.payload},
        )
