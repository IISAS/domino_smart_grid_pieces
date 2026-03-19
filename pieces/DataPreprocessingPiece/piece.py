from domino.base_piece import BasePiece

from .models import InputModel, OutputModel
from .utils.modes import preprocess_correction, preprocess_prediction


class DataPreprocessingPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running DataPreprocessingPiece.")

        payload = input_data.payload or {}
        preprocessing_option = (
            payload.get("preprocessing_option")
            or payload.get("preprocessing_type")
            or payload.get("mode")
            or "none"
        )
        preprocessing_option = str(preprocessing_option).lower()

        if preprocessing_option == "none":
            return OutputModel(
                message="DataPreprocessingPiece executed (none).",
                artifacts={"input_payload": payload},
            )

        if preprocessing_option == "prediction":
            result = preprocess_prediction(payload)
            return OutputModel(message=result["message"], artifacts=result["artifacts"])

        if preprocessing_option == "correction":
            result = preprocess_correction(payload)
            return OutputModel(message=result["message"], artifacts=result["artifacts"])

        raise ValueError(
            f"Invalid preprocessing option: {preprocessing_option}. "
            "Expected one of: none, prediction, correction."
        )
