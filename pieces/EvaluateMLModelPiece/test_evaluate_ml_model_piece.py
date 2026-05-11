from domino.testing import piece_dry_run
import pytest
import os


def test_evaluate_ml_model_piece_smoke():
    output_data = piece_dry_run(
        "EvaluateMLModelPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_evaluate_ml_model_piece_missing_pred_df_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"evaluation requires `payload\['pred_df'\]"):
        piece_dry_run(
            "EvaluateMLModelPiece",
            {"payload": {"evaluation_option": "normal"}},
        )


def test_evaluate_ml_model_piece_invalid_option_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"evaluation_option must be one of"):
        piece_dry_run(
            "EvaluateMLModelPiece",
            {
                "payload": {
                    "evaluation_option": "not_a_real_option",
                    "pred_df": {"pvout_error": [0.1], "pred_sequence_id": [1]},
                }
            },
        )
