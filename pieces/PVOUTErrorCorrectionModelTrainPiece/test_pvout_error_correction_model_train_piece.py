from domino.testing import piece_dry_run


def test_pvout_error_correction_model_train_piece_smoke():
    output_data = piece_dry_run(
        "PVOUTErrorCorrectionModelTrainPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
