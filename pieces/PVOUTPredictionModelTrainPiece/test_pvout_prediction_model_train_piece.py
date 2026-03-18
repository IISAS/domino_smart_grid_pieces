from domino.testing import piece_dry_run


def test_pvout_prediction_model_train_piece_smoke():
    output_data = piece_dry_run(
        "PVOUTPredictionModelTrainPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
