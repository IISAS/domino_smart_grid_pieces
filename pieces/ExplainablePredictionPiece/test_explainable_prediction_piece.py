from domino.testing import piece_dry_run


def test_explainable_prediction_piece_smoke():
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
