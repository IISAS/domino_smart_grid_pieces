from domino.testing import piece_dry_run


def test_evaluate_ml_model_piece_smoke():
    output_data = piece_dry_run(
        "EvaluateMLModelPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
