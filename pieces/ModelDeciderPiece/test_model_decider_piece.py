from domino.testing import piece_dry_run


def test_model_decider_piece_smoke():
    output_data = piece_dry_run(
        "ModelDeciderPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
