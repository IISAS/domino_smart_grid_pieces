from domino.testing import piece_dry_run


def test_inference_piece_smoke():
    output_data = piece_dry_run(
        "InferencePiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
