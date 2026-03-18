from domino.testing import piece_dry_run


def test_data_preprocessing_piece_smoke():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
