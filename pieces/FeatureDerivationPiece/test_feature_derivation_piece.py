from domino.testing import piece_dry_run


def test_feature_derivation_piece_smoke():
    output_data = piece_dry_run(
        "FeatureDerivationPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
