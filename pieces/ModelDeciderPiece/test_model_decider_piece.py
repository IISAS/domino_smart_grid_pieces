from domino.testing import piece_dry_run


def test_model_decider_piece_smoke():
    output_data = piece_dry_run(
        "ModelDeciderPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_model_decider_strips_target_from_feature_columns():
    """
    When DataPreprocessing produces a merged Solargis+OKTE dataset, both target
    columns (`PVOUT` and `spot_price_eur_mwh`) flow into ModelDecider's
    `feature_columns`. Each per-target ModelDecider must echo a feature list
    that excludes its OWN target — otherwise the downstream trainer would see
    the ground truth as a predictor.
    """
    pvout_decider = piece_dry_run(
        "ModelDeciderPiece",
        {
            "payload": {
                "feature_columns": [
                    "GHI", "DNI", "TEMP",
                    "PVOUT",
                    "imbalance_mw", "spot_price_eur_mwh",
                ],
                "target_column": "PVOUT",
                "available_models": ["xgb_regressor_model"],
            }
        },
    )

    assert "PVOUT" not in pvout_decider["feature_columns"], (
        "ModelDecider must strip target_column from feature_columns"
    )
    # The other target stays — it's a valid predictor for PVOUT.
    assert "spot_price_eur_mwh" in pvout_decider["feature_columns"]
    assert pvout_decider["target_column"] == "PVOUT"

    price_decider = piece_dry_run(
        "ModelDeciderPiece",
        {
            "payload": {
                "feature_columns": [
                    "GHI", "DNI", "TEMP",
                    "PVOUT",
                    "imbalance_mw", "spot_price_eur_mwh",
                ],
                "target_column": "spot_price_eur_mwh",
                "available_models": ["xgb_regressor_model"],
            }
        },
    )

    assert "spot_price_eur_mwh" not in price_decider["feature_columns"]
    assert "PVOUT" in price_decider["feature_columns"]
    assert price_decider["target_column"] == "spot_price_eur_mwh"
