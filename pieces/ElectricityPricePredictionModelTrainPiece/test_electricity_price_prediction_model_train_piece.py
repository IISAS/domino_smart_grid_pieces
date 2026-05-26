import json
import os
from pathlib import Path

import pytest
from domino.testing import piece_dry_run


def test_electricity_price_train_xgb(tmp_path):
    """Offline: numeric features + known price labels -> XGB model checkpoint."""
    rows = [
        {"load_kw": str(100.0 + i * 0.5), "spot_price_eur_mwh": str(50.0 + 0.1 * i)}
        for i in range(40)
    ]

    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "checkpoint_dir": str(tmp_path / "train_out"),
                "tabular_data": rows,
                "feature_columns": ["load_kw"],
                "target_column": "spot_price_eur_mwh",
                "model_type": "xgb_regressor_model",
                "model_params": {"n_estimators": 20, "max_depth": 3, "random_state": 0},
            }
        },
    )

    assert output_data["message"] is not None
    assert output_data["model_path"] is not None
    assert output_data["feature_columns"] == ["load_kw"]
    assert output_data["target_column"] == "spot_price_eur_mwh"

    art = output_data["artifacts"]
    assert "checkpoint_path" in art
    assert "preprocessing_metadata_path" in art

    if not os.environ.get("PIECES_IMAGES_MAP"):
        assert Path(art["checkpoint_path"]).is_file()

        import pickle
        data = pickle.loads(Path(art["checkpoint_path"]).read_bytes())
        model = data["trained_model_object"]
        assert hasattr(model, "predict")

        meta = json.loads(
            Path(art["preprocessing_metadata_path"]).read_text(encoding="utf-8")
        )
        assert meta["feature_columns"] == ["load_kw"]
        assert meta["target_column"] == "spot_price_eur_mwh"


def test_electricity_price_train_linear(tmp_path):
    """Offline: linear regression model variant."""
    rows = [
        {"load_kw": str(float(i)), "spot_price_eur_mwh": str(float(i) * 2)}
        for i in range(1, 41)
    ]

    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "checkpoint_dir": str(tmp_path / "train_linear"),
                "tabular_data": rows,
                "feature_columns": ["load_kw"],
                "target_column": "spot_price_eur_mwh",
                "model_type": "linear_regression_model",
            }
        },
    )

    assert output_data["model_path"] is not None
    art = output_data["artifacts"]
    assert Path(art["checkpoint_path"]).is_file() if not os.environ.get("PIECES_IMAGES_MAP") else True


def test_electricity_price_train_missing_feature_columns():
    """Missing feature_columns raises a clear error."""
    rows = [{"load_kw": "100", "spot_price_eur_mwh": "50"} for _ in range(5)]

    with pytest.raises(Exception):
        piece_dry_run(
            "ElectricityPricePredictionModelTrainPiece",
            {
                "payload": {
                    "tabular_data": rows,
                    "target_column": "spot_price_eur_mwh",
                    "model_type": "xgb_regressor_model",
                }
            },
        )


def test_electricity_price_train_default_target_column(tmp_path):
    """Default target_column resolves to spot_price_eur_mwh."""
    rows = [
        {"load_kw": str(float(i)), "spot_price_eur_mwh": str(float(i) + 10)}
        for i in range(1, 41)
    ]

    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "checkpoint_dir": str(tmp_path / "default_target"),
                "tabular_data": rows,
                "feature_columns": ["load_kw"],
                "model_type": "xgb_regressor_model",
                "model_params": {"n_estimators": 10},
            }
        },
    )

    assert output_data["target_column"] == "spot_price_eur_mwh"
