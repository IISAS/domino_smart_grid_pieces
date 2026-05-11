import csv
import os
from pathlib import Path

import pytest
from domino.testing import piece_dry_run


def test_pvout_prediction_model_train_piece_smoke():
    output_data = piece_dry_run(
        "PVOUTPredictionModelTrainPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


@pytest.mark.parametrize(
    "model_type,required_modules,extra_model_params",
    [
        (
            "linear_regression_model",
            ["numpy", "pandas", "sklearn"],
            {},
        ),
        (
            "eda_rule_baseline",
            ["numpy", "pandas", "sklearn"],
            {},
        ),
        (
            "xgb_regressor_model",
            ["numpy", "pandas", "sklearn", "xgboost"],
            {"n_estimators": 5, "max_depth": 2, "verbosity": 0},
        ),
        (
            "interval_xgb_regressor_model",
            ["numpy", "pandas", "sklearn", "xgboost"],
            {"n_estimators": 5, "max_depth": 2, "verbosity": 0},
        ),
        (
            "tabpfn_regressor_model",
            ["numpy", "pandas", "sklearn", "tabpfn", "torch"],
            {},
        ),
    ],
)
def test_pvout_prediction_model_train_piece_all_models(
    model_type: str, required_modules: list[str], extra_model_params: dict
):
    if model_type == "tabpfn_regressor_model" and os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "Skipping TabPFN in CI HTTP dry-run mode (gated model/auth dependency)."
        )

    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            pytest.skip(f"{module_name} is not installed; skipping {model_type}.")

    csv_path = (
        Path(__file__).parent / "test_data" / "sample_training_data.csv"
    ).as_posix()
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    try:
        output_data = piece_dry_run(
            "PVOUTPredictionModelTrainPiece",
            {
                "payload": {
                    "model_type": model_type,
                    "model_params": extra_model_params,
                    "model_setup": {
                        "feature_columns": [
                            "GHI",
                            "DNI",
                            "DIF",
                            "GTI",
                            "SE",
                            "TEMP",
                            "WS",
                        ],
                        "target_column": "PVOUT",
                    },
                    "tabular_data": rows,
                }
            },
        )
    except RuntimeError as e:
        # TabPFN may be installed but unusable without gated model auth.
        if (
            model_type == "tabpfn_regressor_model"
            and "HuggingFace authentication error" in str(e)
        ):
            pytest.skip(
                "TabPFN gated model access is not configured in this environment."
            )
        raise

    artifacts = output_data["artifacts"]
    assert artifacts["trained_model"]["model_type"] == model_type
    assert artifacts["checkpoint_path"].endswith(".pkl")
