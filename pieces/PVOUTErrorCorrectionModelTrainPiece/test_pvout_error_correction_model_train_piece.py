from pathlib import Path
import csv

import pytest
from domino.testing import piece_dry_run


def test_pvout_error_correction_model_train_piece_smoke():
    output_data = piece_dry_run(
        "PVOUTErrorCorrectionModelTrainPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


@pytest.mark.parametrize(
    "model_type",
    [
        "error_correction_xgb_regressor_model",
        "error_correction_residual_meta_xgb_regressor_model",
        "error_correction_difficulty_weighted_xgb_regressor_model",
    ],
)
def test_pvout_error_correction_model_train_piece_csv_pipeline_all_models(
    model_type: str,
):
    try:
        import numpy  # noqa: F401  # type: ignore
    except ImportError:
        pytest.skip("numpy is not installed; skipping training pipeline tests.")
    try:
        import pandas  # noqa: F401  # type: ignore
        import sklearn  # noqa: F401  # type: ignore
        import xgboost  # noqa: F401  # type: ignore
    except ImportError:
        pytest.skip(
            "pandas/sklearn/xgboost is not installed; skipping native model training tests."
        )

    csv_path = (
        Path(__file__).parent / "test_data" / "sample_training_data.csv"
    ).as_posix()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        # baseline prediction intentionally biased from target
        row["PVOUT_PRED"] = str(float(row["PVOUT"]) * 0.9)
        row["pred_sequence_id"] = str((idx % 3) + 1)

    output_data = piece_dry_run(
        "PVOUTErrorCorrectionModelTrainPiece",
        {
            "payload": {
                "model_type": model_type,
                "model_params": {"n_estimators": 5, "max_depth": 2, "verbosity": 0},
                "model_setup": {
                    "feature_columns": [
                        "GHI",
                        "DNI",
                        "DIF",
                        "GTI",
                        "SE",
                        "TEMP",
                        "WS",
                        "RH",
                        "AP",
                        "pred_sequence_id",
                    ],
                    "target_column": "PVOUT",
                    "pred_column": "PVOUT_PRED",
                },
                "tabular_data": rows,
            }
        },
    )

    artifacts = output_data["artifacts"]
    assert artifacts["trained_model"]["model_type"] == model_type
    assert artifacts["checkpoint_path"].endswith(".pkl")

    # Typed bundle (single-element list) for one-click upstream binding from
    # `InferencePiece.pvout_model`.
    spec_list = output_data["model_spec"]
    assert isinstance(spec_list, list) and len(spec_list) == 1
    spec = spec_list[0]
    assert spec["model_id"] == "pvout"
    assert spec["mode"] == "pvout_correction"
    assert spec["model_path"] == output_data["model_path"]
    assert spec["base_forecast_column"] == "PVOUT"
    assert spec["preprocessing_metadata_path"] == output_data["preprocessing_metadata_path"]


def test_pvout_error_correction_top_level_feature_columns_wiring():
    """
    Regression: when the workflow wires `feature_columns` / `target_column` /
    `pred_column` as top-level payload fields (the typed-output route Domino
    uses for upstream-piece bindings) instead of nesting them under
    `model_setup`, the piece must still train successfully. Previously this
    path silently handed XGBoost an empty feature DataFrame.
    """
    try:
        import numpy  # noqa: F401  # type: ignore
        import pandas  # noqa: F401  # type: ignore
        import sklearn  # noqa: F401  # type: ignore
        import xgboost  # noqa: F401  # type: ignore
    except ImportError:
        pytest.skip(
            "pandas/sklearn/xgboost is not installed; skipping native model training tests."
        )

    csv_path = (
        Path(__file__).parent / "test_data" / "sample_training_data.csv"
    ).as_posix()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        row["PVOUT_PRED"] = str(float(row["PVOUT"]) * 0.9)
        row["pred_sequence_id"] = str((idx % 3) + 1)

    output_data = piece_dry_run(
        "PVOUTErrorCorrectionModelTrainPiece",
        {
            "payload": {
                "model_type": "error_correction_xgb_regressor_model",
                "model_params": {"n_estimators": 5, "max_depth": 2, "verbosity": 0},
                # Top-level wiring (no `model_setup` block).
                "feature_columns": [
                    "GHI",
                    "DNI",
                    "DIF",
                    "GTI",
                    "SE",
                    "TEMP",
                    "WS",
                    "RH",
                    "AP",
                    "pred_sequence_id",
                ],
                "target_column": "PVOUT",
                "model_setup": {"pred_column": "PVOUT_PRED"},
                "tabular_data": rows,
            }
        },
    )

    assert (
        output_data["artifacts"]["trained_model"]["model_type"]
        == "error_correction_xgb_regressor_model"
    )
    assert output_data["model_path"].endswith(".pkl")
    assert output_data["feature_columns"]
    assert output_data["target_column"] == "PVOUT"
