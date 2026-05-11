from domino.testing import piece_dry_run
import os
import pytest
import pandas as pd
import sys
from pathlib import Path

from pieces.DataPreprocessingPiece.utils.preprocessor_utils import (
    ensure_datetime_column,
    preprocess_solargis_data,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.modes import preprocess_prediction


def test_data_preprocessing_piece_smoke():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_data_preprocessing_piece_none_mode():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {"preprocessing_option": "none"}},
    )
    assert output_data["message"].endswith("(none).")


def test_data_preprocessing_piece_none_mode_alias():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {"mode": "none"}},
    )
    assert output_data["message"].endswith("(none).")


def test_data_preprocessing_piece_invalid_option_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"Invalid preprocessing option"):
        piece_dry_run(
            "DataPreprocessingPiece",
            {"payload": {"preprocessing_option": "does_not_exist"}},
        )


def test_preprocess_prediction_infers_features_when_missing():
    payload = {
        "dataframe": pd.DataFrame(
            {
                "datetime": ["2026-05-07 10:00:00", "2026-05-07 11:00:00"],
                "GHI": [50.0, 60.0],
                "DIF": [10.0, 15.0],
                "SE": [20.0, 25.0],
                "PVOUT": [30.0, 35.0],
            }
        ),
        "preprocessing_option": "prediction",
    }

    result = preprocess_prediction(payload)
    features = result["artifacts"]["features"]

    assert "PVOUT" not in features
    assert "GHI" in features


def test_preprocess_prediction_supports_solargis_date_time_columns():
    payload = {
        "dataframe": pd.DataFrame(
            {
                "Date": ["07.05.2026", "07.05.2026"],
                "Time": ["10:00", "11:00"],
                "GHI": [50.0, 60.0],
                "DIF": [10.0, 15.0],
                "SE": [20.0, 25.0],
                "PVOUT": [30.0, 35.0],
            }
        ),
        "preprocessing_option": "prediction",
        "keep_datetime": True,
    }

    result = preprocess_prediction(payload)
    features = result["artifacts"]["features"]

    assert "datetime" in features
    assert "GHI" in features


def test_ensure_datetime_column_from_datetime_schema():
    data = pd.DataFrame(
        {
            "datetime": ["11.05.2026 13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = ensure_datetime_column(data)
    assert "datetime" in out.columns
    assert str(out["datetime"].dtype).startswith("datetime64")


def test_ensure_datetime_column_from_date_time_schema():
    data = pd.DataFrame(
        {
            "Date": ["11.05.2026"],
            "Time": ["13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = ensure_datetime_column(data)
    assert "datetime" in out.columns
    assert str(out["datetime"].dtype).startswith("datetime64")


def test_ensure_datetime_column_missing_schema_raises():
    data = pd.DataFrame({"GHI": [100], "DIF": [10], "SE": [20], "PVOUT": [1.0]})
    with pytest.raises(ValueError, match=r"either a `datetime` column"):
        ensure_datetime_column(data)


def test_preprocess_solargis_data_accepts_date_time_schema():
    data = pd.DataFrame(
        {
            "Date": ["11.05.2026"],
            "Time": ["13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = preprocess_solargis_data(data)
    assert "datetime" in out.columns
    assert "hour_of_day" in out.columns
