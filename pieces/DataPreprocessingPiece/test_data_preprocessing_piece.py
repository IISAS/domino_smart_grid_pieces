from domino.testing import piece_dry_run
import os
import pytest
import pandas as pd

from pieces.DataPreprocessingPiece.utils.preprocessor_utils import (
    ensure_datetime_column,
    preprocess_solargis_data,
)


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
