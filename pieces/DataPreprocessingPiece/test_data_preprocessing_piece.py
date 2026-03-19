from domino.testing import piece_dry_run
import pytest


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
    with pytest.raises(ValueError, match=r"Invalid preprocessing option"):
        piece_dry_run(
            "DataPreprocessingPiece",
            {"payload": {"preprocessing_option": "does_not_exist"}},
        )
