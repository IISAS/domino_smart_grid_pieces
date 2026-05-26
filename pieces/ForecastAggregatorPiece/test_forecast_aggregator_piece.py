import os
from pathlib import Path

import pytest
from domino.testing import piece_dry_run


def test_forecast_aggregator_piece_smoke():
    output = piece_dry_run("ForecastAggregatorPiece", {"payload": {}})
    assert output["message"] is not None
    assert output["aggregated_csv_path"] is None


def test_forecast_aggregator_joins_two_models(tmp_path: Path):
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "Host tmp_path is not visible inside the piece container during HTTP dry-run."
        )
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    csv_a = tmp_path / "model_a.csv"
    csv_b = tmp_path / "model_b.csv"
    pd.DataFrame(
        {
            "datetime": ["2025-06-01 12:00:00", "2025-06-01 13:00:00"],
            "pred_sequence_id": [1, 1],
            "final_forecast": [100.0, 110.0],
        }
    ).to_csv(csv_a, index=False)
    pd.DataFrame(
        {
            "datetime": ["2025-06-01 12:00:00", "2025-06-01 13:00:00"],
            "pred_sequence_id": [1, 1],
            "final_forecast": [105.0, 108.0],
        }
    ).to_csv(csv_b, index=False)

    output = piece_dry_run(
        "ForecastAggregatorPiece",
        {
            "payload": {
                "forecasts": [
                    {"model_id": "alpha", "forecast_csv_path": str(csv_a)},
                    {"model_id": "beta", "forecast_csv_path": str(csv_b)},
                ]
            }
        },
    )

    assert output["aggregated_csv_path"] is not None
    assert sorted(output["model_ids"]) == ["alpha", "beta"]
    assert output["n_rows"] == 2

    aggregated = pd.read_csv(output["aggregated_csv_path"])
    assert "pred_alpha" in aggregated.columns
    assert "pred_beta" in aggregated.columns
    assert sorted(aggregated["pred_alpha"].tolist()) == [100.0, 110.0]


def test_forecast_aggregator_adds_actual_column(tmp_path: Path):
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "Host tmp_path is not visible inside the piece container during HTTP dry-run."
        )
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    fc = tmp_path / "fc.csv"
    actual = tmp_path / "actual.csv"
    pd.DataFrame(
        {
            "datetime": ["2025-06-01 12:00:00"],
            "pred_sequence_id": [1],
            "final_forecast": [100.0],
        }
    ).to_csv(fc, index=False)
    pd.DataFrame(
        {
            "datetime": ["2025-06-01 12:00:00"],
            "pred_sequence_id": [1],
            "PVOUT": [97.0],
        }
    ).to_csv(actual, index=False)

    output = piece_dry_run(
        "ForecastAggregatorPiece",
        {
            "payload": {
                "forecasts": [
                    {
                        "model_id": "main",
                        "forecast_csv_path": str(fc),
                        "target_column": "PVOUT",
                    }
                ],
                "actual_csv_path": str(actual),
                "include_diff": True,
            }
        },
    )

    aggregated = pd.read_csv(output["aggregated_csv_path"])
    assert "actual_PVOUT" in aggregated.columns
    assert "diff_main" in aggregated.columns
    assert aggregated["actual_PVOUT"].iloc[0] == 97.0
    assert aggregated["diff_main"].iloc[0] == pytest.approx(3.0)
