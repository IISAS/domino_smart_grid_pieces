from domino.testing import piece_dry_run
import pytest
import os


def test_evaluate_ml_model_piece_smoke():
    output_data = piece_dry_run(
        "EvaluateMLModelPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_evaluate_ml_model_piece_missing_pred_df_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"evaluation requires `payload\['pred_df'\]"):
        piece_dry_run(
            "EvaluateMLModelPiece",
            {"payload": {"evaluation_option": "normal"}},
        )


def test_evaluate_ml_model_piece_invalid_option_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"evaluation_option must be one of"):
        piece_dry_run(
            "EvaluateMLModelPiece",
            {
                "payload": {
                    "evaluation_option": "not_a_real_option",
                    "pred_df": {"pvout_error": [0.1], "pred_sequence_id": [1]},
                }
            },
        )


def test_evaluate_auto_derives_evaluations_from_forecasts_list(tmp_path):
    """
    When `EvaluateMLModelPiece` is wired at the LIST level to
    `InferencePiece.OutputModel.forecasts`, each entry's `pred_df_path`,
    `target_column`, and `forecast_column` should be auto-derived — no
    per-entry literal strings required.
    """
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "Host tmp_path fixtures aren't visible inside the dry-run container."
        )
    import csv

    # Two tiny forecast CSVs, one per model, with predictions vs truth columns
    # that match what InferencePiece's `pvout_correction` and `price_level`
    # modes emit.
    pvout_csv = tmp_path / "forecast_pvout.csv"
    with open(pvout_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "base_forecast", "correction", "final_forecast", "PVOUT"])
        writer.writerow(["2026-05-07T10:00:00", 4.0, 0.1, 4.1, 4.0])
        writer.writerow(["2026-05-07T10:15:00", 4.2, -0.05, 4.15, 4.2])

    price_csv = tmp_path / "forecast_price.csv"
    with open(price_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "base_forecast", "correction", "final_forecast", "spot_price_eur_mwh"])
        writer.writerow(["2026-05-07T10:00:00", 0.0, 80.0, 80.0, 82.0])
        writer.writerow(["2026-05-07T10:15:00", 0.0, 81.0, 81.0, 79.0])

    output_data = piece_dry_run(
        "EvaluateMLModelPiece",
        {
            "payload": {
                "forecasts": [
                    {
                        "model_id": "pvout",
                        "mode": "pvout_correction",
                        "forecast_csv_path": str(pvout_csv),
                        "target_column": "PVOUT",
                    },
                    {
                        "model_id": "price",
                        "mode": "price_level",
                        "forecast_csv_path": str(price_csv),
                        "target_column": "spot_price_eur_mwh",
                    },
                ]
            }
        },
    )

    metrics_list = output_data["metrics"]
    assert len(metrics_list) == 2

    by_id = {m["model_id"]: m for m in metrics_list}
    assert set(by_id) == {"pvout", "price"}

    # pvout entry: mode=pvout_correction → forecast_column auto-derived to
    # "correction", not "final_forecast" (which would just echo the truth).
    pvout_metrics = by_id["pvout"]["metrics"]
    assert "mae" in pvout_metrics
    assert "rmse" in pvout_metrics

    # price entry: mode=price_level → forecast_column auto-derived to
    # "final_forecast".
    price_metrics = by_id["price"]["metrics"]
    assert "mae" in price_metrics
    assert price_metrics.get("n") == 2


def test_evaluate_overrides_take_precedence_over_forecast_defaults(tmp_path):
    """When `evaluations` contains an override matching `model_id`, the
    override fields replace auto-derived defaults from the forecast bind."""
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "Host tmp_path fixtures aren't visible inside the dry-run container."
        )
    import csv

    csv_path = tmp_path / "forecast_pvout.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # `pvout_correction` mode auto-picks `correction`; override picks
        # `final_forecast` instead and the assertion ensures we read THAT column.
        writer.writerow(["datetime", "correction", "final_forecast", "PVOUT"])
        writer.writerow(["2026-05-07T10:00:00", 999.0, 4.1, 4.0])
        writer.writerow(["2026-05-07T10:15:00", 999.0, 4.15, 4.2])

    output_data = piece_dry_run(
        "EvaluateMLModelPiece",
        {
            "payload": {
                "forecasts": [
                    {
                        "model_id": "pvout",
                        "mode": "pvout_correction",
                        "forecast_csv_path": str(csv_path),
                        "target_column": "PVOUT",
                    }
                ],
                "evaluations": [
                    {"model_id": "pvout", "forecast_column": "final_forecast"}
                ],
            }
        },
    )

    metrics_list = output_data["metrics"]
    assert len(metrics_list) == 1
    # MAE measured against final_forecast (≈ 0.025), NOT correction (which
    # would be ~995). If override didn't apply we'd see a huge MAE.
    mae = metrics_list[0]["metrics"]["mae"]
    assert mae < 1.0, f"override didn't apply — MAE={mae} (got from correction column?)"
