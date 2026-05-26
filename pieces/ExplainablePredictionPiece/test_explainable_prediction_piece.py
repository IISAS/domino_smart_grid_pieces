from domino.testing import piece_dry_run
import pytest
import os


def test_explainable_prediction_piece_smoke():
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_explainable_prediction_piece_explain_missing_model_or_data_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(
        ValueError,
        match=r"Explainability requires payload\['model'\] and payload\['data'\]",
    ):
        piece_dry_run(
            "ExplainablePredictionPiece",
            {
                "payload": {
                    "explain": True,
                    "explain_method": "shap",
                    "explainability": {"mode": "regression"},
                    # model + data intentionally omitted
                }
            },
        )


def test_explainable_prediction_piece_diagnostic_skips_without_diagnostic_payload():
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {"payload": {"use_diagnostic_loss": True}},
    )
    assert "diagnostic_heatmaps" in output_data["artifacts"]
    assert output_data["artifacts"]["diagnostic_heatmaps"]["status"] == "skipped"


def test_explainable_auto_derives_explanations_from_forecasts_list():
    """
    When `ExplainablePredictionPiece` is wired at the LIST level to
    `InferencePiece.OutputModel.forecasts`, each entry should auto-derive its
    explanation request — model_path / data_path / feature_columns come
    straight from the forecast entry, and `explain=true` kicks in by default.

    We omit `model_path` here so the piece falls through to the
    "missing model" ValueError, proving the auto-derive path *runs the
    explainability branch* (i.e. `explain=true` was correctly auto-derived).
    """
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")

    with pytest.raises(
        ValueError,
        match=r"Explainability requires payload\['model'\] and payload\['data'\]",
    ):
        piece_dry_run(
            "ExplainablePredictionPiece",
            {
                "payload": {
                    "forecasts": [
                        {
                            "model_id": "pvout",
                            # model_path + data_path intentionally absent so
                            # the auto-derived `explain=true` path raises with
                            # the documented error message instead of crashing
                            # on a bogus file path.
                            "feature_columns": ["GHI", "TEMP"],
                            "target_column": "PVOUT",
                        }
                    ]
                }
            },
        )
