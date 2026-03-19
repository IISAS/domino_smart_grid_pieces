from domino.testing import piece_dry_run
import pytest


def test_explainable_prediction_piece_smoke():
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_explainable_prediction_piece_explain_missing_model_or_data_raises():
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
