from domino.testing import piece_dry_run
import pytest
import os


def test_explainable_prediction_piece_smoke():
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_explainable_prediction_piece_diagnostic_skips_without_diagnostic_payload():
    """`use_diagnostic_loss=True` flows from the top-level toggle into each
    forecast entry; with no diagnostic payload available the per-model run
    records a `skipped` status instead of crashing."""
    output_data = piece_dry_run(
        "ExplainablePredictionPiece",
        {
            "payload": {
                "use_diagnostic_loss": True,
                "forecasts": [{"model_id": "diag_only"}],
                # Override the auto-derived `explain=True` so the entry runs
                # diagnostics-only and doesn't trip the "missing model+data" guard.
                "explanations": [{"model_id": "diag_only", "explain": False}],
            }
        },
    )
    per_model = output_data["artifacts"]["per_model"]
    assert "diag_only" in per_model
    assert per_model["diag_only"]["diagnostic_heatmaps"]["status"] == "skipped"


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
