"""
Unit tests for InferencePiece without on-disk model artifacts.

Model loading is stubbed via monkeypatch until train pieces define a shared
serialization format; `model_path` in payload is only metadata for now.

Monkeypatch targets use the `InferencePiece.*` module path (not `pieces.InferencePiece.*`)
because `piece_dry_run` puts the `pieces/` directory on `sys.path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from domino.testing import piece_dry_run


class _StubPredictor:
    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, X):
        import numpy as np

        return np.full(len(X), self._value)


def test_inference_piece_smoke():
    output_data = piece_dry_run(
        "InferencePiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None
    assert "input_payload" in output_data["artifacts"]


def test_inference_piece_pvout_correction_stub_model(monkeypatch, tmp_path: Path):
    try:
        import numpy as np  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("numpy/pandas not installed")

    dummy_model_path = str(tmp_path / "model.pkl")

    monkeypatch.setattr(
        "InferencePiece.utils.run_inference.load_model_object",
        lambda _payload: _StubPredictor(2.0),
    )

    rows = [
        {
            "datetime": "2025-06-01 12:00:00",
            "pred_sequence_id": 1,
            "PVOUT": 100.0,
            "f1": 0.0,
            "f2": 0.0,
        }
    ]
    output_data = piece_dry_run(
        "InferencePiece",
        {
            "payload": {
                "mode": "pvout_correction",
                "model_path": dummy_model_path,
                "input": {"tabular_data": rows},
                "datetime_column": "datetime",
                "feature_columns": ["f1", "f2"],
                "base_forecast_column": "PVOUT",
                "horizon_column": "pred_sequence_id",
                "strict_schema": True,
            }
        },
    )

    assert output_data["message"] is not None
    forecast = output_data["artifacts"]["forecast"]
    assert forecast["columns"] == [
        "datetime",
        "pred_sequence_id",
        "base_forecast",
        "correction",
        "final_forecast",
    ]
    recs = forecast["inline_records"]
    assert len(recs) == 1
    assert recs[0]["base_forecast"] == 100.0
    assert recs[0]["correction"] == 2.0
    assert recs[0]["final_forecast"] == 102.0


def test_inference_piece_price_ahead_baseline_from_profile(monkeypatch, tmp_path: Path):
    """build_baseline_if_missing + price_profile_path (no serialized model file)."""
    try:
        import numpy as np  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("numpy/pandas not installed")

    dummy_model_path = str(tmp_path / "model.pkl")
    profile_path = tmp_path / "price_profile.csv"
    # 2025-06-01 12:00:00 -> Sunday dow=6, slot_15m = 12*4+0 = 48
    profile_path.write_text(
        "dow,slot_15m,avg_price_eur_mwh\n6,48,70.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "InferencePiece.utils.run_inference.load_model_object",
        lambda _payload: _StubPredictor(1.5),
    )

    rows = [
        {
            "datetime": "2025-06-01 12:00:00",
            "pred_sequence_id": 1,
            "a": 1.0,
            "b": 2.0,
        }
    ]
    output_data = piece_dry_run(
        "InferencePiece",
        {
            "payload": {
                "mode": "price_ahead",
                "model_path": dummy_model_path,
                "input": {"tabular_data": rows},
                "datetime_column": "datetime",
                "feature_columns": ["a", "b"],
                "base_forecast_column": "price_baseline",
                "horizon_column": "pred_sequence_id",
                "build_baseline_if_missing": True,
                "price_profile_path": str(profile_path),
            }
        },
    )

    recs = output_data["artifacts"]["forecast"]["inline_records"]
    assert len(recs) == 1
    assert recs[0]["base_forecast"] == 70.0
    assert recs[0]["correction"] == 1.5
    assert recs[0]["final_forecast"] == 71.5


def test_inference_piece_price_level_stub(monkeypatch, tmp_path: Path):
    try:
        import numpy as np  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("numpy/pandas not installed")

    dummy_model_path = str(tmp_path / "price.joblib")
    monkeypatch.setattr(
        "InferencePiece.utils.run_inference.load_model_object",
        lambda _payload: _StubPredictor(62.5),
    )
    output_data = piece_dry_run(
        "InferencePiece",
        {
            "payload": {
                "mode": "price_level",
                "model_path": dummy_model_path,
                "input": {
                    "tabular_data": [
                        {
                            "datetime": "2025-06-01 12:00:00",
                            "pred_sequence_id": 1,
                            "load_kw": 50.0,
                        }
                    ]
                },
                "feature_columns": ["load_kw"],
                "horizon_column": "pred_sequence_id",
            }
        },
    )
    rec = output_data["artifacts"]["forecast"]["inline_records"][0]
    assert rec["final_forecast"] == 62.5
    assert rec["correction"] == 62.5


def test_inference_piece_stages_pipeline_single_stage(monkeypatch, tmp_path: Path):
    try:
        import numpy as np  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("numpy/pandas not installed")

    monkeypatch.setattr(
        "InferencePiece.utils.run_inference.load_model_object",
        lambda _payload: _StubPredictor(42.0),
    )
    output_data = piece_dry_run(
        "InferencePiece",
        {
            "payload": {
                "input": {
                    "tabular_data": [
                        {
                            "datetime": "2025-06-01 12:00:00",
                            "pred_sequence_id": 1,
                            "f1": 1.0,
                        }
                    ]
                },
                "stages": [
                    {
                        "mode": "price_level",
                        "model_path": str(tmp_path / "m.joblib"),
                        "feature_columns": ["f1"],
                    }
                ],
            }
        },
    )
    art = output_data["artifacts"]
    assert art["forecast"]["inline_records"][0]["final_forecast"] == 42.0
    assert art["stage_summaries"][0]["mode"] == "price_level"
    assert art["metadata"]["pipeline_stages"] == 1
