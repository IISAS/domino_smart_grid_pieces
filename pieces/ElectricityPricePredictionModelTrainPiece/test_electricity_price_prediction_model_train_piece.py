import datetime as dt
import json
import os
from pathlib import Path

import joblib
import pytest
from domino.testing import piece_dry_run
from zoneinfo import ZoneInfo

_BRAT = ZoneInfo("Europe/Bratislava")


def _skip_if_no_live_okte() -> None:
    if os.environ.get("SKIP_OKTE_INTEGRATION", "").lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_OKTE_INTEGRATION is set")


def test_electricity_price_train_column_target_xgb(tmp_path):
    """Offline: energy-side features + known price labels -> joblib XGB model."""
    rows = []
    base = dt.datetime(2024, 1, 1, 0, 0, 0)
    for i in range(40):
        t = base + dt.timedelta(minutes=15 * i)
        load_kw = 100.0 + i * 0.5
        rows.append(
            {
                "datetime": t.isoformat(sep=" "),
                "load_kw": str(load_kw),
                "price_eur_mwh": str(50.0 + 0.1 * load_kw + (i % 3)),
            }
        )

    out_dir = str(tmp_path / "train_out")
    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "output_dir": out_dir,
                "tabular_data": rows,
                "model_setup": {
                    "feature_columns": ["load_kw"],
                    "target_column": "price_eur_mwh",
                    "target_source": "column",
                    "datetime_column": "datetime",
                },
                "xgb_params": {"n_estimators": 20, "max_depth": 3, "random_state": 0},
            }
        },
    )

    art = output_data["artifacts"]
    assert "model_path" in art
    assert "preprocessing_metadata_path" in art
    assert art["train_rows"] >= 2
    assert "rmse" in art["train_metrics"]
    assert art["fallback_used"] is False

    # In HTTP dry-run mode artifacts can point to container-local paths.
    if os.environ.get("PIECES_IMAGES_MAP"):
        assert art["model_path"].endswith((".joblib", ".pkl"))
        assert art["preprocessing_metadata_path"].endswith(".json")
    else:
        model = joblib.load(art["model_path"])
        meta = json.loads(
            Path(art["preprocessing_metadata_path"]).read_text(encoding="utf-8")
        )
        assert meta["feature_columns_used"] == ["load_kw"]
        assert hasattr(model, "predict")


@pytest.mark.integration
def test_electricity_price_train_okte_labels_and_xgb(tmp_path):
    _skip_if_no_live_okte()

    now = dt.datetime.now(_BRAT).replace(second=0, microsecond=0, tzinfo=None)
    row_dt_past = (now - dt.timedelta(days=1)).replace(minute=(now.minute // 15) * 15)
    row_dt_near = (now + dt.timedelta(days=1)).replace(minute=(now.minute // 15) * 15)

    out_dir = str(tmp_path / "out_okte_train")
    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "output_dir": out_dir,
                "tabular_data": [
                    {
                        "datetime": row_dt_past.isoformat(sep=" "),
                        "load_kw": "120.5",
                    },
                    {
                        "datetime": row_dt_near.isoformat(sep=" "),
                        "load_kw": "88.0",
                    },
                ],
                "model_setup": {
                    "feature_columns": ["load_kw"],
                    "target_column": "price_eur_mwh",
                    "target_source": "okte",
                    "datetime_column": "datetime",
                },
                "xgb_params": {"n_estimators": 20, "max_depth": 3, "random_state": 0},
            }
        },
    )

    assert output_data["message"] is not None
    art = output_data["artifacts"]
    assert Path(art["model_path"]).is_file()
    assert art["train_rows"] == 2


@pytest.mark.integration
def test_electricity_price_train_okte_future_fallback(tmp_path):
    _skip_if_no_live_okte()

    now = dt.datetime.now(_BRAT).replace(second=0, microsecond=0, tzinfo=None)
    row_a = (now + dt.timedelta(days=15)).replace(minute=(now.minute // 15) * 15)
    row_b = (now + dt.timedelta(days=16)).replace(minute=(now.minute // 15) * 15)

    out_dir = str(tmp_path / "out_fallback_train")
    output_data = piece_dry_run(
        "ElectricityPricePredictionModelTrainPiece",
        {
            "payload": {
                "output_dir": out_dir,
                "tabular_data": [
                    {"datetime": row_a.isoformat(sep=" "), "load_kw": "10"},
                    {"datetime": row_b.isoformat(sep=" "), "load_kw": "20"},
                ],
                "model_setup": {
                    "feature_columns": ["load_kw"],
                    "target_source": "okte",
                },
            }
        },
    )

    art = output_data["artifacts"]
    assert art["fallback_used"] is True
    assert Path(art["model_path"]).is_file()
