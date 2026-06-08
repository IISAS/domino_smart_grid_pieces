import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

from domino.testing import piece_dry_run

from .piece import _build_records, _gti_from_ghi, _solar_position

_FAKE_HOURLY = {
    "time": ["2024-06-01T10:00", "2024-06-01T11:00", "2024-06-01T12:00"],
    "shortwave_radiation": [400.0, 600.0, 700.0],
    "direct_normal_irradiance": [350.0, 500.0, 620.0],
    "diffuse_radiation": [80.0, 100.0, 90.0],
    "temperature_2m": [18.0, 21.0, 23.0],
    "relative_humidity_2m": [55.0, 50.0, 48.0],
    "wind_speed_10m": [3.0, 4.0, 3.5],
    "wind_gusts_10m": [5.0, 6.0, 5.5],
    "wind_direction_10m": [180.0, 200.0, 190.0],
    "surface_pressure": [1013.0, 1012.0, 1011.0],
}

_FAKE_RESPONSE = {"hourly": _FAKE_HOURLY}

_BASE_INPUT = {
    "latitude": 48.15,
    "longitude": 17.11,
    "start_date": "2024-06-01",
    "end_date": "2024-06-01",
    "pvout_peak_kw": 5.0,
    "panel_tilt": 30.0,
}


@patch("pieces.SolarGISDataGeneratorPiece.piece._fetch_open_meteo", return_value=_FAKE_RESPONSE)
def test_solargis_smoke(mock_fetch):
    output = piece_dry_run("SolarGISDataGeneratorPiece", _BASE_INPUT)
    assert "file_path" in output
    assert output.get("target_column") == "PVOUT"
    mock_fetch.assert_called_once()


@patch("pieces.SolarGISDataGeneratorPiece.piece._fetch_open_meteo", return_value=_FAKE_RESPONSE)
def test_solargis_json_output(_):
    output = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_format": "json"},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 3
    first = records[0]
    for col in ("Date", "Time", "GHI", "DNI", "DIF", "GTI", "SE", "SA", "PVOUT",
                "TEMP", "WS", "WG", "WD", "RH", "AP", "PVOUT_UNC_LOW", "PVOUT_UNC_HIGH"):
        assert col in first, f"Missing column: {col}"
    assert first["GHI"] == 400.0
    assert first["PVOUT"] >= 0.0
    assert first["PVOUT_UNC_LOW"] <= first["PVOUT"] <= first["PVOUT_UNC_HIGH"]


@patch("pieces.SolarGISDataGeneratorPiece.piece._fetch_open_meteo", return_value=_FAKE_RESPONSE)
def test_solargis_csv_output(_):
    output = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_format": "csv"},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert file_path.endswith(".csv")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
    assert len(rows) == 3
    assert "PVOUT" in reader.fieldnames
    assert "GHI" in reader.fieldnames


@patch("pieces.SolarGISDataGeneratorPiece.piece._fetch_open_meteo", return_value=_FAKE_RESPONSE)
def test_solargis_realtime_mode(_):
    output = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_mode": "realtime_stream"},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert "stream" in file_path


@patch("pieces.SolarGISDataGeneratorPiece.piece._fetch_open_meteo", return_value={"hourly": {}})
def test_solargis_empty_response(_):
    output = piece_dry_run("SolarGISDataGeneratorPiece", _BASE_INPUT)
    assert output.get("file_path") is None


def test_build_records_values():
    records = _build_records(_FAKE_HOURLY, lat=48.15, lon=17.11, pvout_peak_kw=5.0, panel_tilt=30.0)
    assert len(records) == 3
    for r in records:
        assert r["PVOUT"] >= 0.0
        assert r["GTI"] >= r["GHI"] * 1.0  # tilt bonus
        assert r["PVOUT_UNC_LOW"] <= r["PVOUT"] <= r["PVOUT_UNC_HIGH"]


def test_solar_position_night():
    from datetime import datetime
    elev, _ = _solar_position(datetime(2024, 6, 1, 2, 0), lat=48.15, lon=17.11)
    assert elev == 0.0


def test_solar_position_day():
    from datetime import datetime
    elev, _ = _solar_position(datetime(2024, 6, 1, 12, 0), lat=48.15, lon=17.11)
    assert elev > 0.0


def test_gti_from_ghi():
    gti = _gti_from_ghi(500.0, panel_tilt=30.0)
    assert gti > 500.0
    assert _gti_from_ghi(0.0, 30.0) == 0.0
