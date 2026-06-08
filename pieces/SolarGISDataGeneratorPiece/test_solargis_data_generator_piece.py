from __future__ import annotations

import importlib
import json
import os
import csv
from pathlib import Path
from unittest.mock import MagicMock

from domino.testing import piece_dry_run

# ---------------------------------------------------------------------------
# Module loader (same pattern as InferencePiece tests)
# ---------------------------------------------------------------------------


def _load_piece_module():
    for name in (
        "SolarGISDataGeneratorPiece.piece",
        "pieces.SolarGISDataGeneratorPiece.piece",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Could not import SolarGISDataGeneratorPiece.piece")


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

_MOCK_HOURLY = {
    "time": [
        "2024-06-21T10:00",
        "2024-06-21T11:00",
        "2024-06-21T12:00",
        "2024-06-21T13:00",
        "2024-06-21T14:00",
    ],
    "shortwave_radiation":      [600.0, 750.0, 800.0, 720.0, 500.0],
    "direct_normal_irradiance": [550.0, 700.0, 750.0, 670.0, 450.0],
    "diffuse_radiation":        [100.0, 120.0, 130.0, 115.0,  90.0],
    "temperature_2m":           [ 22.5,  23.1,  24.0,  23.8,  22.0],
    "relative_humidity_2m":     [ 55.0,  52.0,  50.0,  51.0,  54.0],
    "wind_speed_10m":           [  3.5,   4.0,   3.8,   4.2,   3.9],
    "wind_gusts_10m":           [  6.0,   7.0,   6.5,   7.2,   6.8],
    "wind_direction_10m":       [180.0, 190.0, 185.0, 195.0, 188.0],
    "surface_pressure":         [1013.0, 1012.5, 1012.0, 1011.8, 1012.2],
}

_MOCK_RESPONSE_JSON = {
    "latitude": 48.15,
    "longitude": 17.11,
    "timezone": "Europe/Bratislava",
    "utc_offset_seconds": 3600,
    "hourly": _MOCK_HOURLY,
}

_BASE_INPUT = {
    "latitude": 48.15,
    "longitude": 17.11,
    "start_date": "2024-06-21",
    "end_date": "2024-06-21",
}


def _stub_get(*_args, **_kwargs) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _MOCK_RESPONSE_JSON
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_batch_json_output(monkeypatch):
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_format": "json"},
    )

    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")
    assert output_data["target_column"] == "PVOUT"

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 5
    first = records[0]
    for col in ("GHI", "DNI", "DIF", "GTI", "SE", "SA", "PVOUT",
                "TEMP", "WS", "WG", "WD", "RH", "AP",
                "PVOUT_UNC_LOW", "PVOUT_UNC_HIGH"):
        assert col in first, f"Missing column: {col}"
    assert first["GHI"] == 600.0
    assert first["PVOUT_UNC_LOW"] <= first["PVOUT"] <= first["PVOUT_UNC_HIGH"]


def test_batch_csv_output(monkeypatch):
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_format": "csv"},
    )

    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".csv")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        assert len(rows) == 5
        assert reader.fieldnames == [
            "Date", "Time",
            "GHI", "DNI", "DIF", "GTI",
            "SE", "SA",
            "PVOUT",
            "TEMP", "WS", "WG", "WD", "RH", "AP",
            "PVOUT_UNC_LOW", "PVOUT_UNC_HIGH",
        ]


def test_realtime_stream_mode(monkeypatch):
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "output_mode": "realtime_stream"},
    )

    assert output_data["file_path"] is not None
    assert "stream" in output_data["file_path"]


def test_pvout_calculation(monkeypatch):
    """PVOUT = pvout_peak_kw * (GHI / 1000) * 0.75 for each record."""
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "pvout_peak_kw": 10.0},
    )

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(output_data["file_path"]).read_text(encoding="utf-8"))
    for record, ghi in zip(records, _MOCK_HOURLY["shortwave_radiation"]):
        expected = round(10.0 * (ghi / 1000.0) * 0.75, 3)
        assert record["PVOUT"] == expected, f"Expected PVOUT={expected}, got {record['PVOUT']}"


def test_solar_elevation_at_noon(monkeypatch):
    """Solar elevation at 12:00 in summer at 48°N must be positive."""
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run("SolarGISDataGeneratorPiece", _BASE_INPUT)

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(output_data["file_path"]).read_text(encoding="utf-8"))
    noon = next(r for r in records if r["Time"] == "12:00")
    assert noon["SE"] > 0.0


def test_title_cased_output_format_key(monkeypatch):
    """Domino UI may send `Output format` instead of `output_format`."""
    mod = _load_piece_module()
    monkeypatch.setattr(mod.requests, "get", _stub_get)

    output_data = piece_dry_run(
        "SolarGISDataGeneratorPiece",
        {**_BASE_INPUT, "Output format": "csv"},
    )

    assert output_data["file_path"] is not None
    assert output_data["file_path"].endswith(".csv")
