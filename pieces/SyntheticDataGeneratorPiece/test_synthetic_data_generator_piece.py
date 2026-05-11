import json
import os
import csv
from pathlib import Path

from domino.testing import piece_dry_run
import pytest


def test_synthetic_data_generator_piece_smoke():
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {},
    )
    assert "file_path" in output_data
    assert output_data["file_path"] is None


@pytest.mark.parametrize(
    "dataset_type,required_key",
    [
        ("solargis", "GHI"),
        ("microstep", "station_id"),
        ("shmu", "station_code"),
        ("shmi", "station_code"),
        ("okte", "spot_price_eur_mwh"),
        ("battery", "battery_id"),
        ("machine", "machine_id"),
    ],
)
def test_synthetic_data_generator_piece_batch_all_dataset_types(
    dataset_type, required_key
):
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {
            "dataset_type": dataset_type,
            "output_mode": "batch_sample",
            "records_count": 5,
            "time_step_minutes": 15,
            "seed": 123,
        },
    )

    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    # In HTTP dry-run mode, returned path can be container-local.
    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 5
    assert required_key in records[0]


def test_synthetic_data_generator_piece_realtime_mode():
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {
            "dataset_type": "SoalrGIS Dataset",
            "output_mode": "realtime_stream",
            "records_count": 3,
            "interval_ms": 1000,
            "time_step_minutes": 15,
            "seed": 7,
        },
    )
    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 3
    assert "GHI" in records[0]


def test_synthetic_data_generator_piece_solargis_csv_output():
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {
            "dataset_type": "solargis",
            "output_mode": "batch_sample",
            "output_format": "csv",
            "records_count": 3,
            "time_step_minutes": 15,
            "seed": 123,
        },
    )
    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".csv")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        assert len(rows) == 3
        assert reader.fieldnames == [
            "Date",
            "Time",
            "GHI",
            "DNI",
            "DIF",
            "GTI",
            "SE",
            "SA",
            "PVOUT",
            "TEMP",
            "WS",
            "WG",
            "WD",
            "RH",
            "AP",
            "PVOUT_UNC_LOW",
            "PVOUT_UNC_HIGH",
        ]


def test_synthetic_data_generator_piece_csv_with_title_cased_json_key():
    """Domino / JSON-schema UIs may send `Output format` instead of `output_format`."""
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {
            "dataset_type": "solargis",
            "output_mode": "batch_sample",
            "Output format": "csv",
            "records_count": 2,
            "time_step_minutes": 15,
            "seed": 99,
        },
    )
    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".csv")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        assert len(rows) == 2
