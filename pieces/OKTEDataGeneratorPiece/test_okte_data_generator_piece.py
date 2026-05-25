import csv
import json
import os
from pathlib import Path

import pytest
from domino.testing import piece_dry_run


def test_okte_data_generator_piece_smoke():
    output = piece_dry_run("OKTEDataGeneratorPiece", {})
    assert "file_path" in output
    assert output.get("target_column") == "spot_price_eur_mwh"


def test_okte_data_generator_piece_batch_json():
    output = piece_dry_run(
        "OKTEDataGeneratorPiece",
        {"records_count": 5, "time_step_minutes": 15, "output_format": "json", "seed": 42},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 5
    assert "spot_price_eur_mwh" in records[0]
    assert "Date" in records[0]
    assert "Time" in records[0]
    assert records[0]["market_area"] == "SK"


def test_okte_data_generator_piece_batch_csv():
    output = piece_dry_run(
        "OKTEDataGeneratorPiece",
        {"records_count": 4, "output_format": "csv", "seed": 7},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert file_path.endswith(".csv")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
    assert len(rows) == 4
    assert set(reader.fieldnames) == {
        "Date", "Time", "market_area",
        "imbalance_mw", "spot_price_eur_mwh",
        "scheduled_generation_mw", "actual_generation_mw",
    }


def test_okte_data_generator_piece_realtime_mode():
    output = piece_dry_run(
        "OKTEDataGeneratorPiece",
        {"records_count": 3, "output_mode": "realtime_stream", "seed": 1},
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert "stream" in file_path


def test_okte_data_generator_piece_seed_reproducibility():
    a = piece_dry_run("OKTEDataGeneratorPiece", {"records_count": 3, "seed": 99})
    b = piece_dry_run("OKTEDataGeneratorPiece", {"records_count": 3, "seed": 99})
    if os.environ.get("PIECES_IMAGES_MAP"):
        return
    records_a = json.loads(Path(a["file_path"]).read_text(encoding="utf-8"))
    records_b = json.loads(Path(b["file_path"]).read_text(encoding="utf-8"))
    assert records_a == records_b
