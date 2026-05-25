import csv
import json
import os
from pathlib import Path

import pytest
from domino.testing import piece_dry_run

OKTE_FIELDS = {
    "timestamp_utc",
    "market_area",
    "imbalance_mw",
    "spot_price_eur_mwh",
    "scheduled_generation_mw",
    "actual_generation_mw",
}


def test_okte_synthetic_data_generator_piece_batch_json():
    output_data = piece_dry_run(
        "OKTESyntheticDataGeneratorPiece",
        {
            "output_mode": "batch_sample",
            "records_count": 5,
            "time_step_minutes": 15,
            "seed": 42,
        },
    )
    file_path = output_data["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 5
    assert OKTE_FIELDS.issubset(records[0].keys())
    assert records[0]["market_area"] == "SK"


def test_okte_synthetic_data_generator_piece_csv():
    output_data = piece_dry_run(
        "OKTESyntheticDataGeneratorPiece",
        {
            "output_mode": "batch_sample",
            "output_format": "csv",
            "records_count": 3,
            "time_step_minutes": 15,
            "seed": 7,
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
    assert OKTE_FIELDS.issubset(set(reader.fieldnames or []))


def test_okte_synthetic_data_generator_piece_realtime_mode():
    output_data = piece_dry_run(
        "OKTESyntheticDataGeneratorPiece",
        {
            "output_mode": "realtime_stream",
            "records_count": 4,
            "interval_ms": 500,
            "seed": 1,
        },
    )
    file_path = output_data["file_path"]
    assert file_path is not None
    assert "stream" in file_path

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert len(records) == 4


def test_okte_synthetic_data_generator_piece_seed_reproducibility():
    def run(seed):
        output = piece_dry_run(
            "OKTESyntheticDataGeneratorPiece",
            {"records_count": 3, "seed": seed},
        )
        if os.environ.get("PIECES_IMAGES_MAP"):
            return None
        return json.loads(Path(output["file_path"]).read_text(encoding="utf-8"))

    records_a = run(99)
    records_b = run(99)
    if records_a is not None:
        assert records_a == records_b
