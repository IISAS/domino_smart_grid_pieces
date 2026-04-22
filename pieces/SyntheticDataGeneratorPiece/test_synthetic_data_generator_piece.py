from domino.testing import piece_dry_run
import pytest


def test_synthetic_data_generator_piece_smoke():
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


@pytest.mark.parametrize(
    "dataset_type,required_key",
    [
        ("solargis", "GHI"),
        ("microstep", "station_id"),
        ("shmu", "station_code"),
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
            "payload": {
                "dataset_type": dataset_type,
                "output_mode": "batch_sample",
                "records_count": 5,
                "time_step_minutes": 15,
                "seed": 123,
            }
        },
    )

    artifacts = output_data["artifacts"]
    assert artifacts["dataset_type"] == dataset_type
    assert artifacts["output_mode"] == "batch_sample"
    assert len(artifacts["records"]) == 5
    assert required_key in artifacts["records"][0]


def test_synthetic_data_generator_piece_realtime_mode():
    output_data = piece_dry_run(
        "SyntheticDataGeneratorPiece",
        {
            "payload": {
                "dataset_type": "SoalrGIS Dataset",
                "output_mode": "realtime_stream",
                "records_count": 3,
                "interval_ms": 1000,
                "time_step_minutes": 15,
                "seed": 7,
            }
        },
    )
    artifacts = output_data["artifacts"]
    assert artifacts["dataset_type"] == "solargis"
    assert artifacts["output_mode"] == "realtime_stream"
    assert len(artifacts["records"]) == 3
    assert artifacts["stream_hint"]["interval_ms"] == 1000
