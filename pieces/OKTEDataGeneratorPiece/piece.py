import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel

TARGET_COLUMN = "spot_price_eur_mwh"

FIELDNAMES = [
    "Date",
    "Time",
    "market_area",
    "imbalance_mw",
    "spot_price_eur_mwh",
    "scheduled_generation_mw",
    "actual_generation_mw",
]


def _okte_record(ts: datetime, tz_offset_hours: float = 1.0) -> dict[str, Any]:
    local_ts = ts + timedelta(hours=tz_offset_hours)
    return {
        "Date": local_ts.strftime("%d.%m.%Y"),
        "Time": local_ts.strftime("%H:%M"),
        "market_area": "SK",
        "imbalance_mw": round(random.uniform(-280, 260), 3),
        "spot_price_eur_mwh": round(random.uniform(25, 220), 2),
        "scheduled_generation_mw": round(random.uniform(1200, 4200), 2),
        "actual_generation_mw": round(random.uniform(1200, 4200), 2),
    }


class OKTEDataGeneratorPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running OKTEDataGeneratorPiece.")

        records_count = input_data.records_count
        time_step_minutes = input_data.time_step_minutes
        output_mode = input_data.output_mode.strip().lower()
        output_format = input_data.output_format.strip().lower()
        interval_ms = input_data.interval_ms
        tz_offset_hours = input_data.timezone_offset_hours

        if records_count <= 0:
            raise ValueError("records_count must be > 0")
        if time_step_minutes <= 0:
            raise ValueError("time_step_minutes must be > 0")
        if output_mode not in {"batch_sample", "realtime_stream"}:
            raise ValueError("output_mode must be `batch_sample` or `realtime_stream`.")
        if output_format not in {"json", "csv"}:
            raise ValueError("output_format must be `json` or `csv`.")

        if input_data.seed is not None:
            random.seed(input_data.seed)

        if input_data.start_at:
            start_at = datetime.fromisoformat(input_data.start_at)
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
        else:
            start_at = datetime.now(tz=timezone.utc)

        step = timedelta(minutes=time_step_minutes)
        current = start_at
        records = []
        for _ in range(records_count):
            records.append(_okte_record(current, tz_offset_hours=tz_offset_hours))
            current += step

        file_suffix = "stream" if output_mode == "realtime_stream" else "batch"
        file_name = f"okte_dataset_{file_suffix}.{output_format}"
        file_path = str(Path(self.results_path) / file_name)

        if output_format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(records, indent=4))
        else:
            with open(file_path, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, delimiter=";")
                writer.writeheader()
                writer.writerows(records)

        self.logger.info("OKTE dataset saved to %s (%d records).", file_path, records_count)
        self.display_result = {"file_type": "txt", "file_path": file_path}

        return OutputModel(
            file_path=file_path,
            target_column=TARGET_COLUMN,
        )
