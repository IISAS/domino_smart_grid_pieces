import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


def _okte_record(ts: datetime) -> dict[str, Any]:
    return {
        "timestamp_utc": ts.isoformat(),
        "market_area": "SK",
        "imbalance_mw": round(random.uniform(-280, 260), 3),
        "spot_price_eur_mwh": round(random.uniform(25, 220), 2),
        "scheduled_generation_mw": round(random.uniform(1200, 4200), 2),
        "actual_generation_mw": round(random.uniform(1200, 4200), 2),
    }


class OKTESyntheticDataGeneratorPiece(BasePiece):

    def piece_function(self, input_data: InputModel):
        payload = input_data.to_payload_dict()
        self.logger.info("Running OKTESyntheticDataGeneratorPiece.")

        try:
            output_mode = str(payload.get("output_mode", "batch_sample")).strip().lower()
            if output_mode not in {"batch_sample", "realtime_stream"}:
                raise ValueError("output_mode must be `batch_sample` or `realtime_stream`.")

            output_format = str(payload.get("output_format", "json")).strip().lower()
            if output_format not in {"json", "csv"}:
                raise ValueError("output_format must be `json` or `csv`.")

            records_count = int(payload.get("records_count", 20))
            if records_count <= 0:
                raise ValueError("records_count must be > 0")

            time_step_minutes = int(payload.get("time_step_minutes", 15))
            if time_step_minutes <= 0:
                raise ValueError("time_step_minutes must be > 0")

            seed = payload.get("seed")
            if seed is not None:
                random.seed(int(seed))

            start_at_value = payload.get("start_at")
            if start_at_value:
                start_at = datetime.fromisoformat(str(start_at_value))
                if start_at.tzinfo is None:
                    start_at = start_at.replace(tzinfo=timezone.utc)
            else:
                start_at = datetime.now(tz=timezone.utc)

            step = timedelta(minutes=time_step_minutes)
            records = []
            current = start_at
            for _ in range(records_count):
                records.append(_okte_record(current))
                current += step

            file_suffix = "stream" if output_mode == "realtime_stream" else "batch"
            file_name = f"okte_dataset_{file_suffix}.{output_format}"
            file_path = str(Path(self.results_path) / file_name)

            if output_format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(records, indent=4))
            else:
                fieldnames = list(records[0].keys()) if records else []
                with open(file_path, "w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
                    writer.writeheader()
                    writer.writerows(records)

            self.logger.info("OKTE dataset saved to %s", file_path)
            self.display_result = {"file_type": "txt", "file_path": file_path}

            return OutputModel(file_path=file_path)

        except Exception:
            self.logger.exception(
                "OKTESyntheticDataGeneratorPiece failed. input_payload=%s", payload
            )
            raise
