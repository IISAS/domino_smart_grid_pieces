from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description=(
            "Inference inputs. For a no-op run, pass an empty payload, or omit both `mode` and `stages`.\n"
            "MVP keys when running inference:\n"
            "- `mode`: `pvout_correction`, `price_ahead` (baseline + correction), or `price_level` "
            "(direct regression output, e.g. EUR/MWh from electricity price XGB model).\n"
            "- `model_path`: path to `.pkl` (joblib) or xgboost booster file.\n"
            "- `input`: `{ \"tabular_data\": [...] }` or `{ \"data_path\": \"...\" }` (CSV/parquet).\n"
            "- `datetime_column` (default `datetime`), `feature_columns` (or from preprocessing metadata).\n"
            "- `base_forecast_column`: e.g. `PVOUT` or `price_baseline`.\n"
            "- `horizon_column` (default `pred_sequence_id`), optional `max_horizon`.\n"
            "- Optional: `preprocessing_metadata_path`, `strict_schema`, `missing_fill_value`, "
            "`per_horizon_outputs`, `forecast_output_csv_path`, `return_debug`.\n"
            "- Price mode: `build_baseline_if_missing` + `price_profile_path` (CSV with "
            "`dow`, `slot_15m`, `avg_price_eur_mwh`) when baseline column is missing.\n"
            "Upstream data fetching (SolarGis, weather, OKTE) is expected in separate pipeline steps; "
            "this piece consumes prepared tabular features."
        ),
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "No-op: `input_payload`. "
            "Run: `forecast` (inline_records, csv_path, columns), `per_horizon`, `metadata`, "
            "optional `debug`. Pipeline: additionally `stage_summaries`."
        ),
    )
