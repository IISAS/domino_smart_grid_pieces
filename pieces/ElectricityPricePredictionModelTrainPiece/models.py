import json

from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    payload: str = Field(
        default="{}",
        description=(
            "Train an XGBoost regressor for electricity price (EUR/MWh) from tabular energy data.\n\n"
            "**Input data** (one row per delivery interval, e.g. 15 min):\n"
            "- `tabular_data` / `data_path` / `csv_path`: rows with energy-side features you already have "
            "(load, RES forecast, temperature, lags, calendar, etc.).\n"
            "- `model_setup.datetime_column` (default `datetime`): wall time of the interval; naive values "
            "are interpreted as Europe/Bratislava when using OKTE.\n"
            "- `model_setup.feature_columns` (required): numeric columns used as X.\n"
            "- `model_setup.target_column` (default `price_eur_mwh`): regression target y.\n"
            "- `model_setup.target_source`:\n"
            "  - `column` — target must already be present in each row (historical prices from any source).\n"
            "  - `okte` — fetch DAM prices from OKTE for each row's slot (optional `okte.endpoint`); "
            "writes `price_eur_mwh` and `price_source` before training.\n\n"
            "**Training**:\n"
            "- `xgb_params`: optional dict passed to `xgboost.XGBRegressor` (defaults are sensible for tabular).\n"
            "- `output_dir`, `model_filename` (default `electricity_price_xgb.joblib`).\n"
            "- `save_enriched_csv`: if true and `target_source=okte`, also write merged CSV for debugging.\n\n"
            "**Outputs**: `model_path` (joblib), `preprocessing_metadata.json` path, `train_metrics`, "
            "`train_rows`, `fallback_used` (OKTE 28d fallback when applicable)."
        ),
    )

    @field_validator("payload", mode="before")
    @classmethod
    def _coerce_payload(cls, value):
        if value is None:
            return "{}"
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def payload_as_dict(self) -> dict:
        try:
            parsed = json.loads(self.payload) if self.payload else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "Training outputs: `model_path`, `preprocessing_metadata_path`, `train_metrics`, "
            "`train_rows`, `fallback_used`; optional `enriched_csv_path`."
        ),
    )
