# ElectricityPricePredictionModelTrainPiece

> **Prototype** - This piece is under active development and its interface may change.

A Domino piece for training an electricity price regression model (`xgboost.XGBRegressor`) from tabular features. It supports two target strategies:
- direct target from a column in your input data
- target enrichment from OKTE DAM API (`target_source=okte`)

It outputs a trained model checkpoint and preprocessing metadata for inference.

## Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `payload.tabular_data` / `payload.data_path` / `payload.csv_path` | `list[dict] \| str` | *(required)* | Input rows (typically one row per delivery interval, e.g. 15 min) |
| `payload.model_setup.feature_columns` | `list[str]` | *(required)* | Numeric feature columns for X |
| `payload.model_setup.target_column` | `str` | `price_eur_mwh` | Regression target column name |
| `payload.model_setup.datetime_column` | `str` | `datetime` | Datetime column used for parsing and OKTE alignment |
| `payload.model_setup.target_source` | `"column" \| "okte"` | `column` | Use existing target column or fetch target prices from OKTE |
| `payload.okte.endpoint` | `str` | OKTE API default | Optional custom endpoint for DAM data fetch |
| `payload.xgb_params` | `dict` | internal defaults | Extra params for `XGBRegressor` |
| `payload.output_dir` | `str` | piece results path / temp dir | Output directory |
| `payload.model_filename` | `str` | `electricity_price_xgb.joblib` | Output model filename |
| `payload.save_enriched_csv` | `bool` | `False` | Save enriched training CSV when `target_source=okte` |

## Output

| Field | Description |
|-------|-------------|
| `message` | Status string (`Electricity price XGBoost regressor trained and saved.`) |
| `artifacts.model_path` | Path to saved model (`.joblib` / `.pkl`) |
| `artifacts.preprocessing_metadata_path` | Path to `preprocessing_metadata.json` |
| `artifacts.train_metrics` | Training metrics (`rmse`, `mae`) |
| `artifacts.train_rows` | Number of rows used for fitting |
| `artifacts.fallback_used` | Whether OKTE fallback averaging was used |
| `artifacts.enriched_csv_path` | Optional path when enriched CSV output is enabled |

## Typical Workflow

```
Data ingestion / feature prep
  -> ElectricityPricePredictionModelTrainPiece
       -> electricity_price_xgb.joblib
       -> preprocessing_metadata.json
  -> InferencePiece (mode: price_level / price_ahead)
```

## Running Tests

```bash
# From repository root
pytest pieces/ElectricityPricePredictionModelTrainPiece/test_electricity_price_prediction_model_train_piece.py -v
```
