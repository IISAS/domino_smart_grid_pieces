# PVOUTPredictionModelTrainPiece

> **Prototype** - This piece is under active development and its interface may change.

A Domino piece for training PV output prediction models from tabular data. The piece supports multiple model backends (linear, XGBoost, interval XGBoost, EDA baseline, TabPFN), persists a checkpoint, and returns model metadata for downstream inference.

## Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `payload.model_type` | `str` | `linear_regression_model` | Model selector: `linear_regression_model`, `xgb_regressor_model`, `interval_xgb_regressor_model`, `eda_rule_baseline`, `tabpfn_regressor_model` |
| `payload.model_params` | `dict` | `{}` | Constructor parameters forwarded to selected model |
| `payload.model_setup.feature_columns` / `payload.feature_columns` | `list[str]` | *(required)* | Feature column names used for training. Nested or top-level both accepted |
| `payload.model_setup.target_column` / `payload.target_column` | `str` | `PVOUT` | Target column used as y. Nested or top-level both accepted |
| `payload.data_path` / `payload.csv_path` | `str` | `None` | Path to CSV training data |
| `payload.tabular_data` / `payload.dataframe` | `list[dict] \| dict[str, list]` | `None` | Inline tabular input alternative |
| `payload.checkpoint_dir` | `str` | piece results path / temp dir | Folder where checkpoint is written |

Input rows are converted to a pandas DataFrame. Feature and target columns are coerced to numeric and invalid rows are dropped before fitting.

## Output

| Field | Description |
|-------|-------------|
| `message` | Status string (`PVOUTPredictionModelTrainPiece executed.`) |
| `artifacts.trained_model` | Metadata (`model_type`, `feature_columns`, `target_column`, `params`) |
| `artifacts.checkpoint_path` | Path to saved `.pkl` checkpoint |
| `artifacts.train_metrics` | Currently empty dict (reserved for metrics) |

Checkpoint payload includes model metadata and the trained model object serialized with `pickle`.

## Typical Workflow

Use this piece after preprocessing/feature engineering and before `InferencePiece`:

```
DataPreprocessingPiece (or custom data prep)
  -> PVOUTPredictionModelTrainPiece
       -> model checkpoint (.pkl)
       -> metadata artifacts
  -> InferencePiece (with model_path + feature_columns)
```

## Running Tests

```bash
# From repository root
pytest pieces/PVOUTPredictionModelTrainPiece/test_pvout_prediction_model_train_piece.py -v
```

Notes:
- `tabpfn_regressor_model` test may be skipped if gated model/auth is unavailable.
- XGBoost/TabPFN model tests are dependency-aware and skip if required packages are missing.
