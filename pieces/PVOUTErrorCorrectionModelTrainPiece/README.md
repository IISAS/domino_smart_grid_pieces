# PVOUTErrorCorrectionModelTrainPiece

> **Prototype** - This piece is under active development and its interface may change.

A Domino piece for training PV output error-correction models. It supports advanced correction models based on XGBoost (including residual-meta and difficulty-weighted variants), plus lightweight linear/ridge options for minimal environments.

The piece consumes tabular data with baseline prediction and true PVOUT, trains the selected correction model, and writes a checkpoint for downstream inference.

## Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `payload.model_type` | `str` | `linear_regression` | Supported: `error_correction_xgb_regressor_model`, `error_correction_residual_meta_xgb_regressor_model`, `error_correction_difficulty_weighted_xgb_regressor_model`, `linear_regression`, `ridge_regression` |
| `payload.model_params` | `dict` | `{}` | Constructor parameters for the selected model |
| `payload.model_setup.feature_columns` | `list[str]` | *(required)* | Input feature columns used for training |
| `payload.model_setup.target_column` | `str` | `PVOUT` | Ground-truth PV output column |
| `payload.model_setup.pred_column` | `str` | `None` | Required for error-correction XGB models (baseline prediction column) |
| `payload.data_path` / `payload.csv_path` | `str` | `None` | Path to CSV input |
| `payload.tabular_data` / `payload.dataframe` | `list[dict] \| dict[str, list]` | `None` | Inline input alternative |
| `payload.checkpoint_dir` | `str` | piece results path / temp dir | Folder for saved checkpoint |

For pandas-based models, numeric columns are coerced and invalid rows are dropped before training.

## Output

| Field | Description |
|-------|-------------|
| `message` | Status string (`PVOUTErrorCorrectionModelTrainPiece executed.`) |
| `artifacts.trained_model` | Metadata envelope for external models, or full lightweight model dict for linear/ridge |
| `artifacts.checkpoint_path` | Saved `.pkl` checkpoint path |
| `artifacts.train_metrics` | Metrics for lightweight models (`rmse`, `mae`); may be empty for external model classes |

## Typical Workflow

```
PVOUT prediction baseline
  -> DataPreprocessingPiece (features + baseline column)
  -> PVOUTErrorCorrectionModelTrainPiece
       -> correction checkpoint (.pkl)
  -> InferencePiece (mode: pvout_correction)
```

## Running Tests

```bash
# From repository root
pytest pieces/PVOUTErrorCorrectionModelTrainPiece/test_pvout_error_correction_model_train_piece.py -v
```

Notes:
- Advanced model tests are dependency-aware and may be skipped when `pandas`, `xgboost`, or `scikit-learn` are unavailable.
