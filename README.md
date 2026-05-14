# Domino Piece Repository

## Smart Grid Pieces

Pieces for smart grid PVOUT prediction, PVOUT error correction, and electricity price prediction.

### Included pieces

- **DataNormalizationPiece**
- **DataPreprocessingPiece**
- **ModelDeciderPiece**
- **InferencePiece**
- **PVOUTPredictionModelTrainPiece**
- **PVOUTErrorCorrectionModelTrainPiece**
- **ElectricityPricePredictionModelTrainPiece**
- **EvaluateMLModelPiece**
- **ExplainablePredictionPiece**

### Debug outputs

Every piece in the prediction flow writes a debug artifact to its own `results_path` directory, which Domino mounts to the host. Look for these files under
`domino/airflow/domino_data/<run>/<PieceName>_<id>/results/`:

| Piece | File on disk | Disable / override |
|---|---|---|
| SyntheticDataGeneratorPiece | `dataset_batch.{csv,json}` or `dataset_stream.{csv,json}` | controlled by `output_mode` + `output_format` |
| DataPreprocessingPiece | `preprocessed.csv` (or `preprocessed_pred.csv` / `preprocessed_true.csv` in correction mode) | set `payload.save_data_path` to override; passing an empty string also overrides |
| DataNormalizationPiece | `normalized.csv` | written whenever a real pandas DataFrame is produced; no override flag yet |
| ModelDeciderPiece | `decision.json` (selected `model_type`, `normalization_type`, schema hints) | always written |
| PVOUTPredictionModelTrainPiece | `pvout_prediction_<model_type>.pkl` | override directory with `payload.checkpoint_dir` |
| PVOUTErrorCorrectionModelTrainPiece | `pvout_error_correction_<model_type>.pkl` | override directory with `payload.checkpoint_dir` |
| ElectricityPricePredictionModelTrainPiece | `electricity_price_xgb.joblib` + sidecar JSON | override directory with `payload.output_dir`; toggle enriched CSV with `payload.save_enriched_csv` |
| InferencePiece | `forecast.csv` | override path with `payload.forecast_output_csv_path` |
| EvaluateMLModelPiece | `metrics.json` | always written |

Each piece also sets `display_result` so the generated file is opened in the Domino GUI run view.