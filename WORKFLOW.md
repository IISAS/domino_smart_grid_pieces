# Prediction Workflow — Domino DAG Configuration

End-to-end PVOUT prediction pipeline. The two grey nodes are **optional add-ons**: drop them into the canvas only when you need residual correction on top of the baseline forecast or feature-attribution artifacts for the trained model.

```
SyntheticDataGenerator
    │
    ▼
DataPreprocessing
    ├──────────────────┐
    ▼                  │
ModelDecider           │
    ├──────────┐       │
    ▼          │       │
DataNormalization      │
    │   │              │
    │   └─────┐        │
    ▼        ▼         ▼
PVOUTPredictionModelTrain ◄────┘
    │       │       │
    │       │       └──────────────────────┐
    │       │                              │
    │       └────────┐                     │
    │                │                     │
    │                │           PVOUTErrorCorrectionModelTrain   (optional)
    │                │                     │       │
    │                ▼                     ▼       │
    │             Inference  (← Normalization, ← DataPreprocessing for features)
    │                │                             │
    │                │                             ▼
    │                │                  ExplainablePrediction      (optional)
    │                ▼
    │            EvaluateMLModel
    └──► (same data path used to backfill PVOUT_PRED in the correction trainer)
```

![Prediction workflow](WORKFLOW.webp)

**Why the fan-out:** every "consumer" piece needs an explicit edge from each "producer" piece it pulls from. Domino's Upstream dropdown only shows fields from direct parents, not transitive ancestors. See the per-piece tables below for which edges feed which field.

## DAG edges to draw

### Core flow (always required)

| From | To | Reason |
|---|---|---|
| Synthetic | DataPreprocessing | dataset file |
| DataPreprocessing | DataNormalization | data_path |
| DataPreprocessing | PVOUTPredictionModelTrain | feature_columns |
| DataPreprocessing | Inference | feature_columns |
| ModelDecider | DataNormalization | normalization_type |
| ModelDecider | PVOUTPredictionModelTrain | model_type, target_column |
| DataNormalization | PVOUTPredictionModelTrain | data_path |
| DataNormalization | Inference | data_path |
| PVOUTPredictionModelTrain | Inference | model_path |
| Inference | EvaluateMLModel | forecast_csv_path |

### Optional: error correction add-on

| From | To | Reason |
|---|---|---|
| PVOUTPredictionModelTrain | PVOUTErrorCorrectionModelTrain | baseline_model_path |
| DataPreprocessing | PVOUTErrorCorrectionModelTrain | feature_columns, target_column |
| DataNormalization | PVOUTErrorCorrectionModelTrain | data_path |
| PVOUTErrorCorrectionModelTrain | Inference | model_path (replaces baseline edge for staged inference; see Inference section) |

### Optional: explainability add-on

| From | To | Reason |
|---|---|---|
| PVOUTPredictionModelTrain **or** PVOUTErrorCorrectionModelTrain | ExplainablePrediction | model_path |
| DataPreprocessing | ExplainablePrediction | feature_columns |
| DataNormalization | ExplainablePrediction | data_path |

---

## SyntheticDataGeneratorPiece

| Field | Value | Upstream |
|---|---|---|
| Dataset Type | `solargis` | — |
| Output Mode | `batch_sample` | — |
| Output Format | `csv` | — |
| Records Count | `100` (any >0) | — |
| Time Step Minutes | `15` | — |
| Seed | (optional) | — |

Produces `dataset_batch.csv` in its `results/`.

## DataPreprocessingPiece

| Field | Value | Upstream |
|---|---|---|
| Preprocessing Option | `prediction` | — |
| Data Path | ← **SyntheticDataGenerator.File Path** | ✓ |
| Save Data Path | (leave empty — defaults to `<results_path>/preprocessed.csv`) | — |
| Test Size | (leave empty) | — |
| Keep Datetime | unchecked / `false` | — |

**Important:** keep `keep_datetime` off. If it's on, `datetime` ends up in feature_columns and the trainer's numeric coercion will null-out every row.

Produces `preprocessed.csv` and emits `Data Path`, `Feature Columns` (numeric only), `Target Column = PVOUT` as typed outputs.

## ModelDeciderPiece

| Field | Value | Upstream |
|---|---|---|
| Problem Type | `pvout_prediction` | — |
| Horizon | `1` | — |
| Available Models | `+` → `xgb_regressor_model` | — |
| Feature Columns | (leave empty) | — |
| Target Column | `PVOUT` | — |

Decider auto-picks `xgb_regressor_model` from Available Models and infers `normalization_type=none` because XGBoost doesn't need scaling. Produces `decision.json`.

## DataNormalizationPiece

| Field | Value | Upstream |
|---|---|---|
| Normalization Type | ← **ModelDecider.Normalization Type** | ✓ |
| Features | (leave empty) | — |
| Data Path | ← **DataPreprocessing.Data Path** | ✓ |
| Dataframe | (leave empty) | — |

For XGBoost the decider sets normalization to `none` → the piece does a passthrough but still writes `normalized.csv` under its `results/` for traceability.

## PVOUTPredictionModelTrainPiece

| Field | Value | Upstream |
|---|---|---|
| Model Type | ← **ModelDecider.Model Type** | ✓ |
| Data Path | ← **DataNormalization.Data Path** | ✓ |
| Csv Path | (leave empty — alias for Data Path) | — |
| Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Target Column | ← **ModelDecider.Target Column** | ✓ |
| Checkpoint Dir | (leave empty — defaults to `results_path`) | — |

Produces `pvout_prediction_xgb_regressor_model.pkl` and emits `Model Path` (string) as typed output for Inference (and, optionally, for the error-correction trainer / explainability piece).

## PVOUTErrorCorrectionModelTrainPiece *(optional)*

Add this node when you want a second-stage XGBoost that learns the residual of the baseline forecast. With `baseline_model_path` wired up, the piece reads the upstream checkpoint, runs it on the training data to synthesize `PVOUT_PRED`, and trains on the correction target `PVOUT − PVOUT_PRED`. No extra inference step is needed between the two trainers.

| Field | Value | Upstream |
|---|---|---|
| Model Type | `error_correction_xgb_regressor_model` | — |
| Model Params | `{"n_estimators": 100, "max_depth": 4, "verbosity": 0}` (tune as needed) | — |
| Model Setup → Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Model Setup → Target Column | ← **ModelDecider.Target Column** (or set to `PVOUT`) | ✓ |
| Model Setup → Pred Column | (leave empty — defaults to `PVOUT_PRED` and is auto-filled from the baseline checkpoint) | — |
| Baseline Model Path | ← **PVOUTPredictionModelTrain.Model Path** | ✓ |
| Data Path | ← **DataNormalization.Data Path** | ✓ |
| Checkpoint Dir | (leave empty — defaults to `results_path`) | — |

Produces `pvout_error_correction_error_correction_xgb_regressor_model.pkl` (same envelope shape as the baseline trainer — `{"metadata", "trained_model_object"}`) and emits `Model Path`, `Feature Columns`, `Target Column` as typed outputs.

**Other supported Model Types** (require the same numeric feature columns; only `error_correction_xgb_regressor_model` fits the strictly XGB-only minimal stack):

- `error_correction_residual_meta_xgb_regressor_model` — two-stage XGBoost where a second model learns the residual of the first.
- `error_correction_difficulty_weighted_xgb_regressor_model` — difficulty-weighted variant.
- `linear_regression` / `ridge_regression` — closed-form fallbacks; do **not** require `Baseline Model Path` (they train a plain regression on `PVOUT`, not on a residual).

## InferencePiece

The Inference node has two viable shapes depending on whether the optional error-correction trainer is in the DAG.

### Shape A — baseline only (no error-correction node)

| Field | Value | Upstream |
|---|---|---|
| Mode | `pvout_correction` | — |
| Model Path | ← **PVOUTPredictionModelTrain.Model Path** | ✓ |
| Data Path | ← **DataNormalization.Data Path** | ✓ |
| Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Datetime Column | `datetime` | — |
| Base Forecast Column | `PVOUT` | — |
| Horizon Column | (leave empty — only set when `flag_each_day=true` on preprocessor) | — |
| Max Horizon | (leave empty) | — |

Produces `forecast.csv` (datetime + base_forecast + correction + final_forecast + PVOUT) and emits `Forecast Csv Path` for the evaluator.

**Semantic note about Mode:**
- `pvout_correction` computes `final_forecast = PVOUT + model.predict(X)`. With `base_forecast_column=PVOUT`, this means `final_forecast = truth + prediction`. It runs, but `final_forecast − PVOUT` is just the model prediction, not an error.
- `price_level` computes `final_forecast = model.predict(X)` directly with no baseline. For pure prediction semantics, switch Mode to this and leave Base Forecast Column blank.

### Shape B — staged baseline + correction (with the error-correction node)

When the correction trainer is in the DAG, the cleanest way to get *true* `final_forecast = baseline_pred + correction(X)` is to configure the Inference node in **staged** mode. Stage 1 runs the baseline (writes its prediction into a column), stage 2 runs the correction model with that column as the baseline.

| Field | Value | Upstream |
|---|---|---|
| Mode | (leave empty — `stages` overrides it) | — |
| Data Path | ← **DataNormalization.Data Path** | ✓ |
| Stages | see JSON below | — |

```json
[
  {
    "mode": "price_level",
    "model_path": "<PVOUTPredictionModelTrain.Model Path>",
    "feature_columns": "<DataPreprocessing.Feature Columns>",
    "inject_forecast_as": "PVOUT_PRED"
  },
  {
    "mode": "pvout_correction",
    "model_path": "<PVOUTErrorCorrectionModelTrain.Model Path>",
    "feature_columns": "<DataPreprocessing.Feature Columns>",
    "base_forecast_column": "PVOUT_PRED"
  }
]
```

Produces a `forecast.csv` whose `final_forecast = PVOUT_PRED + correction(X)` — the actual error-corrected forecast.

## ExplainablePredictionPiece *(optional)*

Add this node to produce SHAP feature-attribution artifacts for the trained model. In the strictly XGBoost minimal stack, SHAP's `TreeExplainer` is selected automatically.

Point it at **either** the baseline trainer **or** the error-correction trainer (the artifacts are interpreted relative to whichever model is wired in).

| Field | Value | Upstream |
|---|---|---|
| Explain | `true` | — |
| Explain Method | `shap` (default when `explain=true`) | — |
| Model Path | ← **PVOUTPredictionModelTrain.Model Path** *or* **PVOUTErrorCorrectionModelTrain.Model Path** | ✓ |
| Data Path | ← **DataNormalization.Data Path** | ✓ |
| Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Target Column | ← **ModelDecider.Target Column** (informational) | ✓ |
| Use Diagnostic Loss | unchecked unless the upstream correction trainer was built with `use_diagnostic_loss=True` | — |

Produces `artifacts.explainability` (`shap_values`, `feature_names`, `base_value`, `explainer_type=TreeExplainer`). When the upstream is the diagnostic-weighted error-correction model and `Use Diagnostic Loss=true`, also produces `artifacts.diagnostic_heatmaps` with base64-encoded heatmaps over `(horizon × regime)` and `(horizon × hour)`.

**Note on `explainability` config:** advanced knobs (`background_size`, `max_evals`, `num_explanations`, `instance_idx`, `lime_kwargs`, `shap_kwargs`) are passed via the `Explainability` payload field as a nested dict. For the minimal flow, defaults are fine.

## EvaluateMLModelPiece

| Field | Value | Upstream |
|---|---|---|
| Evaluation Option | `normal` | — |
| Baseline Id | `1` | — |
| Plot | unchecked | — |
| Forecast Column | `correction` (for `pvout_correction` Mode without staging) **or** `final_forecast` (for `price_level` Mode, or staged Shape B above) | — |
| Target Column | `PVOUT` | — |
| Pred Df Path | ← **Inference.Forecast Csv Path** | ✓ |
| True Baseline Df Path | (leave empty — only for `errorcorrection` mode) | — |
| Pred Df / True Baseline Df / Y True | (leave empty) | — |

Produces `metrics.json` with `mae`, `rmse`, `mape`, `forecast_column`, `target_column`, `n`.

**Why `correction` not `final_forecast` for Shape A (`pvout_correction` Mode with `base_forecast_column=PVOUT`):** see the Inference semantic note above. `correction` = `model.predict(X)`, so `correction − PVOUT` measures actual prediction error. For Shape B (staged), `final_forecast` is already the meaningful corrected forecast, so use it directly.

---

## Quick sanity checklist before each run

- All four images pulled fresh after CI publishes:
  ```
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group0
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group1
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group2
  ```
- Domino piece repository refreshed to the latest `dev-3` release so the DAG points at current images and exposes the latest typed fields.
- All edges from the table above are drawn on the canvas.
- ModelDecider's Available Models contains `xgb_regressor_model`.
- DataPreprocessing's Keep Datetime is off.
- **If using the error-correction add-on:** Baseline Model Path on the correction trainer is wired to the baseline trainer's Model Path, and `feature_columns` is identical for both trainers (the baseline runs `.predict(df[feature_columns])` internally to backfill `PVOUT_PRED`).
- **If using staged inference (Shape B):** the `stages` JSON points stage 1 at the baseline checkpoint and stage 2 at the correction checkpoint, with `inject_forecast_as` matching `base_forecast_column` between stages.
- **If using the explainability add-on:** `Model Path` resolves to a `.pkl` produced by one of the trainers (the loader unwraps the `{"metadata", "trained_model_object"}` envelope automatically).

## Files dropped into each piece's `results/`

| Piece | File | Purpose |
|---|---|---|
| Synthetic | `dataset_batch.csv` | input dataset |
| DataPreprocessing | `preprocessed.csv` | numeric-only features + PVOUT |
| ModelDecider | `decision.json` | chosen `model_type`, `normalization_type` |
| DataNormalization | `normalized.csv` | passthrough (or scaled) version of preprocessed |
| PVOUTPredictionModelTrain | `pvout_prediction_<model_type>.pkl` | trained baseline model checkpoint |
| PVOUTErrorCorrectionModelTrain *(optional)* | `pvout_error_correction_<model_type>.pkl` | trained correction model checkpoint |
| Inference | `forecast.csv` | datetime + base_forecast + correction + final_forecast + PVOUT |
| ExplainablePrediction *(optional)* | (artifacts only, no on-disk file by default) | SHAP `shap_values` + `feature_names` (+ optional diagnostic heatmaps as base64 PNGs in artifacts) |
| EvaluateMLModel | `metrics.json` | MAE / RMSE / MAPE |

All these land on the host through Domino's `results_path` mount — no Docker exec needed to inspect them.
