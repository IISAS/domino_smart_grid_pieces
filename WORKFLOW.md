# Prediction Workflow — Domino DAG Configuration

End-to-end forecasting pipeline with **two parallel targets** — PVOUT (solar power from Solargis-style data) and electricity spot price (from OKTE-style market data). Both data generators feed a **single DataPreprocessing node** that inner-joins them on `datetime` into one merged dataset. The fork happens **after** DataPreprocessing: each branch has its own `ModelDecider → DataNormalization → Trainer` triple (because normalization choice depends on the model chosen per target). Both trainers V-merge into one Inference node, which fans out to Evaluate / Explainable / ForecastAggregator as siblings. PVOUTErrorCorrectionModelTrain is an **optional inline stage** on the PVOUT branch only.

```
SyntheticDataGenerator ─┐
                         ▼
                   DataPreprocessing  (inner-join Solargis + OKTE on datetime → one merged CSV)
                         ▲
OKTEDataGenerator ───────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     ModelDecider_PVOUT        ModelDecider_Price
            │                         │
            ▼                         ▼
     DataNormalization_PVOUT   DataNormalization_Price
            │                         │
            ▼                         ▼
     PVOUTPredictionModelTrain  ElectricityPricePredictionModelTrain
            │                         │
            ▼                         │
   [ PVOUTErrorCorrectionModelTrain ] │   (optional inline)
            │                         │
            └────────────┬────────────┘
                         ▼
                  InferencePiece  (models = [pvout_entry, price_entry, …])
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         Evaluate   Explainable   ForecastAggregator
        (evaluations[]) (explanations[]) (forecasts[])
```

> The DAG forks at DataPreprocessing into two sub-chains (`ModelDecider → DataNormalization → Trainer`) — one per target. The two `DataNormalization_*` nodes consume the **same merged CSV** but apply *per-branch* normalization (`normalization_type` decided by their own ModelDecider). They duplicate because `normalization_type` depends on the model picked per target.

![Prediction workflow](WORKFLOW.webp)

## What changed in this revision

- **`DataPreprocessingPiece`** now accepts two source paths — `data_path_solargis` and `data_path_okte` — and **inner-joins them on `datetime`** to emit one merged dataset with weather columns + market columns side-by-side. `data_path` is kept as a back-compat alias for `data_path_solargis`. With both wired up, the PVOUT model can use OKTE features as predictors and vice-versa.
- **`ModelDeciderPiece`** now **strips its `target_column` from the echoed `feature_columns`**, so each per-target Decider node emits a feature list that excludes its own target (no leakage). When two Deciders fan out from the same DataNorm-fed branch, each gets a different target — `ModelDecider_PVOUT` strips `PVOUT`, `ModelDecider_Price` strips `spot_price_eur_mwh`.
- The DAG fork now happens **after DataPreprocessing** (single shared merged CSV up to that point), splitting into two `ModelDecider → DataNorm → Trainer` sub-chains before V-merging at Inference. Two `DataNorm` nodes instead of one — necessary because the normalization choice is per-model, not per-dataset.

## DAG edges to draw

### Shared prep (always)

| From | To | Reason |
|---|---|---|
| SyntheticDataGenerator | DataPreprocessing | `data_path_solargis` ← File Path |
| OKTEDataGenerator | DataPreprocessing | `data_path_okte` ← File Path |

### PVOUT sub-chain (always)

| From | To | Reason |
|---|---|---|
| DataPreprocessing | ModelDecider_PVOUT | data_path, feature_columns, target_column = PVOUT |
| ModelDecider_PVOUT | DataNormalization_PVOUT | normalization_type + echoes (target stripped from feature_columns) |
| DataNormalization_PVOUT | PVOUTPredictionModelTrain | data_path + echoes |
| PVOUTPredictionModelTrain | InferencePiece (pvout entry) | model_path + echoes |

### Electricity-price sub-chain (always, when targeting price)

| From | To | Reason |
|---|---|---|
| DataPreprocessing | ModelDecider_Price | data_path, feature_columns, target_column = spot_price_eur_mwh |
| ModelDecider_Price | DataNormalization_Price | normalization_type + echoes (target stripped from feature_columns) |
| DataNormalization_Price | ElectricityPricePredictionModelTrain | data_path + echoes |
| ElectricityPricePredictionModelTrain | InferencePiece (price entry) | model_path + echoes |

### Optional PVOUT error-correction inline stage

| From | To | Reason |
|---|---|---|
| PVOUTPredictionModelTrain | PVOUTErrorCorrectionModelTrain | baseline_model_path + echoes |
| PVOUTErrorCorrectionModelTrain | InferencePiece (pvout entry) | replaces the direct trainer→inference edge above |

### Inference fan-out (always)

| From | To | Reason |
|---|---|---|
| InferencePiece | EvaluateMLModel | `forecasts` list → per-model `pred_df_path` |
| InferencePiece | ExplainablePrediction | `forecasts` list → per-model `model_path` / `data_path` |
| InferencePiece | ForecastAggregator | `forecasts` list (datetime + per-model predictions) |
| DataPreprocessing | ForecastAggregator | `actual_csv_path` (optional, to add the actual target column) |

**Edges:** 12 for the canonical dual-target chain, 13 with error-correction inline, +1 if you wire the aggregator's actual column. The fan-out at DataPrep (to 2 ModelDeciders), the fan-in at Inference (from 2 trainers), and the fan-out from Inference (to 3 consumers) are the only branches — everything else stays linear.

---

## SyntheticDataGeneratorPiece *(PVOUT branch source)*

| Field | Value | Upstream |
|---|---|---|
| Dataset Type | `solargis` | — |
| Output Mode | `batch_sample` | — |
| Output Format | `csv` | — |
| Records Count | `100` (any >0) | — |
| Time Step Minutes | `15` | — |
| Seed | (optional) | — |

Produces `dataset_batch.csv` in its `results/`.

## OKTEDataGeneratorPiece *(price branch source)*

| Field | Value | Upstream |
|---|---|---|
| Records Count | `100` (any >0) | — |
| Time Step Minutes | `15` | — |
| Output Mode | `batch_sample` | — |
| Output Format | `csv` | — |
| Start At | (optional ISO timestamp) | — |
| Timezone Offset Hours | `1.0` (CET) | — |
| Seed | (optional) | — |

Produces a CSV with `Date, Time, market_area, imbalance_mw, spot_price_eur_mwh, scheduled_generation_mw, actual_generation_mw` and emits `file_path` + `target_column = spot_price_eur_mwh`.

**Important:** keep `Records Count` and `Time Step Minutes` aligned with the Solargis generator. The two are inner-joined on `datetime` in DataPreprocessing — mismatched grids yield few/no overlapping rows.

## DataPreprocessingPiece *(single shared node)*

One DataPreprocessing node consumes both generators. With both `data_path_solargis` and `data_path_okte` wired, it performs an inner-join on `datetime` to emit one merged CSV containing Solargis weather columns AND OKTE market columns side-by-side.

| Field | Value | Upstream |
|---|---|---|
| Preprocessing Option | `prediction` | — |
| Data Path Solargis | ← **SyntheticDataGenerator.File Path** | ✓ |
| Data Path Okte | ← **OKTEDataGenerator.File Path** | ✓ |
| Data Path | (leave empty — back-compat alias for `data_path_solargis`) | — |
| Target Column | (leave empty — each downstream ModelDecider sets its own) | — |
| Save Data Path | (empty — defaults to `<results_path>/preprocessed.csv`) | — |
| Keep Datetime | unchecked / `false` | — |

**Important:** keep `Keep Datetime` off. If it's on, `datetime` ends up in `feature_columns` and the trainer's numeric coercion will null-out every row.

Produces `preprocessed.csv` (one row per matched timestamp, all columns from both sources) and emits `Data Path`, `Feature Columns` (all numeric columns from both sources, no targets stripped yet), and `Target Column` (empty when not specified — each downstream Decider sets its own).

## ModelDeciderPiece *(deploy once per target — `_PVOUT` and `_Price`)*

Two instances on the canvas, both fed from `DataPreprocessing`. Each picks its target and (optionally) its model type, and **strips its own target from the echoed `feature_columns`** so the downstream trainer never sees the ground truth as a predictor.

### ModelDecider_PVOUT

| Field | Value | Upstream |
|---|---|---|
| Problem Type | `pvout_prediction` | — |
| Horizon | `1` | — |
| Available Models | `+` → `xgb_regressor_model` | — |
| Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Target Column | `PVOUT` | — |
| Data Path | ← **DataPreprocessing.Data Path** | ✓ |

### ModelDecider_Price

| Field | Value | Upstream |
|---|---|---|
| Problem Type | `price_prediction` | — |
| Horizon | `1` | — |
| Available Models | `+` → `xgb_regressor_model` (or another model better suited to price) | — |
| Feature Columns | ← **DataPreprocessing.Feature Columns** | ✓ |
| Target Column | `spot_price_eur_mwh` | — |
| Data Path | ← **DataPreprocessing.Data Path** | ✓ |

Each Decider echoes `data_path`, `model_type`, `normalization_type`, and a **target-stripped** `feature_columns` to its own DataNorm downstream.

## DataNormalizationPiece *(deploy once per branch — `_PVOUT` and `_Price`)*

Two instances on the canvas, each downstream of its branch's ModelDecider. Both consume the same merged CSV from DataPreprocessing but apply **different normalization** based on the model chosen by their ModelDecider.

| Field | Value | Upstream (PVOUT branch) | Upstream (price branch) |
|---|---|---|---|
| Normalization Type | (from upstream) | ← **ModelDecider_PVOUT.Normalization Type** ✓ | ← **ModelDecider_Price.Normalization Type** ✓ |
| Data Path | (from upstream) | ← **ModelDecider_PVOUT.Data Path** ✓ | ← **ModelDecider_Price.Data Path** ✓ |
| Feature Columns | (target-stripped, from upstream) | ← **ModelDecider_PVOUT.Feature Columns** ✓ | ← **ModelDecider_Price.Feature Columns** ✓ |
| Target Column | (from upstream) | ← **ModelDecider_PVOUT.Target Column** ✓ | ← **ModelDecider_Price.Target Column** ✓ |
| Model Type | (from upstream) | ← **ModelDecider_PVOUT.Model Type** ✓ | ← **ModelDecider_Price.Model Type** ✓ |
| Features | (empty) | — | — |
| Dataframe | (empty) | — | — |

Each instance writes its own `normalized.csv` under `results/` (passthrough for XGBoost). Re-emits all echoes plus its own `data_path` for the trainer.

## PVOUTPredictionModelTrainPiece *(PVOUT branch)*

| Field | Value | Upstream |
|---|---|---|
| Model Type | ← **DataNormalization_PVOUT.Model Type** | ✓ |
| Data Path | ← **DataNormalization_PVOUT.Data Path** | ✓ |
| Feature Columns | ← **DataNormalization_PVOUT.Feature Columns** | ✓ |
| Target Column | ← **DataNormalization_PVOUT.Target Column** | ✓ |
| Checkpoint Dir | (empty — defaults to `results_path`) | — |

Produces `pvout_prediction_<model_type>.pkl` and emits `Model Path`, echoed `Data Path`, `Feature Columns`, `Target Column`.

## PVOUTErrorCorrectionModelTrainPiece *(optional inline on PVOUT branch)*

Drop this between the baseline PVOUT trainer and Inference when you want a residual-correction layer. With `Baseline Model Path` wired up, the piece auto-generates `PVOUT_PRED` from the baseline checkpoint during training.

| Field | Value | Upstream |
|---|---|---|
| Model Type | `error_correction_xgb_regressor_model` | — |
| Data Path | ← **PVOUTPredictionModelTrain.Data Path** | ✓ |
| Baseline Model Path | ← **PVOUTPredictionModelTrain.Model Path** | ✓ |
| Feature Columns | ← **PVOUTPredictionModelTrain.Feature Columns** | ✓ |
| Target Column | ← **PVOUTPredictionModelTrain.Target Column** | ✓ |
| Model Setup → Pred Column | (empty — defaults to `PVOUT_PRED`) | — |
| Checkpoint Dir | (empty) | — |

Emits its own `Model Path` (correction model) plus echoed fields and `Baseline Model Path` (for staged inference). When inserted, **the PVOUT entry on Inference points at the correction model's `Model Path`, not the baseline's.**

## ElectricityPricePredictionModelTrainPiece *(price branch)*

| Field | Value | Upstream |
|---|---|---|
| Model Type | (empty — uses internal XGBRegressor) | — |
| Data Path | ← **DataNormalization_Price.Data Path** | ✓ |
| Feature Columns | ← **DataNormalization_Price.Feature Columns** | ✓ |
| Target Column | ← **DataNormalization_Price.Target Column** (`spot_price_eur_mwh`) | ✓ |
| Output Dir | (empty — defaults to `results_path`) | — |
| Model Filename | (empty — `electricity_price_xgb.pkl` by default) | — |

Produces `electricity_price_xgb.pkl` + `preprocessing_metadata.json` next to it, and emits `Model Path`, `Feature Columns`, `Target Column`, `Preprocessing Metadata Path`.

## InferencePiece *(V-merge — both trainers feed this single node)*

InferencePiece runs each entry in its `models` list independently and produces one forecast CSV per entry under `forecast_<model_id>.csv`. Wire the `models` list with one entry per trainer.

| Field | Value | Upstream |
|---|---|---|
| Models | List with one entry per upstream trainer — see below | ✓ |
| Datetime Column | `datetime` | — |
| Horizon Column | (empty unless `flag_each_day=true` on a preprocessor) | — |
| Max Horizon | (empty) | — |

### `Models` list entries

```json
[
  {
    "model_id": "pvout",
    "mode": "pvout_correction",
    "model_path": "<PVOUTPredictionModelTrain.Model Path  or  PVOUTErrorCorrectionModelTrain.Model Path>",
    "data_path": "<PVOUTPredictionModelTrain.Data Path>",
    "feature_columns": "<PVOUTPredictionModelTrain.Feature Columns>",
    "target_column": "PVOUT",
    "base_forecast_column": "PVOUT"
  },
  {
    "model_id": "price",
    "mode": "price_level",
    "model_path": "<ElectricityPricePredictionModelTrain.Model Path>",
    "data_path": "<DataNormalization_Price.Data Path>",
    "feature_columns": "<ElectricityPricePredictionModelTrain.Feature Columns>",
    "target_column": "spot_price_eur_mwh",
    "preprocessing_metadata_path": "<ElectricityPricePredictionModelTrain.Preprocessing Metadata Path>"
  }
]
```

Produces one `forecast_<model_id>.csv` per entry (e.g. `forecast_pvout.csv`, `forecast_price.csv`) and emits `forecasts: list[ForecastEntry]` containing `{model_id, forecast_csv_path, mode, model_path, feature_columns, target_column}` for each.

**Mode reference:**
- `pvout_correction` — `final_forecast = base_forecast_column + model.predict(X)`. Use with PVOUT (or its correction model).
- `price_level` — `final_forecast = model.predict(X)`, no baseline. Use with the electricity price model.
- `price_ahead` — `final_forecast = baseline_column + correction(X)`. Use when you have a baseline day-ahead price column and a price-correction model.

## EvaluateMLModelPiece *(sibling of Explainable + Aggregator)*

| Field | Value | Upstream |
|---|---|---|
| Evaluations | List with one entry per model — see below | ✓ |
| Baseline Id | `1` | — |
| Plot | unchecked | — |

### `Evaluations` list entries

```json
[
  {
    "model_id": "pvout",
    "evaluation_option": "normal",
    "forecast_column": "correction",
    "target_column": "PVOUT",
    "pred_df_path": "<forecasts[0].forecast_csv_path  (pvout)>"
  },
  {
    "model_id": "price",
    "evaluation_option": "normal",
    "forecast_column": "final_forecast",
    "target_column": "spot_price_eur_mwh",
    "pred_df_path": "<forecasts[1].forecast_csv_path  (price)>"
  }
]
```

Produces `metrics_<model_id>.json` per entry plus an aggregated `artifacts.per_model = {model_id: metrics}`.

## ExplainablePredictionPiece *(sibling of Evaluate + Aggregator)*

| Field | Value | Upstream |
|---|---|---|
| Explanations | List with one entry per model — see below | ✓ |
| Explain | `true` | — |
| Explain Method | `shap` (default when `Explain=true`) | — |

### `Explanations` list entries

```json
[
  {
    "model_id": "pvout",
    "explain": true,
    "explain_method": "shap",
    "model_path": "<PVOUTPredictionModelTrain.Model Path>",
    "data_path": "<PVOUTPredictionModelTrain.Data Path>",
    "feature_columns": "<PVOUTPredictionModelTrain.Feature Columns>"
  },
  {
    "model_id": "price",
    "explain": true,
    "explain_method": "shap",
    "model_path": "<ElectricityPricePredictionModelTrain.Model Path>",
    "data_path": "<DataNormalization_Price.Data Path>",
    "feature_columns": "<ElectricityPricePredictionModelTrain.Feature Columns>"
  }
]
```

Produces `artifacts.per_model[model_id].explainability` with SHAP `shap_values`, `feature_names`, `base_value` for each model.

## ForecastAggregatorPiece *(sibling of Evaluate + Explainable)*

Joins all per-model forecast CSVs into one comparison CSV. The `forecasts` field on its input is the same list shape that `InferencePiece` emits, so wiring is a single edge.

| Field | Value | Upstream |
|---|---|---|
| Forecasts | ← **Inference.Forecasts** | ✓ |
| Actual CSV Path | ← **DataPreprocessing.Data Path** | ✓ (optional, supplies the `actual_<target>` column) |
| Target Column | (empty — falls back to first forecast entry's `target_column`) | — |
| Datetime Column | `datetime` | — |
| Horizon Column | `pred_sequence_id` | — |
| Forecast Column | `final_forecast` | — |
| Include Diff | unchecked (toggle if you want `diff_<model_id>` columns) | — |
| Output CSV Name | `aggregated_forecast.csv` (default) | — |

Produces `aggregated_forecast.csv` with columns `datetime, pred_sequence_id, pred_<model_id>, …` (and optional `actual_<target>` + `diff_<model_id>`).

---

## Quick sanity checklist before each run

- Pull fresh images for every group after CI publishes (one for each `requirements_*.txt`):
  ```
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group0   # numpy+pandas (Synthetic, OKTE, DataPrep, Decider, Norm, Evaluate, Aggregator)
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group1   # tabpfn-heavy (PVOUTPredictionModelTrain)
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group2   # xgboost+sklearn+joblib (Inference, ErrorCorrection, ElectricityTrain)
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group3   # PyTorch GPU (TestGpuSupport, optional)
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group4   # shap (ExplainablePrediction)
  ```
- `git pull` first so `.domino/dependencies_map.json` matches the latest auto-organize commit; refresh the piece repository in Domino UI afterwards.
- **DataPreprocessing** has both `Data Path Solargis` and `Data Path Okte` wired (otherwise the merge falls back to a single-source dataset and the other branch's features are missing).
- **Time grids** of the two generators align: same `Time Step Minutes` and `Records Count`, similar `Start At`. The inner-join drops rows with no match.
- **ModelDecider_PVOUT** has `Target Column = PVOUT`; **ModelDecider_Price** has `Target Column = spot_price_eur_mwh`. The target-stripping logic relies on this.
- **Inference Models list** has at least one entry per active trainer, each with a non-empty `mode` and `model_path`.
- **If using PVOUT error-correction:** the PVOUT entry's `model_path` points at the *correction* trainer's checkpoint, not the baseline's.
- **DataPreprocessing's Keep Datetime is off.**

## Files dropped into each piece's `results/`

| Piece | File | Purpose |
|---|---|---|
| SyntheticDataGenerator | `dataset_batch.csv` | input dataset (PVOUT side) |
| OKTEDataGenerator | OKTE CSV | input dataset (price side) |
| DataPreprocessing | `preprocessed.csv` | merged numeric-only features + targets |
| ModelDecider (× 2) | `decision.json` | chosen `model_type`, `normalization_type`, target-stripped feature list |
| DataNormalization (× 2) | `normalized.csv` | passthrough or scaled (per-branch) |
| PVOUTPredictionModelTrain | `pvout_prediction_<model_type>.pkl` | baseline PVOUT checkpoint |
| PVOUTErrorCorrectionModelTrain *(optional)* | `pvout_error_correction_<model_type>.pkl` | residual-correction checkpoint |
| ElectricityPricePredictionModelTrain | `electricity_price_xgb.pkl` + `preprocessing_metadata.json` | price model checkpoint + feature metadata |
| InferencePiece | `forecast_<model_id>.csv` (one per entry) | datetime + base_forecast + correction + final_forecast (+ target if present) |
| EvaluateMLModel | `metrics_<model_id>.json` (one per entry) | MAE / RMSE / MAPE / n per model |
| ExplainablePrediction | (artifacts only by default) | SHAP per-model `shap_values` + `feature_names` |
| ForecastAggregator | `aggregated_forecast.csv` | datetime + `pred_<model_id>` columns (+ optional `actual_<target>` + `diff_<model_id>`) |

All these land on the host through Domino's `results_path` mount — no Docker exec needed to inspect them.
