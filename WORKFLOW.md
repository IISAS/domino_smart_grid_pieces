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
- **Every trainer now emits a typed `model_spec` bundle** (`PVOUTPredictionModelTrain`, `PVOUTErrorCorrectionModelTrain`, `ElectricityPricePredictionModelTrain`). Each `InferencePiece.models[i]` entry binds to this bundle in **one click** instead of toggling 5–6 fields by hand. `model_id`, `mode`, `model_path`, `data_path`, `feature_columns`, `target_column`, `base_forecast_column` (and `preprocessing_metadata_path` for the price trainer) all flow through that single binding.
- **`InferencePiece.OutputModel.forecasts[*]` now carries `data_path`** — the *input* dataset used per model. This unblocks the downstream auto-derive paths.
- **`EvaluateMLModelPiece`** now accepts `forecasts: list[ForecastEntry]` directly. Wire it once to `Inference.forecasts`; the piece auto-derives `evaluations` (one per forecast, `pred_df_path` = `forecast_csv_path`, `forecast_column` chosen by mode, `target_column` echoed). The existing `evaluations` field becomes an optional override-by-`model_id`.
- **`ExplainablePredictionPiece`** mirrors that: takes `forecasts: list[ForecastEntry]`, auto-derives `explanations` with `model_path` / `data_path` / `feature_columns` filled per entry and `explain=true` by default. `explanations` remains as the override path.
- **No more literal `<placeholder>` strings to paste.** Everywhere the trainer-to-Inference and Inference-to-consumer wiring used to require multi-field manual toggles, it's now a single Upstream bind per entry (or per piece). The doc below reflects the new UX.

## DAG edges to draw

### Shared prep (always)

| From | To | Reason |
|---|---|---|
| SyntheticDataGenerator | DataPreprocessing | `data_path_solargis` ← File Path |
| OKTEDataGenerator | DataPreprocessing | `data_path_okte` ← File Path |

### PVOUT sub-chain (always)

| From | To | Bind |
|---|---|---|
| DataPreprocessing | ModelDecider_PVOUT | `data_path`, `feature_columns`, `target_column = PVOUT` |
| ModelDecider_PVOUT | DataNormalization_PVOUT | `normalization_type` + echoes (target stripped) |
| DataNormalization_PVOUT | PVOUTPredictionModelTrain | `data_path` + echoes |
| PVOUTPredictionModelTrain | InferencePiece (one `models[i]` entry) | **one click** — `models[i] ← Model Spec` |

### Electricity-price sub-chain (always, when targeting price)

| From | To | Bind |
|---|---|---|
| DataPreprocessing | ModelDecider_Price | `data_path`, `feature_columns`, `target_column = spot_price_eur_mwh` |
| ModelDecider_Price | DataNormalization_Price | `normalization_type` + echoes (target stripped) |
| DataNormalization_Price | ElectricityPricePredictionModelTrain | `data_path` + echoes |
| ElectricityPricePredictionModelTrain | InferencePiece (one `models[i]` entry) | **one click** — `models[i] ← Model Spec` |

### Optional PVOUT error-correction inline stage

| From | To | Bind |
|---|---|---|
| PVOUTPredictionModelTrain | PVOUTErrorCorrectionModelTrain | `baseline_model_path` + echoes |
| PVOUTErrorCorrectionModelTrain | InferencePiece (one `models[i]` entry) | **one click** — `models[i] ← Model Spec` (replaces the baseline→inference bind) |

### Inference fan-out (always)

| From | To | Bind |
|---|---|---|
| InferencePiece | EvaluateMLModel | **one click** — `Forecasts ← Inference.Forecasts` |
| InferencePiece | ExplainablePrediction | **one click** — `Forecasts ← Inference.Forecasts` |
| InferencePiece | ForecastAggregator | **one click** — `Forecasts ← Inference.Forecasts` |
| DataPreprocessing | ForecastAggregator | `Actual CSV Path` (optional, supplies the `actual_<target>` column) |

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

Produces `pvout_prediction_<model_type>.pkl` and emits the typed `Model Spec` bundle (= `model_id=pvout`, `mode=pvout_correction`, `base_forecast_column=PVOUT`, plus the obvious `model_path` / `data_path` / `feature_columns` / `target_column`). Wire this directly into one `InferencePiece.models[i]` entry in one click. The individual `Model Path` / `Data Path` / `Feature Columns` / `Target Column` scalars are still emitted for back-compat.

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

Emits the typed `Model Spec` bundle (`model_id=pvout_correction`, `mode=pvout_correction`, `model_path` = correction checkpoint, `base_forecast_column=PVOUT`) — bind this from `InferencePiece.models[i]` in one click. Also echoes the scalar `Model Path`, `Data Path`, `Feature Columns`, `Target Column`, and `Baseline Model Path` (for staged inference) for back-compat. When this piece is inserted, point Inference's PVOUT entry at **this** trainer's `Model Spec`, not the baseline's.

## ElectricityPricePredictionModelTrainPiece *(price branch)*

| Field | Value | Upstream |
|---|---|---|
| Model Type | (empty — uses internal XGBRegressor) | — |
| Data Path | ← **DataNormalization_Price.Data Path** | ✓ |
| Feature Columns | ← **DataNormalization_Price.Feature Columns** | ✓ |
| Target Column | ← **DataNormalization_Price.Target Column** (`spot_price_eur_mwh`) | ✓ |
| Output Dir | (empty — defaults to `results_path`) | — |
| Model Filename | (empty — `electricity_price_xgb.pkl` by default) | — |

Produces `electricity_price_xgb.pkl` + `preprocessing_metadata.json` next to it. Emits the typed `Model Spec` bundle (`model_id=price`, `mode=price_level`, `target_column=spot_price_eur_mwh`, `base_forecast_column=None`, plus `model_path` / `data_path` / `feature_columns` / `preprocessing_metadata_path`) — bind this from `InferencePiece.models[i]` in one click. Scalar fields are still emitted for back-compat.

## InferencePiece *(V-merge — both trainers feed this single node)*

InferencePiece runs each entry in its `models` list independently and produces one forecast CSV per entry under `forecasts/<model_id>.csv`. Each entry now binds to one trainer's typed `Model Spec` output in **a single click** — no manual per-field toggling.

| Field | Value | Upstream |
|---|---|---|
| Models | One entry per active trainer — see below | ✓ |
| Datetime Column | `datetime` | — |
| Horizon Column | (empty unless `flag_each_day=true` on a preprocessor) | — |
| Max Horizon | (empty) | — |

### How to fill `Models` in the Domino UI

1. Click the **`+`** next to `Models` to add an entry.
2. On the entry, toggle **Upstream** on the entry as a whole.
3. Choose the upstream trainer's **`Model Spec`** output:
   - For the PVOUT branch *with* error correction: **`PVOUTErrorCorrectionModelTrain.Model Spec`**
   - For the PVOUT branch *without* error correction: **`PVOUTPredictionModelTrain.Model Spec`**
   - For the price branch: **`ElectricityPricePredictionModelTrain.Model Spec`**
4. That one binding pre-fills `model_id`, `mode`, `model_path`, `data_path`, `feature_columns`, `target_column`, `base_forecast_column`, and (for price) `preprocessing_metadata_path` with the trainer's known-good defaults. Override any field per-entry if you need to.

Repeat steps 1–3 for each target. For the canonical dual-target chain you'll end up with two entries: one bound to a PVOUT trainer's `Model Spec`, one bound to the Electricity trainer's `Model Spec`.

Produces one `forecasts/<model_id>.csv` per entry and emits `forecasts: list[ForecastEntry]` containing `{model_id, forecast_csv_path, data_path, mode, model_path, feature_columns, target_column}`. This list drives Evaluate / Explainable / Aggregator below — one single edge each.

**Mode reference (set in the trainer's `Model Spec` defaults; override per-entry if needed):**
- `pvout_correction` — `final_forecast = base_forecast_column + model.predict(X)`. Default for both PVOUT trainers.
- `price_level` — `final_forecast = model.predict(X)`, no baseline. Default for the electricity price trainer.
- `price_ahead` — `final_forecast = baseline_column + correction(X)`. Use when you have a separate baseline column and a correction model.

## EvaluateMLModelPiece *(sibling of Explainable + Aggregator)*

Wire it once to `InferencePiece.Forecasts` (single edge at the list level) and it auto-evaluates every model. No per-entry manual binding needed.

| Field | Value | Upstream |
|---|---|---|
| Forecasts | ← **InferencePiece.Forecasts** | ✓ |
| Evaluations | (leave empty unless you need to override defaults for a specific `model_id`) | — |
| Evaluation Option | `normal` | — |
| Baseline Id | `1` | — |
| Plot | unchecked | — |

**Auto-derived per forecast entry:**

| Auto-derived field | How it's chosen |
|---|---|
| `pred_df_path` | `forecast.forecast_csv_path` |
| `target_column` | `forecast.target_column` (e.g. `PVOUT`, `spot_price_eur_mwh`) |
| `forecast_column` | `correction` when `mode == pvout_correction`; else `final_forecast` |
| `model_id` | `forecast.model_id` (used in `metrics_<model_id>.json` filename) |

**Override for a specific model:** add an entry to `Evaluations` with a matching `model_id` and only the fields you want to change. Defaults from the bound `forecasts` list apply for everything else.

Produces `metrics_<model_id>.json` per entry plus an aggregated `artifacts.per_model = {model_id: metrics}`.

**Why `correction` for `pvout_correction` mode:** with `base_forecast_column=PVOUT`, `final_forecast = truth + correction`, so `final_forecast − PVOUT` is *just* `correction`. Scoring `correction` against `PVOUT` measures actual prediction quality.

## ExplainablePredictionPiece *(sibling of Evaluate + Aggregator)*

Same single-edge UX as Evaluate. Wire once to `InferencePiece.Forecasts`; the piece auto-runs SHAP per model.

| Field | Value | Upstream |
|---|---|---|
| Forecasts | ← **InferencePiece.Forecasts** | ✓ |
| Explanations | (leave empty unless overriding defaults for a specific `model_id`) | — |
| Explain | `true` (also flipped to `true` per entry automatically) | — |
| Explain Method | `shap` (default when `Explain=true`) | — |
| Use Diagnostic Loss | unchecked unless the upstream PVOUT correction trainer was built with `use_diagnostic_loss=True` | — |

**Auto-derived per forecast entry:** `model_path`, `data_path`, `feature_columns`, `target_column`, `model_id`. SHAP `TreeExplainer` is selected automatically for XGBoost.

Produces `artifacts.per_model[model_id].explainability` with `shap_values`, `feature_names`, `base_value` per model. Override via the `Explanations` field per `model_id` if needed.

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
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group0   
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group1   
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group2   
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group3   
  docker pull ghcr.io/iisas/spice_smart_grid_pieces:dev-3-group4   
  ```
- `git pull` first so `.domino/dependencies_map.json` matches the latest auto-organize commit; refresh the piece repository in Domino UI afterwards.
- **DataPreprocessing** has both `Data Path Solargis` and `Data Path Okte` wired (otherwise the merge falls back to a single-source dataset and the other branch's features are missing).
- **Time grids** of the two generators align: same `Time Step Minutes` and `Records Count`, similar `Start At`. The inner-join drops rows with no match.
- **ModelDecider_PVOUT** has `Target Column = PVOUT`; **ModelDecider_Price** has `Target Column = spot_price_eur_mwh`. The target-stripping logic relies on this.
- **Inference Models list** has one entry per active trainer, each bound at the entry level to the trainer's `Model Spec` output (one click). No literal `<placeholder>` strings — toggle Upstream on the entry instead.
- **If using PVOUT error-correction:** the PVOUT entry on Inference is bound to **`PVOUTErrorCorrectionModelTrain.Model Spec`**, not the baseline's.
- **Evaluate / Explainable / ForecastAggregator** each have their `Forecasts` field bound to `InferencePiece.Forecasts` (single edge at the list level — no per-entry literal text).
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
