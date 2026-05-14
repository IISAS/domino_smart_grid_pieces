# InferencePiece

> **Prototype** - This piece is under active development and its interface may change.

A Domino piece for running forecast inference from trained model artifacts. It supports single-stage and staged inference flows and multiple forecast modes (PVOUT correction and electricity price modes), using prepared tabular inputs from upstream pipeline steps.

## Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `payload.mode` | `str` | `None` | Inference mode (`pvout_correction`, `price_ahead`, `price_level`) |
| `payload.stages` | `list[dict]` | `None` | Multi-stage inference definition; when present, staged inference is executed |
| `payload.model_path` | `str` | `None` | Path to model artifact (`.pkl` / joblib / booster format depending on mode) |
| `payload.input.tabular_data` / `payload.input.data_path` | `list[dict] \| str` | `None` | Prepared tabular input data |
| `payload.feature_columns` | `list[str]` | `None` | Explicit feature list, if not taken from preprocessing metadata |
| `payload.preprocessing_metadata_path` | `str` | `None` | Optional metadata file from training/preprocessing |
| `payload.forecast_output_csv_path` | `str` | `<results_path>/forecast.csv` | Where the forecast CSV is written for debugging; defaults to a file in the piece's results directory |
| `payload.return_debug` | `bool` | `False` | Include debug outputs in artifacts |

Behavior:
- Empty payload -> no-op response.
- `stages` provided -> staged inference path.
- `mode` provided (without `stages`) -> single inference path.
- A `forecast.csv` is always written under the piece's `results_path` so the forecast is inspectable on disk (overridable with `forecast_output_csv_path`).

## Output

| Field | Description |
|-------|-------------|
| `message` | Status string (`InferencePiece executed.` or no-op variant) |
| `artifacts.forecast` | Forecast output (inline rows / csv path / schema metadata depending on mode) |
| `artifacts.per_horizon` | Optional horizon-split outputs |
| `artifacts.metadata` | Runtime/inference metadata |
| `artifacts.stage_summaries` | Present for staged inference runs |
| `artifacts.debug` | Optional diagnostics when debug is enabled |
| `artifacts.forecast.csv_path` | Filesystem path of the forecast CSV (also surfaced via `display_result` in the Domino GUI) |

## Typical Workflow

```
Train pieces (PVOUT / Electricity price)
  -> produce model checkpoints + preprocessing metadata
  -> InferencePiece (single mode or staged pipeline)
  -> forecast artifacts for downstream consumers
```

## Running Tests

```bash
# From repository root
pytest pieces/InferencePiece/test_inference_piece.py -v
```
