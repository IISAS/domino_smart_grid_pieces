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
| `payload.return_debug` | `bool` | `False` | Include debug outputs in artifacts |

Behavior:
- Empty payload -> no-op response.
- `stages` provided -> staged inference path.
- `mode` provided (without `stages`) -> single inference path.

## Output

| Field | Description |
|-------|-------------|
| `message` | Status string (`InferencePiece executed.` or no-op variant) |
| `artifacts.forecast` | Forecast output (inline rows / csv path / schema metadata depending on mode) |
| `artifacts.per_horizon` | Optional horizon-split outputs |
| `artifacts.metadata` | Runtime/inference metadata |
| `artifacts.stage_summaries` | Present for staged inference runs |
| `artifacts.debug` | Optional diagnostics when debug is enabled |

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
