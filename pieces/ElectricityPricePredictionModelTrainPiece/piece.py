from domino.base_piece import BasePiece

from .models import InputModel, ModelSpec, OutputModel
from .utils.model_decider import MODEL_TYPES, create_model


class ElectricityPricePredictionModelTrainPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        import csv

        self.logger.info("Running ElectricityPricePredictionModelTrainPiece.")

        payload = input_data.payload_as_dict()
        if not payload:
            return OutputModel(
                message="ElectricityPricePredictionModelTrainPiece template executed (no-op).",
                artifacts={"input_payload": payload},
            )

        model_type = str(payload.get("model_type", "xgb_regressor_model")).lower()
        model_params = payload.get("model_params") or {}
        setup = payload.get("model_setup") or {}
        feature_columns = setup.get("feature_columns") or payload.get("feature_columns")
        target_column = (
            setup.get("target_column")
            or payload.get("target_column")
            or "spot_price_eur_mwh"
        )

        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Available: {sorted(MODEL_TYPES)}"
            )
        if not feature_columns:
            raise ValueError("`feature_columns` is required (wire from DataPreprocessingPiece or ModelDeciderPiece).")

        def _load_rows_from_csv(path: str) -> list[dict]:
            with open(path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        rows = None
        data_path = payload.get("data_path") or payload.get("csv_path")
        tabular_data = payload.get("tabular_data") or payload.get("dataframe")
        if data_path:
            rows = _load_rows_from_csv(data_path)
        elif isinstance(tabular_data, list):
            rows = tabular_data
        elif isinstance(tabular_data, dict):
            keys = list(tabular_data.keys())
            n = len(tabular_data[keys[0]]) if keys else 0
            rows = [{k: tabular_data[k][i] for k in keys} for i in range(n)]
        else:
            raise ValueError(
                "Provide either `payload['data_path']`/`payload['csv_path']` or "
                "`payload['tabular_data']`."
            )

        if not rows:
            raise ValueError("No training rows were loaded.")

        try:
            import pandas as pd
        except ImportError as e:
            raise ValueError(
                "pandas is required for electricity price model training."
            ) from e

        df = pd.DataFrame(rows)
        numeric_columns = set(feature_columns + [target_column])
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        required_columns = [c for c in numeric_columns if c in df.columns]
        if required_columns:
            df = df.dropna(subset=required_columns)
        if df.empty:
            raise ValueError("No valid numeric rows available after preprocessing.")

        X = df[feature_columns]
        y = df[target_column]

        model = create_model(model_type=model_type, model_params=model_params)
        model.train(X, y)

        artifacts = self._build_artifacts(
            model=model,
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            model_params=model_params,
            payload=payload,
        )
        resolved_data_path = payload.get("data_path") or payload.get("csv_path")
        model_path = artifacts.get("checkpoint_path")
        preprocessing_metadata_path = artifacts.get("preprocessing_metadata_path")

        # Typed bundle for one-click upstream binding from InferencePiece.models[i].
        # `price_level` mode = direct regression with no baseline column.
        model_spec = ModelSpec(
            model_id="price",
            mode="price_level",
            model_path=model_path,
            data_path=resolved_data_path,
            preprocessing_metadata_path=preprocessing_metadata_path,
            feature_columns=list(feature_columns),
            target_column=str(target_column),
            base_forecast_column=None,
        )

        return OutputModel(
            message="ElectricityPricePredictionModelTrainPiece executed.",
            model_path=model_path,
            feature_columns=list(feature_columns),
            target_column=str(target_column),
            preprocessing_metadata_path=preprocessing_metadata_path,
            data_path=resolved_data_path,
            model_spec=[model_spec],
            artifacts=artifacts,
        )

    def _build_artifacts(
        self,
        model,
        model_type: str,
        feature_columns: list[str],
        target_column: str,
        model_params: dict,
        payload: dict,
    ) -> dict:
        import json
        import os
        import pickle
        import tempfile

        checkpoint_dir = payload.get("checkpoint_dir")
        if not checkpoint_dir:
            checkpoint_dir = str(getattr(self, "results_path", tempfile.gettempdir()))
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"electricity_price_prediction_{model_type}.pkl"
        )

        trained_model_metadata = {
            "model_type": model_type,
            "feature_columns": feature_columns,
            "feature_columns_used": feature_columns,
            "target_column": target_column,
            "params": model_params,
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {"metadata": trained_model_metadata, "trained_model_object": model},
                f,
            )

        meta_path = os.path.join(checkpoint_dir, "preprocessing_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(trained_model_metadata, f, indent=2)

        return {
            "trained_model": trained_model_metadata,
            "checkpoint_path": checkpoint_path,
            "preprocessing_metadata_path": meta_path,
            "train_metrics": {},
        }
