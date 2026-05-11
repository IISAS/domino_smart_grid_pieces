from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


class Normalizations:
    """
    Lightweight normalization helpers.

    This mirrors the logic from `Normalizations.py` in your other project, but operates on
    a "dataframe-like" object: it only requires `columns`, `__getitem__`/`__setitem__`,
    and column-wise aggregations (`min`, `max`, `mean`, `std`).
    """

    def logaritmic_normalization(self, col):
        import numpy as np  # type: ignore

        return np.log(col)

    def exponential_normalization(self, col):
        import numpy as np  # type: ignore

        return np.exp(col)

    def min_max_normalization(self, col):
        col_min = col.min()
        col_max = col.max()
        denom = col_max - col_min
        # Avoid division-by-zero for constant columns.
        if denom == 0:
            return col * 0
        return (col - col_min) / denom

    def z_score_normalization(self, col):
        col_mean = col.mean()
        col_std = col.std()
        if col_std == 0:
            return col * 0
        return (col - col_mean) / col_std

    def normalize(self, X, type: str = None, features: list = None):
        if type is None:
            return X

        # Default to all columns if `features` isn't provided.
        if features is None:
            features = list(getattr(X, "columns", []))

        # Normalize only specified features.
        for feature in features:
            if feature in getattr(X, "columns", []):
                # Backward-compat: accept both spellings.
                normalization_type = type
                if normalization_type == "logarithmic":
                    normalization_type = "logaritmic"

                if normalization_type == "logaritmic":
                    X[feature] = self.logaritmic_normalization(X[feature])
                elif normalization_type == "exponential":
                    X[feature] = self.exponential_normalization(X[feature])
                elif normalization_type == "min_max":
                    X[feature] = self.min_max_normalization(X[feature])
                elif normalization_type == "z_score":
                    X[feature] = self.z_score_normalization(X[feature])
                else:
                    raise ValueError(
                        f"Invalid normalization type: {type}, select from "
                        "logaritmic, exponential, min_max, z_score"
                    )

        return X


class DataNormalizationPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        def _to_serializable_dataframe_like(df_like):
            """
            Convert a dataframe-like object into a JSON-serializable dict.
            """
            if df_like is None:
                return None

            if hasattr(df_like, "to_dict"):
                try:
                    data = df_like.to_dict(orient="list")
                except TypeError:
                    data = df_like.to_dict()
            elif hasattr(df_like, "data"):
                # Used by our small test fake.
                data = df_like.data
            else:
                return str(df_like)

            # Ensure numpy arrays/scalars become plain python lists/numbers.
            for k, v in list(data.items()):
                if hasattr(v, "tolist"):
                    data[k] = v.tolist()
                elif isinstance(v, tuple):
                    data[k] = list(v)
            return data

        payload = input_data.payload_as_dict()
        df = payload.get("dataframe") or payload.get("X") or payload.get("data")
        normalization_type = payload.get("type") or payload.get("normalization_type")
        features = payload.get("features")

        if isinstance(features, str):
            features = [features]

        if df is None:
            self.logger.info("No dataframe provided; skipping normalization.")
            return OutputModel(
                message="No dataframe provided; skipping normalization.",
                artifacts={"input_payload": payload},
            )

        # Work on a copy when possible to avoid surprising callers.
        df_out = df.copy() if hasattr(df, "copy") else df
        normalizer = Normalizations()
        df_out = normalizer.normalize(
            df_out, type=normalization_type, features=features
        )

        applied_features = (
            features if features is not None else list(getattr(df_out, "columns", []))
        )

        return OutputModel(
            message="DataNormalizationPiece executed.",
            artifacts={
                "normalized_data": _to_serializable_dataframe_like(df_out),
                "normalization_type": normalization_type,
                "features": applied_features,
            },
        )
