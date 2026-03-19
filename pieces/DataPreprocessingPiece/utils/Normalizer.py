class Normalizations:
    """
    Normalization utilities used by `DataPreprocessingPiece`.

    Implemented to be lightweight and to avoid importing heavy deps (`numpy`)
    at module import time (important for the `preprocessing_option=none` mode).
    """

    def logaritmic_normalization(self, X):
        import numpy as np  # type: ignore

        return np.log(X)

    def exponential_normalization(self, X):
        import numpy as np  # type: ignore

        return np.exp(X)

    def min_max_normalization(self, X):
        denom = X.max() - X.min()
        # Replace 0 denominators to avoid division-by-zero on constant columns.
        if hasattr(denom, "replace"):
            denom = denom.replace(0, 1)
        elif denom == 0:
            denom = 1
        return (X - X.min()) / denom

    def z_score_normalization(self, X):
        std = X.std()
        if hasattr(std, "replace"):
            std = std.replace(0, 1)
        elif std == 0:
            std = 1
        return (X - X.mean()) / std

    def normalize(self, X, type: str = None, features: list = None):
        """
        Normalize only `features` columns.

        `type` is one of: `logaritmic`, `exponential`, `min_max`, `z_score`.
        (We also accept `logarithmic` as a synonym for `logaritmic`.)
        """

        if type is None or features is None:
            return X

        normalization_type = "logaritmic" if type == "logarithmic" else type

        for feature in features:
            if feature in X.columns:
                if normalization_type == "logaritmic":
                    X[feature] = self.logaritmic_normalization(X[[feature]]).iloc[:, 0]
                elif normalization_type == "exponential":
                    X[feature] = self.exponential_normalization(X[[feature]]).iloc[:, 0]
                elif normalization_type == "min_max":
                    X[feature] = self.min_max_normalization(X[[feature]]).iloc[:, 0]
                elif normalization_type == "z_score":
                    X[feature] = self.z_score_normalization(X[[feature]]).iloc[:, 0]
                else:
                    raise ValueError(
                        "Invalid normalization type: "
                        f"{type}, select from logaritmic, exponential, min_max, z_score"
                    )

        return X
