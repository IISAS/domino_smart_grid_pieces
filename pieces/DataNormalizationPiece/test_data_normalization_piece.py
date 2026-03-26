from domino.testing import piece_dry_run
import os
import pytest


def test_data_normalization_piece_smoke():
    output_data = piece_dry_run(
        "DataNormalizationPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_data_normalization_piece_z_score():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip(
            "z_score unit test relies on in-process dataframe-like object; "
            "HTTP dry-run mode only supports JSON payloads."
        )

    try:
        import numpy as np
    except ModuleNotFoundError:
        pytest.skip("numpy is required for the z_score normalization test")

    class FakeDataFrame:
        """
        Minimal dataframe-like object for unit testing normalization logic.

        The piece only needs:
        - `columns`
        - `copy()`
        - `__getitem__` / `__setitem__` (column-wise)
        - numpy array column objects supporting min/max/mean/std
        - `to_dict(orient="list")` for serialization
        """

        def __init__(self, data):
            self.data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
            self.columns = list(self.data.keys())

        def copy(self):
            return FakeDataFrame({k: v.copy() for k, v in self.data.items()})

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = np.asarray(value, dtype=float)

        def to_dict(self, orient="list"):
            # Keep the same values as simple python lists for JSON-serializable output.
            return {k: v.tolist() for k, v in self.data.items()}

    df = FakeDataFrame({"a": [1, 2, 3]})

    output_data = piece_dry_run(
        "DataNormalizationPiece",
        {
            "payload": {
                "dataframe": df,
                "type": "z_score",
                "features": ["a"],
            }
        },
    )

    normalized = output_data["artifacts"]["normalized_data"]["a"]

    # Population std (numpy's default, ddof=0):
    # mean=2, std=sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3)=sqrt(2/3)
    expected = [-(np.sqrt(2 / 3) ** -1), 0.0, (np.sqrt(2 / 3) ** -1)]
    assert np.allclose(normalized, expected, rtol=1e-6, atol=1e-6)
