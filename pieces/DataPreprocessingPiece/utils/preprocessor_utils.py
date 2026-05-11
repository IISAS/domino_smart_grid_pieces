def flag_each_day(df):
    """
    Flag each day with a unique identifier.

    All rows from the same day will have the same `pred_sequence_id` number.
    """
    import pandas as pd  # type: ignore

    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    unique_dates = df["date"].unique()
    date_to_id = {d: idx + 1 for idx, d in enumerate(sorted(unique_dates))}
    df["pred_sequence_id"] = df["date"].map(date_to_id)
    df = df.drop(columns=["date"])
    return df


def preprocess_solargis_data(data):
    """
    Drop rows with missing values and engineer a few CIS panel features.
    """
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    data = data.copy()
    # Be robust whether `datetime` is already datetime-like or still string-like.
    try:
        data["datetime"] = pd.to_datetime(data["datetime"], format="%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        data["datetime"] = pd.to_datetime(data["datetime"])

    # Filter out night hours
    original_rows = len(data)
    data = data[data["GHI"] > 1].copy()

    # Feature engineering
    data["diffuse_fraction"] = np.where(data["GHI"] > 0, data["DIF"] / data["GHI"], 0)
    data["solar_elevation_sin"] = np.sin(np.radians(data["SE"]))
    data["hour_of_day"] = data["datetime"].dt.hour

    print(f"[INFO] Processed {len(data)} rows out of {original_rows} original rows")
    return data
