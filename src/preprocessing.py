"""Data cleaning: drop useless columns, fix dtypes, remove duplicates and
clamp physically impossible values. Missing values are handled later inside
the scikit-learn pipeline (median imputation) so that cross-validation does
not leak information.
"""

import pandas as pd

from . import config


def clean_dataframe(df):
    """Return a cleaned copy of the raw EV charging dataset.

    Steps
    -----
    * Drop identifier / timestamp columns (useless for new predictions).
    * Remove exact duplicate rows (defensive; none present in this dataset).
    * Clamp State-of-Charge values to the physically valid 0-100 % range.
    * Clamp negative vehicle ages and negative energy/rate/duration values.
    """
    df = df.copy()

    # 1. Drop columns that cannot be used to predict a new session.
    missing_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=missing_cols)

    # 2. Remove exact duplicates (none exist, but stay safe).
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Removed {before - len(df)} duplicate rows.")

    # 3. Fix physically impossible State-of-Charge values.
    for col in ["State of Charge (Start %)", "State of Charge (End %)"]:
        if col in df.columns:
            n = ((df[col] < 0) | (df[col] > 100)).sum()
            if n:
                print(f"Clipped {n} out-of-range values in '{col}' to 0-100 %.")
            df[col] = df[col].clip(lower=0.0, upper=100.0)

    # 4. Fix impossible negative values.
    for col in ["Vehicle Age (years)", "Battery Capacity (kWh)",
                "Energy Consumed (kWh)", "Charging Duration (hours)",
                "Charging Rate (kW)", "Distance Driven (since last charge) (km)"]:
        if col in df.columns:
            n = (df[col] < 0).sum()
            if n:
                print(f"Clipped {n} negative values in '{col}' to 0.")
            df[col] = df[col].clip(lower=0.0)

    return df