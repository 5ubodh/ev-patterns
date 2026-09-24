"""Feature engineering: physically sensible derived features.

The base inputs all come from the cleaned dataset. The derived features below
are standard EV metrics and are computed only from those inputs, so they never
contain information about the target (Charging Cost).
"""

import pandas as pd


def add_derived_features(df):
    """Add derived EV metrics to a cleaned dataframe and return it.

    New columns
    -----------
    * SOC Gain (%)            - battery percentage added during the session
                               (End SOC minus Start SOC).
    * Energy per km (kWh/km)  - energy consumption rate per kilometre driven
                               since the last charge.
    * Potential Energy (kWh)  - physical energy the session *should* have added
                               (= battery capacity * SOC gain / 100). Kept as a
                               feature because the raw recording varies.
    """
    df = df.copy()

    df["SOC Gain (%)"] = (
        df["State of Charge (End %)"] - df["State of Charge (Start %)"]
    )

    distance = df["Distance Driven (since last charge) (km)"]
    energy = df["Energy Consumed (kWh)"]
    df["Energy per km (kWh/km)"] = energy / distance.replace(0, pd.NA)

    df["Potential Energy (kWh)"] = (
        df["Battery Capacity (kWh)"] * df["SOC Gain (%)"] / 100.0
    )

    return df