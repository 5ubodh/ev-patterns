"""Loading the EV charging dataset from a reliable location."""

from pathlib import Path

import pandas as pd

from . import config


def resolve_dataset_path(explicit_path=None):
    """Return the path of the dataset, searching known locations.

    Raises FileNotFoundError with a human-readable message if the dataset
    cannot be found anywhere.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Could not find the dataset at '{explicit_path}'. "
            "Please check the path."
        )

    for candidate in config.DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find 'ev_charging_patterns.csv'. Expected it in one of:\n"
        + "\n".join(f"  - {p}" for p in config.DATASET_CANDIDATES)
        + "\nMake sure the dataset is present before running the pipeline."
    )


def load_dataset(explicit_path=None):
    """Load the dataset as a pandas DataFrame."""
    path = resolve_dataset_path(explicit_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The dataset at '{path}' is empty.")
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from: {path}")
    # Nothing else happens here; cleaning lives in preprocessing.py.
    return df