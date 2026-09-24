"""Model interpretability.

Computes permutation importance for the saved model and, for linear models,
the raw coefficients. Results are merged into models/metadata.json and can be
shown by the dashboard.

Run with:
    python -m src.interpret
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .data_loader import load_dataset
from .features import add_derived_features
from .preprocessing import clean_dataframe

SAMPLE_SIZE = 400
N_REPEATS = 5


def _feature_matrix(df):
    df = add_derived_features(clean_dataframe(df))
    cols = config.NUMERIC_INPUTS + config.CATEGORICAL_INPUTS + config.DERIVED_FEATURES
    return df[cols], df[config.TARGET]


def _short_name(feature):
    """Turn sklearn encoded names into readable labels."""
    return feature.split("__", 1)[-1]


def compute_interpretability():
    if not config.MODEL_FILE.exists() or not config.METADATA_FILE.exists():
        raise FileNotFoundError(
            "No trained model found. Run 'python -m src.train' first."
        )

    pipeline = joblib.load(config.MODEL_FILE)
    metadata = json.loads(config.METADATA_FILE.read_text())

    df = load_dataset()
    X, y = _feature_matrix(df)

    # Work on a fixed random sample for speed; the goal is a rough picture of
    # which features matter, not an exact number.
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    # Shuffling a feature and watching MAE rise: the bigger the rise, the more
    # the prediction depends on that feature.
    result = permutation_importance(
        pipeline, X_sample, y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS, random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    # importances_mean = baseline - permuted score for a neg-MAE scorer, which
    # equals the increase in MAE when the feature is shuffled.
    importance = {
        _short_name(f): float(round(drop, 4))
        for f, drop in zip(X.columns, result.importances_mean)
    }

    # For linear/ridge models, also store raw coefficients on scaled features.
    coefficients = None
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_).ravel()
        names = [
            _short_name(f)
            for f in pipeline.named_steps["preprocess"].get_feature_names_out()
        ]
        coefficients = sorted(
            zip(names, coefs.tolist()),
            key=lambda t: abs(t[1]), reverse=True,
        )[:20]

    # Recompute importance as "drop in R2" is not possible for zero-signal
    # data, so we keep the MAE-increase view and label it clearly.
    sorted_importance = sorted(importance.items(), key=lambda t: t[1], reverse=True)

    metadata["interpretability"] = {
        "method": "permutation_importance",
        "scoring": "increase in MAE when the feature is randomly shuffled",
        "sample_size": int(len(X_sample)),
        "n_repeats": N_REPEATS,
        "importance": {k: v for k, v in sorted_importance},
        "top_features": [k for k, _ in sorted_importance[:5]],
        "coefficients": coefficients,
    }

    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print("=== Permutation importance (increase in MAE when shuffled) ===")
    if len(sorted_importance) <= 10:
        rows = sorted_importance
    else:
        rows = sorted_importance[:10]
    for feature, score in rows:
        print(f"{feature:45s} {score:+.4f}")
    print("\nHigher value = the model relies more on that feature.")
    print("NOTE: with this dataset every score is small because the data "
          "contains almost no relationship between inputs and cost.")

    if coefficients:
        print("\n=== Linear coefficients (top 20 by absolute value) ===")
        for name, coef in coefficients:
            print(f"{name:45s} {coef:+.4f}")


if __name__ == "__main__":
    compute_interpretability()

# Interpretation output is merged into models/metadata.json.