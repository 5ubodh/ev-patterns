"""Reporting the stored model comparison and metrics.

Run with:
    python -m src.evaluate            # print the saved results
    python -m src.evaluate --data path/to/extra.csv   # also score on extra data
"""

import argparse

import pandas as pd

from . import config
from .data_loader import load_dataset
from .features import add_derived_features
from .preprocessing import clean_dataframe


def load_comparison():
    if not config.COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Missing results file '{config.COMPARISON_FILE}'. "
            "Run 'python -m src.train' first to create the model and results."
        )
    return pd.read_csv(config.COMPARISON_FILE)


def load_metadata():
    if not config.METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata file '{config.METADATA_FILE}'. "
            "Run 'python -m src.train' first."
        )
    import json
    return json.loads(config.METADATA_FILE.read_text())


def print_report():
    comparison = load_comparison()
    metadata = load_metadata()

    print("=" * 60)
    print(f"EV charging cost model report")
    print(f"Target: {metadata['target']}  ({metadata['target_units']})")
    print("=" * 60)

    print("\nCross-validation comparison (5-fold, lower is better for MAE/RMSE):")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\nSelected model     : {metadata['model']}")
    print(f"Naive baseline MAE : {metadata['baseline_mae']:.3f} "
          "(always predicting the average)")
    hold = metadata.get("holdout_test", {})
    print(f"Hold-out test (20%): R2={hold.get('r2', '?'):+}  "
          f"MAE={hold.get('mae', '?')}  RMSE={hold.get('rmse', '?')}")


def score_extra_data(path):
    """Predict on a separate CSV file that also contains the target column."""
    import joblib
    from sklearn.metrics import mean_absolute_error, r2_score

    pipeline = joblib.load(config.MODEL_FILE)
    df = clean_dataframe(load_dataset(path))
    X, y = make_feature_matrix_from_df(df)
    y_pred = pipeline.predict(X)
    print(f"\nExtra data ({len(X)} rows): R2={r2_score(y, y_pred):+.4f}  "
          f"MAE={mean_absolute_error(y, y_pred):.3f}")


def make_feature_matrix_from_df(df):
    df = add_derived_features(df)
    feature_cols = (
        config.NUMERIC_INPUTS + config.CATEGORICAL_INPUTS + config.DERIVED_FEATURES
    )
    return df[feature_cols], df[config.TARGET]


def main():
    parser = argparse.ArgumentParser(description="Show model evaluation results.")
    parser.add_argument("--data", default=None,
                        help="Optional CSV to score with the saved model.")
    args = parser.parse_args()

    print_report()
    if args.data:
        score_extra_data(args.data)


if __name__ == "__main__":
    main()

# Results are stored in models/; run 'python -m src.train' first.