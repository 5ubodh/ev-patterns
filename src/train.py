"""Training pipeline.

Loads the dataset, builds the feature matrix, evaluates several regression
models with 5-fold cross-validation, selects the best one, and saves the
trained model + preprocessor + results to the models/ folder.

Run with:
    python -m src.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data_loader import load_dataset
from .features import add_derived_features
from .preprocessing import clean_dataframe

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=config.RANDOM_SEED),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=config.RANDOM_SEED),
}

SCORING = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]


def build_preprocessor(numeric_cols, categorical_cols):
    """Return a ColumnTransformer: median-impute + scale numerics,
    one-hot encode categoricals. Unknown categories are ignored so new data
    can always be transformed."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])


def make_feature_matrix(df):
    """Return (X, y) after cleaning and feature engineering."""
    cleaned = add_derived_features(clean_dataframe(df))
    feature_cols = (
        config.NUMERIC_INPUTS
        + config.CATEGORICAL_INPUTS
        + config.DERIVED_FEATURES
    )
    if len({c for c in feature_cols if c in cleaned.columns}) != len(feature_cols):
        missing = [c for c in feature_cols if c not in cleaned.columns]
        raise ValueError(f"Missing expected columns in data: {missing}")

    X = cleaned[feature_cols]
    y = cleaned[config.TARGET]
    return X, y


def run_cross_validation(pipeline, X, y, cv):
    """Return a dict of mean metrics (R2, MAE, RMSE) from k-fold CV."""
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1,
        return_train_score=False,
    )
    return {
        "R2": float(np.mean(results["test_r2"])),
        "MAE": float(-np.mean(results["test_neg_mean_absolute_error"])),
        "RMSE": float(-np.mean(results["test_neg_root_mean_squared_error"])),
    }


def summary_table(cv_scores, order_by="R2"):
    """Convert the CV results into a readable comparison DataFrame."""
    rows = []
    for name, metrics in cv_scores.items():
        row = {"Model": name}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(order_by, ascending=False)
    return table.reset_index(drop=True)


def main():
    df = load_dataset()
    X, y = make_feature_matrix(df)

    numeric_cols = [
        c for c in config.NUMERIC_INPUTS + config.DERIVED_FEATURES
        if c in X.columns
    ]
    categorical_cols = [c for c in config.CATEGORICAL_INPUTS if c in X.columns]

    # Hold-out test set for the final, honest evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True,
               random_state=config.RANDOM_SEED)

    print(f"\n=== Model comparison: predicting {config.TARGET} ===\n")

    cv_scores = {}
    for name, model in MODELS.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ])
        metrics = run_cross_validation(pipeline, X_train, y_train, cv)
        cv_scores[name] = metrics
        print(f"{name:20s} R2={metrics['R2']:+.4f}  "
              f"MAE={metrics['MAE']:6.3f}  RMSE={metrics['RMSE']:6.3f}")

    comparison = summary_table(cv_scores)
    print("\n=== Cross-validation comparison (5-fold, on training data) ===")
    print(comparison.to_string(index=False))

    # Average baseline for reference: predicting the mean charging cost.
    baseline_mae = np.abs(y_train - y_train.mean()).mean()
    print(f"\nNaive baseline (always predict the mean): MAE={baseline_mae:.3f}")

    # ------------------------------------------------------------------
    # Select the best model. If two scores are essentially tied, prefer the
    # simpler model (earlier in the MODELS dictionary) to keep things robust.
    # ------------------------------------------------------------------
    best_name = max(
        cv_scores,
        key=lambda name: (cv_scores[name]["R2"], -list(MODELS).index(name)),
    )
    print(f"\nSelected model: {best_name}")

    # Refit the best model on the FULL dataset so the shipped model sees all
    # the data. The metrics above were computed on (part of) the same data, so
    # they are the honest estimate of real performance.
    best_model = MODELS[best_name]
    best_numeric_cols = [c for c in numeric_cols]
    final_pipeline = Pipeline([
        ("preprocess", build_preprocessor(best_numeric_cols, categorical_cols)),
        ("model", best_model),
    ])
    final_pipeline.fit(X, y)

    preprocessor = final_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Final evaluation on the held-out test set.
    y_pred = final_pipeline.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    print(f"Hold-out test metrics ({config.TEST_SIZE:.0%} of data): "
          f"R2={r2:+.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    # ------------------------------------------------------------------
    # Persist everything
    # ------------------------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, config.MODEL_FILE)
    joblib.dump(preprocessor, config.PREPROCESSOR_FILE)
    comparison.to_csv(config.COMPARISON_FILE, index=False)

    metadata = {
        "target": config.TARGET,
        "target_units": config.TARGET_UNITS,
        "model": best_name,
        "framework": "scikit-learn",
        "n_rows_trained_on": int(len(X)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "numeric_inputs": config.NUMERIC_INPUTS,
        "categorical_inputs": config.CATEGORICAL_INPUTS,
        "derived_features": config.DERIVED_FEATURES,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.N_CV_FOLDS,
        "random_seed": config.RANDOM_SEED,
        "cv_scores": {k: {m: round(v, 5) for m, v in s.items()}
                      for k, s in cv_scores.items()},
        "holdout_test": {"r2": round(r2, 5), "mae": round(mae, 5),
                         "rmse": round(rmse, 5)},
        "baseline_mae": float(round(baseline_mae, 5)),
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model  ->  {config.MODEL_FILE}")
    print(f"Saved preprocessing ->  {config.PREPROCESSOR_FILE}")
    print(f"Saved metadata      ->  {config.METADATA_FILE}")
    print(f"Saved comparison    ->  {config.COMPARISON_FILE}")


if __name__ == "__main__":
    main()

# Reproducibility is controlled by config.RANDOM_SEED (split + folds).