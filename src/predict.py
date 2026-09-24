"""Making predictions with the saved model.

The Predictor class:
* loads the trained pipeline + metadata once,
* validates user input with clear, human-readable messages,
* applies the exact same feature engineering as training,
* returns the prediction together with an honest explanation.

NOTE on uncertainty: this dataset contains almost no relationship between the
inputs and charging cost (cross-validation R2 < 0 for every model). We
therefore do NOT provide a fake confidence percentage. We report the typical
MAE of the model as "typical error" and clearly warn the user.
"""

import json
from typing import Dict

import joblib
import pandas as pd

from . import config
from .features import add_derived_features
from .preprocessing import clean_dataframe

BOUNDS = {
    "Battery Capacity (kWh)": (0, None, "kWh"),
    "Energy Consumed (kWh)": (0, None, "kWh"),
    "Charging Duration (hours)": (0, None, "hours"),
    "Charging Rate (kW)": (0, None, "kW"),
    "Distance Driven (since last charge) (km)": (0, None, "km"),
    "Vehicle Age (years)": (0, None, "years"),
    "State of Charge (Start %)": (0, 100, "%"),
    "State of Charge (End %)": (0, 100, "%"),
    "Temperature (°C)": (None, None, "°C"),
}
TEMPERATURE_UNITS = "°C"


def check_input(value, lo, hi, col, units):
    """Validate a single numeric input; return (clean_value, None) or
    (None, error_message)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, f"'{col}' must be a number."
    if lo is not None and value < lo:
        return None, f"'{col}' cannot be less than {lo:g} {units}."
    if hi is not None and value > hi:
        return None, f"'{col}' cannot be greater than {hi:g} {units}."
    return value, None


class Predictor:
    def __init__(self, model_file=None, metadata_file=None):
        self.model_file = model_file or config.MODEL_FILE
        self.metadata_file = metadata_file or config.METADATA_FILE
        if not self.model_file.exists():
            raise FileNotFoundError(
                "The trained model file is missing.\n"
                f"  Expected at: {self.model_file}\n"
                "Please run 'python -m src.train' first."
            )
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                "The model metadata file is missing.\n"
                f"  Expected at: {self.metadata_file}\n"
                "Please run 'python -m src.train' first."
            )
        self.pipeline = joblib.load(self.model_file)
        self.metadata = json.loads(self.metadata_file.read_text())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, inputs: Dict[str, float]):
        errors = {}
        cleaned = {}
        for col, (lo, hi, units) in BOUNDS.items():
            if col not in inputs or inputs[col] is None:
                errors[col] = f"'{col}' is required."
                continue
            value, err = check_input(inputs[col], lo, hi, col, units)
            if err:
                errors[col] = err
            else:
                cleaned[col] = value
        if not errors:
            soc_gain = cleaned["State of Charge (End %)"] - cleaned[
                "State of Charge (Start %)"]
            if soc_gain < 0:
                errors["State of Charge (End %)"] = (
                    "Ending state of charge should be at least the starting "
                    "value (charging raises the battery)."
                )
        if errors:
            raise ValueError("; ".join(f"{k}: {v}" for k, v in errors.items()))
        return cleaned

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, inputs: Dict[str, float]):
        """Return a dictionary with the prediction, unit and explanation."""
        cleaned = self.validate(inputs)

        # Build a single-row raw dataframe (same shape as the training rows).
        row = {col: cleaned.get(col) for col in config.NUMERIC_INPUTS}
        row.update({col: inputs.get(col) for col in config.CATEGORICAL_INPUTS})

        df = pd.DataFrame([row])
        df = add_derived_features(clean_dataframe(df))
        feature_cols = (
            config.NUMERIC_INPUTS + config.CATEGORICAL_INPUTS
            + config.DERIVED_FEATURES
        )
        X = df[feature_cols]

        value = float(self.pipeline.predict(X)[0])
        units = self.metadata.get("target_units", "USD")

        holdout = self.metadata.get("holdout_test", {})
        typical_error = holdout.get("mae")
        holdout_r2 = holdout.get("r2")

        explanation = self._explanation(inputs=cleaned)

        return {
            "value": round(value, 2),
            "units": units,
            "typical_error": round(typical_error, 2) if typical_error else None,
            "holdout_r2": holdout_r2,
            "reliable": bool(holdout_r2 and holdout_r2 > 0.3),
            "input_summary": cleaned,
            "explanation": explanation,
        }

    def _explanation(self, inputs):
        metadata = self.metadata
        top = (metadata.get("interpretability", {})
               .get("top_features", []))
        lines = [f"This estimate is based on your inputs below."]

        if top:
            lines.append(
                "The features the model leans on most are: "
                + ", ".join(top) + "."
            )
        reliable = metadata.get("holdout_test", {}).get("r2", None)
        if reliable is not None and reliable <= 0:
            lines.append(
                "Important: during validation this model could not beat "
                "simply guessing the average charging cost "
                f"(R² = {reliable:+.2f}). The number above is an illustrative "
                "projection, NOT a reliable forecast."
            )
        elif reliable is not None and reliable < 0.5:
            lines.append(
                f"The model explains only part of the cost variation "
                f"(R² = {reliable:+.2f}); treat the result as approximate."
            )
        else:
            lines.append("Prediction confidence is moderate.")
        return " ".join(lines)


if __name__ == "__main__":
    import sys

    predictor = Predictor()
    sample = {
        "Battery Capacity (kWh)": 75.0,
        "Energy Consumed (kWh)": 40.0,
        "Charging Duration (hours)": 2.5,
        "Charging Rate (kW)": 30.0,
        "State of Charge (Start %)": 30.0,
        "State of Charge (End %)": 85.0,
        "Distance Driven (since last charge) (km)": 150.0,
        "Temperature (°C)": 20.0,
        "Vehicle Age (years)": 4.0,
        "Vehicle Model": "Chevy Bolt",
        "Charging Station Location": "Los Angeles",
        "Time of Day": "Morning",
        "Day of Week": "Monday",
        "Charger Type": "Level 2",
        "User Type": "Commuter",
    }
    try:
        result = predictor.predict(sample)
        print("Prediction OK (run the app for the friendly interface):")
        print(f"  {result['value']} {result['units']}")
        print(f"  Typical error (hold-out MAE): {result['typical_error']}"
              f" {result['units']}")
        print(f"  Hold-out R2: {result['holdout_r2']:+.3f}")
        print(f"  Reliable forecast? {result['reliable']}")
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        sys.exit(1)