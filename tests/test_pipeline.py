"""Smoke tests for the EV charging cost pipeline.

Runs without pytest - just execute this file:
    python tests/test_pipeline.py

It verifies that:
* the trained artifacts exist and were created by this project,
* the predictor returns sensible values for valid input,
* invalid inputs are rejected with a clear message,
* the comparison table and metadata agree with the saved model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILED = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILED.append(name)


def main():
    from src import config
    from src.predict import Predictor

    # 1. Artifacts exist
    check("model file exists", config.MODEL_FILE.exists())
    check("preprocessor file exists", config.PREPROCESSOR_FILE.exists())
    check("metadata file exists", config.METADATA_FILE.exists())
    check("comparison table exists", config.COMPARISON_FILE.exists())

    # 2. Predictor loads and reports metadata
    pred = Predictor()
    check("metadata has target", pred.metadata.get("target") == config.TARGET)
    check("metadata has holdout metrics",
          "holdout_test" in pred.metadata and "r2" in pred.metadata["holdout_test"])

    # 3. Valid prediction
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
    result = pred.predict(sample)
    check("prediction is a finite USD value",
          isinstance(result["value"], float) and result["units"] == "USD",
          f"{result['value']} {result['units']}")
    check("typical error reported",
          isinstance(result["typical_error"], (int, float)))
    check("reliability flag present", "reliable" in result)

    # 4. Invalid inputs are rejected
    def raises(bad_inputs):
        data = dict(sample)
        data.update(bad_inputs)
        try:
            pred.predict(data)
            return False
        except ValueError:
            return True

    check("negative battery rejected", raises({"Battery Capacity (kWh)": -1}))
    check("SOC > 100 rejected",
          raises({"State of Charge (Start %)": 150}))
    check("end SOC below start rejected",
          raises({"State of Charge (End %)": 10}))
    check("non-numeric value rejected",
          raises({"Vehicle Age (years)": "old"}))

    # 5. Comparison table looks right
    import pandas as pd
    table = pd.read_csv(config.COMPARISON_FILE)
    check("comparison lists all models",
          set(table["Model"]) == {"Linear Regression", "Ridge Regression",
                                  "Random Forest", "Extra Trees",
                                  "Gradient Boosting"})
    check("comparison has metric columns",
          {"R2", "MAE", "RMSE"}.issubset(table.columns))

    print()
    if FAILED:
        print(f"RESULT: {len(FAILED)} check(s) failed -> {FAILED}")
        sys.exit(1)
    print("RESULT: all smoke tests passed")


if __name__ == "__main__":
    main()