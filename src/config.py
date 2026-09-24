"""Central configuration for the EV charging cost prediction project."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Candidate locations so the code keeps working no matter where it is run from.
DATASET_CANDIDATES = [
    DATA_DIR / "ev_charging_patterns.csv",
    PROJECT_ROOT / "analysis" / "dataset" / "ev_charging_patterns.csv",
]

MODEL_FILE = MODELS_DIR / "model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"
COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
TARGET = "Charging Cost (USD)"
TARGET_UNITS = "USD"
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# Columns that are identifiers or already dropped timestamps. They never enter
# the model because they are useless for new predictions.
DROP_COLUMNS = [
    "User ID",
    "Charging Station ID",
    "Charging Start Time",
    "Charging End Time",
]

# Raw input columns used as features (before derived features are added).
NUMERIC_INPUTS = [
    "Battery Capacity (kWh)",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
]

CATEGORICAL_INPUTS = [
    "Vehicle Model",
    "Charging Station Location",
    "Time of Day",
    "Day of Week",
    "Charger Type",
    "User Type",
]

# Columns that can contain missing values (all in the same rows).
COLUMNS_WITH_MISSING = ["Energy Consumed (kWh)", "Charging Rate (kW)",
                        "Distance Driven (since last charge) (km)"]

# Derived features created in src/features.py. They are computed from the raw
# inputs above, so they never leak information about the target.
DERIVED_FEATURES = [
    "SOC Gain (%)",
    "Energy per km (kWh/km)",
    "Potential Energy (kWh)",
]

# NOTE: adjust the paths in this file if you ever move the project folders.