"""EV charging cost dashboard.

Run from the project root with:
    streamlit run app/app.py

Sections
--------
Predict:      enter EV session details -> get a cost estimate + explanation.
Model:        cross-validated model comparison and final metrics.
Importance:   which features influence the model (permutation importance).
Data:         dataset insights (distributions, correlations, missing values).

The project ships with a training pipeline (python -m src.train). The
dashboard only loads the saved model - it never retrains.
"""

import sys
from pathlib import Path

# Allow importing the src package when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_dataset
from src.features import add_derived_features
from src.predict import Predictor
from src.preprocessing import clean_dataframe

import matplotlib.pyplot as plt

st.set_page_config(page_title="EV Charging Cost", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return Predictor()


@st.cache_resource
def get_dataframe():
    df = load_dataset()
    return add_derived_features(clean_dataframe(df))


def get_raw_dataframe():
    return pd.read_csv(config.DATASET_CANDIDATES[0])


@st.cache_resource
def get_comparison():
    if not config.COMPARISON_FILE.exists():
        return None
    return pd.read_csv(config.COMPARISON_FILE)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("⚡ EV Charging Cost Estimator")
st.caption("Predicts the cost of an EV charging session from user-friendly "
           "vehicle and trip details.")

try:
    predictor = get_predictor()
    metadata = predictor.metadata
    model_ready = True
except FileNotFoundError as exc:
    st.error(f"**Model not found.** {exc}")
    st.info("Run the training pipeline once:\n\n    pip install -r requirements.txt\n"
            "    python -m src.train\n    python -m src.interpret")
    model_ready = False
    metadata = {}

if model_ready:
    tab_predict, tab_model, tab_importance, tab_data = st.tabs(
        ["Predict", "Model performance", "Feature importance", "Dataset insights"]
    )

    # ------------------------------------------------------------------
    # INPUTS (shared, in the sidebar)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("EV session details")
        st.caption(f"All values are used to estimate the charging cost "
                   f"({metadata.get('target_units', 'USD')}).")

        raw = get_raw_dataframe()
        vehicle_models = sorted(raw["Vehicle Model"].dropna().unique())
        locations = sorted(raw["Charging Station Location"].dropna().unique())
        times = sorted(raw["Time of Day"].dropna().unique())
        days = sorted(raw["Day of Week"].dropna().unique())
        chargers = sorted(raw["Charger Type"].dropna().unique())
        user_types = sorted(raw["User Type"].dropna().unique())

        inputs = {}
        inputs["Battery Capacity (kWh)"] = st.number_input(
            "Battery capacity (kWh)", min_value=0.0, value=75.0, step=5.0)
        inputs["Energy Consumed (kWh)"] = st.number_input(
            "Energy delivered (kWh)", min_value=0.0, value=40.0, step=1.0)
        inputs["Charging Duration (hours)"] = st.number_input(
            "Charging duration (hours)", min_value=0.0, value=2.5, step=0.5)
        inputs["Charging Rate (kW)"] = st.number_input(
            "Charging rate (kW)", min_value=0.0, value=30.0, step=1.0)
        inputs["State of Charge (Start %)"] = st.number_input(
            "Battery level when starting (%)", min_value=0.0, max_value=100.0,
            value=30.0, step=5.0)
        inputs["State of Charge (End %)"] = st.number_input(
            "Battery level when done (%)", min_value=0.0, max_value=100.0,
            value=85.0, step=5.0)
        inputs["Distance Driven (since last charge) (km)"] = st.number_input(
            "Distance since last charge (km)", min_value=0.0, value=150.0, step=10.0)
        inputs["Temperature (°C)"] = st.number_input(
            "Outside temperature (°C)", value=20.0, step=1.0)
        inputs["Vehicle Age (years)"] = st.number_input(
            "Vehicle age (years)", min_value=0.0, value=4.0, step=1.0)
        inputs["Vehicle Model"] = st.selectbox("Vehicle model", vehicle_models)
        inputs["Charging Station Location"] = st.selectbox("City", locations)
        inputs["Time of Day"] = st.selectbox("Time of day", times)
        inputs["Day of Week"] = st.selectbox("Day of week", days)
        inputs["Charger Type"] = st.selectbox("Charger type", chargers)
        inputs["User Type"] = st.selectbox("User type", user_types)

    # ------------------------------------------------------------------
    # TAB 1 - PREDICTION
    # ------------------------------------------------------------------
    with tab_predict:
        st.header("Prediction")
        if st.button("Estimate charging cost", type="primary"):
            try:
                result = predictor.predict(inputs)
                st.metric(
                    "Estimated charging cost",
                    f"{result['value']:,.2f} {result['units']}",
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Typical error*", f"± {result['typical_error']} "
                                              f"{result['units']}")
                col2.metric("Hold-out R²", f"{result['holdout_r2']:+.3f}")
                col3.metric(
                    "Forecast quality",
                    "Not reliable" if not result["reliable"] else "Approximate",
                )
                st.caption("*Typical error = mean absolute error measured on "
                           "the hold-out test set. It is the average distance "
                           "between real costs and predictions - not an "
                           "official confidence interval.")
                st.info(result["explanation"])

                st.subheader("Inputs used")
                st.dataframe(
                    pd.DataFrame(
                        [result["input_summary"]]
                    ).T.rename(columns={0: "Value"})
                    .reset_index().rename(columns={"index": "Feature"}),
                    width='stretch',
                )
            except ValueError as exc:
                st.error(str(exc))

    # ------------------------------------------------------------------
    # TAB 2 - MODEL PERFORMANCE
    # ------------------------------------------------------------------
    with tab_model:
        st.header("Model performance")
        comparison = get_comparison()
        cv_scores = metadata.get("cv_scores", {})
        if comparison is not None:
            st.subheader("Cross-validation comparison (5-fold)")
            st.dataframe(comparison, width='stretch')
        elif cv_scores:
            st.dataframe(pd.DataFrame(cv_scores).T, width='stretch')

        hold = metadata.get("holdout_test", {})
        st.subheader("Final model (hold-out 20% test set)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metadata.get("model", "?"))
        col2.metric("R²", f"{hold.get('r2', 0):+.3f}")
        col3.metric("MAE", f"{hold.get('mae', 0):.2f} {metadata.get('target_units','USD')}")
        col4.metric("RMSE", f"{hold.get('rmse', 0):.2f}")

        st.subheader("Honest read of these numbers")
        st.warning(
            "Validation shows this model cannot reliably predict charging cost "
            f"(R² ≈ {hold.get('r2', 0):+.2f}, barely above guessing the "
            "average). The dataset appears to be synthetic and contains almost "
            "no relationship between the available features and charging cost. "
            "Use the estimates as illustrative examples only."
        )

    # ------------------------------------------------------------------
    # TAB 3 - FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    with tab_importance:
        st.header("Feature importance")
        interp = metadata.get("interpretability")
        if not interp:
            st.info("Importance not computed yet. Run "
                    "`python -m src.interpret` and restart the app.")
        else:
            imp = pd.DataFrame(
                list(interp["importance"].items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=False)
            imp["Importance"] = imp["Importance"].clip(lower=0)
            top = imp.head(10)
            st.bar_chart(top.set_index("Feature"))
            st.caption("Permutation importance: how much the average error "
                       "increases when a feature's values are randomly "
                       "shuffled. Higher = the model leans on it more.")
            st.info(
                "Scores are tiny for every feature because this dataset holds "
                "almost no signal. No single factor clearly drives charging "
                "cost here."
            )

    # ------------------------------------------------------------------
    # TAB 4 - DATASET INSIGHTS
    # ------------------------------------------------------------------
    with tab_data:
        st.header("Dataset insights")
        try:
            df = get_dataframe()
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Could not load the dataset.\n{exc}")
            df = None

        if df is not None:
            st.write(f"**Shape:** {df.shape[0]} charging sessions, "
                     f"{df.shape[1]} features.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", len(df))
            m2.metric("Columns", df.shape[1])
            m3.metric("Missing values (before model imputation)",
                      int(df.isna().sum().sum()))

            target = config.TARGET
            num_cols = [c for c in config.NUMERIC_INPUTS if c in df.columns]
            tabs = st.tabs(["Charging cost", "Other numeric", "Correlations"])
            with tabs[0]:
                st.subheader("Charging cost distribution")
                fig, ax = plt.subplots()
                ax.hist(df[target], bins=30, color="#4c78a8",
                        edgecolor="white")
                ax.set_xlabel(
                    f"Charging cost ({metadata.get('target_units', 'USD')})")
                ax.set_ylabel("Sessions")
                st.pyplot(fig)
            with tabs[1]:
                st.subheader("Numeric feature summary")
                st.dataframe(df[num_cols].describe().round(2),
                             width='stretch')
            with tabs[2]:
                st.subheader("Correlation with charging cost")
                corr = (
                    df[num_cols + [target]].corr()[target]
                    .drop(labels=[target])
                    .sort_values(ascending=False)
                )
                st.bar_chart(corr)
                st.caption("Values near zero mean that feature does not "
                           "explain the cost by itself, which is consistent "
                           "with the weak model performance.")

else:
    st.stop()