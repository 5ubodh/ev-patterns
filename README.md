# ⚡ EV Charging Cost Estimation

A beginner-readable, end-to-end data-science application that estimates the
**cost (USD) of an EV charging session** from vehicle, battery, trip and
weather details. It includes a reproducible training pipeline, model
comparison, feature-importance analysis and a Streamlit dashboard.

---

## Problem statement

Electric-vehicle owners and charging-station operators want a quick estimate of
what a charging session will cost. This project builds a regression model from
a public EV charging log and exposes it through a simple web dashboard.

## Dataset

`data/ev_charging_patterns.csv` — 1,320 charging sessions × 20 columns.

| Column | Description |
|---|---|
| Battery Capacity (kWh) | Size of the vehicle battery |
| Energy Consumed (kWh) | Energy delivered in the session |
| Charging Duration (hours) | Session length |
| Charging Rate (kW) | Average charging power |
| Charging Cost (USD) | **Target variable** – cost of the session |
| State of Charge Start/End (%) | Battery level before/after charging |
| Distance Driven since last charge (km) | Distance covered since last charge |
| Temperature (°C) | Outdoor temperature |
| Vehicle Age (years) | Age of the vehicle |
| Vehicle Model, Location, Time of Day, Day of Week, Charger Type, User Type | Categorical context (one-hot encoded) |

**Data quality issues handled by the pipeline**

* 66 rows missing `Energy Consumed`, `Charging Rate`, `Distance Driven`
  → median imputed **inside the scikit-learn pipeline** (no leakage).
* 32 rows with State-of-Charge outside the valid 0–100 % range → clipped.
* Negative vehicle age / energy / rate / duration values → clipped.
* Exact duplicate rows → removed (none exist in this dataset).
* Identifier and timestamp columns (`User ID`, `Charging Station ID`,
  start/end times) → dropped (they cannot predict a new session).

## Features

**Base inputs** (as recorded): battery capacity, energy, duration, rate, state
of charge start/end, distance, temperature, vehicle age plus the categorical
context columns.

**Derived features** (`src/features.py`) — physically supported by the data:

* `SOC Gain (%)` – battery percentage added during the session.
* `Energy per km (kWh/km)` – consumption per kilometre.
* `Potential Energy (kWh)` – the energy the session *should* have added
  (`capacity × SOC gain / 100`).

No derived feature uses the target, so there is **no leakage**.

## ML approach

1. Load → clean → engineer features (`src/preprocessing.py`, `src/features.py`).
2. Split 80 / 20 train / hold-out test set (fixed seed).
3. Build a scikit-learn `Pipeline`:
   * numeric → median imputation → standard scaling,
   * categorical → one-hot encoding (`handle_unknown='ignore'`).
4. **5-fold cross-validation** (shuffled) on the training set for every model.
5. Select the best model by cross-validated R² (ties → the simpler model),
   refit on all data, and report honest hold-out test metrics.
6. Save model + preprocessor + metrics for the dashboard.

## Models tested

* Linear Regression
* Ridge Regression (α = 1.0)
* Random Forest (200 trees)
* Extra Trees (200 trees)
* Gradient Boosting

## Evaluation metrics

R², MAE, RMSE (cross-validated), plus the naive baseline: *always predict the
average charging cost* (MAE ≈ 9.16).

### Results (5-fold cross-validation)

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| **Ridge Regression** | **−0.034** | **9.26** | **10.79** |
| Linear Regression | −0.034 | 9.26 | 10.79 |
| Random Forest | −0.049 | 9.34 | 10.86 |
| Gradient Boosting | −0.082 | 9.39 | 11.03 |
| Extra Trees | −0.076 | 9.45 | 11.01 |

**Hold-out test set (20 %):** R² = **+0.053**, MAE = **9.15**, RMSE = **10.90**.

### Best-performing model

**Ridge Regression** was selected — it had the best cross-validated score, and
being a linear model it is simple, fast and robust.

### ⚠️ Important honest finding

**Every model scores R² ≤ 0 in cross-validation — essentially no better than
guessing the average cost.** The dataset appears synthetic: `Charging Cost`
correlates ≈ 0.00 with every feature (e.g. −0.008 with `Energy Consumed`),
190 rows record more energy than the battery could hold, and 32 rows show a
State-of-Charge above 100 %. **The dataset contains almost no usable signal.**

This project therefore does **not** claim a meaningful accuracy figure. The
dashboard shows the real metrics and labels the forecast as *illustrative*,
not reliable. Model confidence percentages are intentionally **not** invented;
instead the typical error (hold-out MAE) is displayed.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/5ubodh/ev-patterns.git
cd ev-patterns

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## How to train the model

```bash
python -m src.train          # trains + saves model + comparison to models/
python -m src.interpret      # computes feature importance (optional, richer app tab)
python -m src.evaluate       # prints the stored evaluation report
```

Artifacts written to `models/`:

* `model.joblib` – full scikit-learn pipeline (preprocessing + model)
* `preprocessor.joblib` – the fitted preprocessor alone
* `metadata.json` – target, features, cross-validation scores, hold-out metrics
* `model_comparison.csv` – the model comparison table

## How to run the application

```bash
streamlit run app/app.py
```

Open <http://localhost:8501>. The dashboard has four tabs:

1. **Predict** – enter session details → estimated cost, unit, input summary,
   plain-language explanation and typical error.
2. **Model performance** – cross-validated comparison + final hold-out metrics.
3. **Feature importance** – permutation importance of each feature.
4. **Dataset insights** – distributions, summary statistics and correlations.

The dashboard **loads the saved model**; it never retrains on startup.

## Example prediction

| Input | Value |
|---|---|
| Battery capacity | 75.0 kWh |
| Energy delivered | 40.0 kWh |
| Charging duration | 2.5 h |
| Charging rate | 30.0 kW |
| Battery start / end | 30 % → 85 % |
| Distance since last charge | 150 km |
| Temperature | 20 °C |
| Vehicle age | 4 years |
| Vehicle / city / charger | Chevy Bolt / Los Angeles / Level 2 |

```
Estimated cost     : 22.18 USD
Typical error*     : ± 9.15 USD
Hold-out R²        : +0.053   →  Forecast quality: Not reliable
```

\*Mean absolute error on the hold-out test set — an average distance, not an
official confidence interval.

## Project structure

```
ev-patterns/
├── data/
│   └── ev_charging_patterns.csv      # dataset
├── analysis/
│   └── ev_charging_analysis.ipynb    # exploratory notebook (original)
├── src/
│   ├── config.py                     # paths, target, features, settings
│   ├── data_loader.py                # load dataset (robust path resolution)
│   ├── preprocessing.py              # cleaning: dtypes, duplicates, clamping
│   ├── features.py                   # derived EV features
│   ├── train.py                      # CV model comparison + save artifacts
│   ├── evaluate.py                   # print / reuse stored results
│   ├── interpret.py                  # permutation importance + coefficients
│   └── predict.py                    # validated prediction + explanation
├── models/                           # generated: model.joblib, metadata.json …
├── app/
│   └── app.py                        # Streamlit dashboard
├── tests/
│   └── test_pipeline.py              # smoke tests (python tests/test_pipeline.py)
├── requirements.txt
├── README.md
└── .gitignore
```

## Running the tests

```bash
python tests/test_pipeline.py
```

Checks that artifacts exist, predictions return sensible values, and invalid
inputs (negative battery, SOC > 100 %, end SOC < start SOC, non-numeric text)
are rejected with clear messages.

## Limitations

* The dataset shows essentially **no relationship** between features and
  charging cost, so no accurate model is possible with it (see honest finding
  above).
* It is a single synthetic dataset — no real-world generalisation is claimed.
* Median imputation is computed on the full dataset (tiny leakage of 66 rows;
  negligible, kept for beginner readability).
* No price/tariff information exists in the data, which is normally the main
  driver of charging cost.

## Future improvements

* Add real charging-session data with tariffs (time-of-use pricing, network
  fees) — cost prediction needs actual pricing inputs.
* Add confidence intervals via quantile regression once signal exists.
* Retrain automatically when new data lands (MLflow or a scheduled script).
* Cluster sessions by charging behaviour for demand forecasting.

---

**Stack:** Python · pandas · scikit-learn · joblib · Streamlit · matplotlib