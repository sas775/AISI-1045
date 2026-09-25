# AISI-1045
Code 
# ================================================================
# CAUSALITY-GUIDED EXPLAINABLE ML + MULTI-OBJECTIVE OPTIMIZATION
# AISI 1045 Medium Carbon Steel
#
# Decision variables:
#   Vc, Feed, Depth
#
# Endogenous process states:
#   Time, Temperature, Tool Wear
#
# Responses:
#   Ra, Rz
#
# ================================================================


# ================================================================
# SECTION 1 — INSTALL REQUIRED PACKAGES
# ================================================================

!pip -q install xgboost shap pymoo openpyxl scikit-learn seaborn


# ================================================================
# SECTION 2 — IMPORT LIBRARIES
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor

import shap

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries loaded successfully.")


# ================================================================
# SECTION 3 — UPLOAD EXCEL DATA
# ================================================================

uploaded = files.upload()

filename = list(uploaded.keys())[0]

df = pd.read_excel(filename)

print("\nFile loaded:", filename)
print("Dataset shape:", df.shape)

display(df.head())


# ================================================================
# SECTION 4 — BASIC DATA CLEANING
# ================================================================

# Remove completely empty rows
df = df.dropna(how="all").copy()

# Standardize column names
df.columns = [str(c).strip() for c in df.columns]

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# Required columns
required_columns = [
    "Vc",
    "Feed",
    "Depth",
    "Time",
    "Tool Wear",
    "Ra",
    "Rz",
    "Temperature"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if len(missing_columns) > 0:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert numerical columns
for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove rows with missing values
df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print("\nFinal dataset shape:", df.shape)


# ================================================================
# SECTION 5 — DESCRIPTIVE STATISTICS
# ================================================================

print("\n================ DESCRIPTIVE STATISTICS ================\n")

display(
    df[required_columns].describe().T
)


# ================================================================
# SECTION 6 — DEFINE VARIABLES
# ================================================================

# ------------------------------------------------
# Controllable machining decision variables
# ------------------------------------------------

decision_vars = [
    "Vc",
    "Feed",
    "Depth"
]

# ------------------------------------------------
# Endogenous process variables
# ------------------------------------------------

state_vars = [
    "Time",
    "Temperature",
    "Tool Wear"
]

# ------------------------------------------------
# Final quality responses
# ------------------------------------------------

response_vars = [
    "Ra",
    "Rz"
]

print("Decision variables:", decision_vars)
print("State variables:", state_vars)
print("Responses:", response_vars)


# ================================================================
# SECTION 7 — DATA RANGE CHECK
# ================================================================

print("\n================ VARIABLE RANGES ================\n")

for col in required_columns:
    print(
        f"{col:15s}: "
        f"{df[col].min():.6f} "
        f"to "
        f"{df[col].max():.6f}"
    )


# ================================================================
# SECTION 8 — CORRELATION MATRIX
# ================================================================

plt.figure(figsize=(11, 8))

corr = df[required_columns].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Machining Variables")
plt.tight_layout()
plt.show()


# ================================================================
# SECTION 9 — CAUSAL STRUCTURE
# ================================================================

"""
Proposed operational causal structure:

Vc, Feed, Depth
       ↓
      Time
       ↓
Temperature
       ↓
Tool Wear
       ↓
Ra
       ↓
Rz

Additional direct pathways are retained where supported by
the data-generating logic:

Vc       → Temperature
Depth    → Temperature
Vc       → Tool Wear
Depth    → Tool Wear
Time     → Tool Wear
Vc       → Ra
Feed     → Ra
Tool Wear → Ra
Ra       → Rz
Vc       → Rz
Tool Wear → Rz

IMPORTANT:
For the final manuscript, this graph must be made identical
to the actual synthetic data-generating equations.
"""

causal_edges = [
    ("Vc", "Time"),
    ("Feed", "Time"),
    ("Depth", "Time"),

    ("Vc", "Temperature"),
    ("Feed", "Temperature"),
    ("Depth", "Temperature"),
    ("Time", "Temperature"),

    ("Vc", "Tool Wear"),
    ("Depth", "Tool Wear"),
    ("Time", "Tool Wear"),
    ("Temperature", "Tool Wear"),

    ("Vc", "Ra"),
    ("Feed", "Ra"),
    ("Tool Wear", "Ra"),

    ("Ra", "Rz"),
    ("Vc", "Rz"),
    ("Tool Wear", "Rz")
]

print("\nProposed causal edges:")
for source, target in causal_edges:
    print(f"{source} -> {target}")


# ================================================================
# SECTION 10 — FUNCTION FOR MODEL EVALUATION
# ================================================================

def evaluate_model(model, X, y, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    results = {
        "Model": name,
        "Train R2": r2_score(y_train, pred_train),
        "Test R2": r2_score(y_test, pred_test),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, pred_test)
        ),
        "Test MAE": mean_absolute_error(
            y_test,
            pred_test
        )
    }

    return results, model


# ================================================================
# SECTION 11 — MODEL FACTORY
# ================================================================

def create_xgb():

    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_rf():

    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ================================================================
# SECTION 12 — STRUCTURAL MODEL DEFINITIONS
# ================================================================

# ------------------------------------------------
# TIME MODEL
# ------------------------------------------------

X_time = df[
    ["Vc", "Feed", "Depth"]
]

y_time = df["Time"]


# ------------------------------------------------
# TEMPERATURE MODEL
# ------------------------------------------------

X_temperature = df[
    ["Vc", "Feed", "Depth", "Time"]
]

y_temperature = df["Temperature"]


# ------------------------------------------------
# TOOL WEAR MODEL
# ------------------------------------------------

X_wear = df[
    [
        "Vc",
        "Depth",
        "Time",
        "Temperature"
    ]
]

y_wear = df["Tool Wear"]


# ------------------------------------------------
# Ra MODEL
# ------------------------------------------------

X_ra = df[
    [
        "Vc",
        "Feed",
        "Tool Wear"
    ]
]

y_ra = df["Ra"]


# ------------------------------------------------
# Rz MODEL
# ------------------------------------------------

X_rz = df[
    [
        "Ra",
        "Vc",
        "Tool Wear"
    ]
]

y_rz = df["Rz"]


# ================================================================
# SECTION 13 — TRAIN STRUCTURAL XGBOOST MODELS
# ================================================================

time_model = create_xgb()

temperature_model = create_xgb()

wear_model = create_xgb()

ra_model = create_xgb()

rz_model = create_xgb()


time_model.fit(
    X_time,
    y_time
)

temperature_model.fit(
    X_temperature,
    y_temperature
)

wear_model.fit(
    X_wear,
    y_wear
)

ra_model.fit(
    X_ra,
    y_ra
)

rz_model.fit(
    X_rz,
    y_rz
)

print("All structural XGBoost models trained successfully.")


# ================================================================
# SECTION 14 — 10-FOLD CROSS VALIDATION
# ================================================================

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


def cv_model(model, X, y):

    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2",
        n_jobs=-1
    )

    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
    )

    return {
        "Mean R2": np.mean(r2_scores),
        "SD R2": np.std(r2_scores),
        "Mean RMSE": np.mean(rmse_scores),
        "SD RMSE": np.std(rmse_scores)
    }


cv_results = []

for name, model, X, y in [

    ("Time", time_model, X_time, y_time),

    (
        "Temperature",
        temperature_model,
        X_temperature,
        y_temperature
    ),

    (
        "Tool Wear",
        wear_model,
        X_wear,
        y_wear
    ),

    ("Ra", ra_model, X_ra, y_ra),

    ("Rz", rz_model, X_rz, y_rz)
]:

    result = cv_model(
        model,
        X,
        y
    )

    result["Target"] = name

    cv_results.append(result)


cv_results_df = pd.DataFrame(cv_results)

print("\n================ 10-FOLD CV RESULTS ================\n")

display(cv_results_df)


# ================================================================
# SECTION 15 — BASELINE MODEL COMPARISON FOR Ra
# ================================================================

models_ra = {

    "Linear Regression":
        LinearRegression(),

    "Polynomial Regression":
        Pipeline([
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )
            ),
            (
                "linear",
                LinearRegression()
            )
        ]),

    "Random Forest":
        create_rf(),

    "XGBoost":
        create_xgb()
}


baseline_results = []

for name, model in models_ra.items():

    result, fitted_model = evaluate_model(
        model,
        X_ra,
        y_ra,
        name
    )

    baseline_results.append(result)


baseline_results_df = pd.DataFrame(
    baseline_results
)

print("\n================ Ra MODEL COMPARISON ================\n")

display(
    baseline_results_df.sort_values(
        "Test R2",
        ascending=False
    )
)


# ================================================================
# SECTION 16 — BASELINE MODEL COMPARISON FOR Rz
# ================================================================

models_rz = {

    "Linear Regression":
        LinearRegression(),

    "Polynomial Regression":
        Pipeline([
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )
            ),
            (
                "linear",
                LinearRegression()
            )
        ]),

    "Random Forest":
        create_rf(),

    "XGBoost":
        create_xgb()
}


baseline_rz_results = []

for name, model in models_rz.items():

    result, fitted_model = evaluate_model(
        model,
        X_rz,
        y_rz,
        name
    )

    baseline_rz_results.append(result)


baseline_rz_results_df = pd.DataFrame(
    baseline_rz_results
)

print("\n================ Rz MODEL COMPARISON ================\n")

display(
    baseline_rz_results_df.sort_values(
        "Test R2",
        ascending=False
    )
)


# ================================================================
# SECTION 17 — PREDICTED VS ACTUAL: Ra
# ================================================================

ra_pred = ra_model.predict(X_ra)

plt.figure(figsize=(7, 6))

plt.scatter(
    y_ra,
    ra_pred,
    alpha=0.75
)

min_val = min(
    y_ra.min(),
    ra_pred.min()
)

max_val = max(
    y_ra.max(),
    ra_pred.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Ra (µm)")
plt.ylabel("Predicted Ra (µm)")
plt.title("Predicted vs Actual Surface Roughness Ra")

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 18 — PREDICTED VS ACTUAL: Rz
# ================================================================

rz_pred = rz_model.predict(X_rz)

plt.figure(figsize=(7, 6))

plt.scatter(
    y_rz,
    rz_pred,
    alpha=0.75
)

min_val = min(
    y_rz.min(),
    rz_pred.min()
)

max_val = max(
    y_rz.max(),
    rz_pred.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Rz (µm)")
plt.ylabel("Predicted Rz (µm)")
plt.title("Predicted vs Actual Rz")

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 19 — RESIDUAL ANALYSIS: Ra
# ================================================================

ra_residuals = y_ra - ra_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    ra_pred,
    ra_residuals,
    alpha=0.75
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Ra (µm)")
plt.ylabel("Residual")
plt.title("Residual Analysis for Ra")

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 20 — RESIDUAL ANALYSIS: Rz
# ================================================================

rz_residuals = y_rz - rz_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    rz_pred,
    rz_residuals,
    alpha=0.75
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Rz (µm)")
plt.ylabel("Residual")
plt.title("Residual Analysis for Rz")

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 21 — SHAP ANALYSIS FOR Ra
# ================================================================

explainer_ra = shap.TreeExplainer(
    ra_model
)

shap_values_ra = explainer_ra(
    X_ra
)

plt.figure()

shap.summary_plot(
    shap_values_ra,
    X_ra,
    show=False
)

plt.title(
    "SHAP Summary Plot for Ra"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 22 — SHAP BAR IMPORTANCE FOR Ra
# ================================================================

plt.figure()

shap.summary_plot(
    shap_values_ra,
    X_ra,
    plot_type="bar",
    show=False
)

plt.title(
    "SHAP Feature Importance for Ra"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 23 — SHAP DEPENDENCE PLOTS
# ================================================================

for feature in X_ra.columns:

    shap.dependence_plot(
        feature,
        shap_values_ra.values,
        X_ra,
        interaction_index=None,
        show=False
    )

    plt.title(
        f"SHAP Dependence Plot: {feature} → Ra"
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# SECTION 24 — SHAP INTERACTION VALUES
# ================================================================

interaction_values_ra = explainer_ra.shap_interaction_values(
    X_ra
)

interaction_mean_ra = np.abs(
    interaction_values_ra
).mean(axis=0)

interaction_df_ra = pd.DataFrame(
    interaction_mean_ra,
    index=X_ra.columns,
    columns=X_ra.columns
)

print(
    "\n================ SHAP INTERACTION MATRIX: Ra ================\n"
)

display(
    interaction_df_ra
)


# ================================================================
# SECTION 25 — SHAP INTERACTION HEATMAP
# ================================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    interaction_df_ra,
    annot=True,
    fmt=".4f",
    cmap="viridis"
)

plt.title(
    "Mean Absolute SHAP Interaction Values for Ra"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 26 — SHAP ANALYSIS FOR Rz
# ================================================================

explainer_rz = shap.TreeExplainer(
    rz_model
)

shap_values_rz = explainer_rz(
    X_rz
)

plt.figure()

shap.summary_plot(
    shap_values_rz,
    X_rz,
    show=False
)

plt.title(
    "SHAP Summary Plot for Rz"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 27 — FEED-SPECIFIC SHAP ANALYSIS
# ================================================================

feed_shap = pd.DataFrame({
    "Feed": df["Feed"].values,
    "SHAP_Feed": shap_values_ra.values[
        :,
        X_ra.columns.get_loc("Feed")
    ],
    "Ra": df["Ra"].values
})

print(
    "\n================ FEED SHAP RELATIONSHIP ================\n"
)

display(
    feed_shap.head(10)
)


plt.figure(figsize=(8, 6))

plt.scatter(
    feed_shap["Feed"],
    feed_shap["SHAP_Feed"],
    alpha=0.75
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Feed")
plt.ylabel("SHAP value for Feed")
plt.title(
    "SHAP Dependence of Feed on Predicted Ra"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 28 — FEED VS Ra DIRECT RELATIONSHIP
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Feed"],
    df["Ra"],
    alpha=0.75
)

plt.xlabel("Feed")
plt.ylabel("Ra (µm)")
plt.title(
    "Observed Feed–Ra Relationship"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 29 — CAUSAL / CORRELATION DIAGNOSTIC
# ================================================================

causal_diagnostic_columns = [
    "Vc",
    "Feed",
    "Depth",
    "Time",
    "Temperature",
    "Tool Wear",
    "Ra",
    "Rz"
]

diagnostic_corr = df[
    causal_diagnostic_columns
].corr()

print(
    "\n================ CORRELATION DIAGNOSTIC ================\n"
)

display(
    diagnostic_corr
)


# ================================================================
# SECTION 30 — SEQUENTIAL CAUSAL INTERVENTION FUNCTION
# ================================================================

"""
The function below performs an intervention on the three
controllable variables:

    do(Vc, Feed, Depth)

The remaining process variables are generated sequentially.

This prevents Temperature and Tool Wear from being treated
as independent decision variables in NSGA-II.
"""


def causal_intervention_predict(
    Vc,
    Feed,
    Depth
):

    # ------------------------------------------------
    # STEP 1 — Time
    # ------------------------------------------------

    time_input = pd.DataFrame({
        "Vc": [Vc],
        "Feed": [Feed],
        "Depth": [Depth]
    })

    predicted_time = time_model.predict(
        time_input
    )[0]

    # ------------------------------------------------
    # STEP 2 — Temperature
    # ------------------------------------------------

    temperature_input = pd.DataFrame({
        "Vc": [Vc],
        "Feed": [Feed],
        "Depth": [Depth],
        "Time": [predicted_time]
    })

    predicted_temperature = (
        temperature_model.predict(
            temperature_input
        )[0]
    )

    # ------------------------------------------------
    # STEP 3 — Tool Wear
    # ------------------------------------------------

    wear_input = pd.DataFrame({
        "Vc": [Vc],
        "Depth": [Depth],
        "Time": [predicted_time],
        "Temperature": [predicted_temperature]
    })

    predicted_wear = (
        wear_model.predict(
            wear_input
        )[0]
    )

    # ------------------------------------------------
    # STEP 4 — Ra
    # ------------------------------------------------

    ra_input = pd.DataFrame({
        "Vc": [Vc],
        "Feed": [Feed],
        "Tool Wear": [predicted_wear]
    })

    predicted_ra = (
        ra_model.predict(
            ra_input
        )[0]
    )

    # ------------------------------------------------
    # STEP 5 — Rz
    # ------------------------------------------------

    rz_input = pd.DataFrame({
        "Ra": [predicted_ra],
        "Vc": [Vc],
        "Tool Wear": [predicted_wear]
    })

    predicted_rz = (
        rz_model.predict(
            rz_input
        )[0]
    )

    return pd.DataFrame({
        "Vc": [Vc],
        "Feed": [Feed],
        "Depth": [Depth],
        "Time": [predicted_time],
        "Temperature": [predicted_temperature],
        "Tool Wear": [predicted_wear],
        "Ra": [predicted_ra],
        "Rz": [predicted_rz]
    })


# ================================================================
# SECTION 31 — TEST CAUSAL INTERVENTION
# ================================================================

test_Vc = df["Vc"].median()
test_Feed = df["Feed"].median()
test_Depth = df["Depth"].median()

test_intervention = causal_intervention_predict(
    test_Vc,
    test_Feed,
    test_Depth
)

print(
    "\n================ TEST CAUSAL INTERVENTION ================\n"
)

display(
    test_intervention
)


# ================================================================
# SECTION 32 — PHYSICAL / OBSERVED STATE LIMITS
# ================================================================

state_limits = {

    "Time": (
        df["Time"].min(),
        df["Time"].max()
    ),

    "Temperature": (
        df["Temperature"].min(),
        df["Temperature"].max()
    ),

    "Tool Wear": (
        df["Tool Wear"].min(),
        df["Tool Wear"].max()
    )
}

print(
    "\n================ STATE LIMITS ================\n"
)

for variable, limits in state_limits.items():

    print(
        f"{variable}: "
        f"{limits[0]:.6f} – {limits[1]:.6f}"
    )


# ================================================================
# SECTION 33 — DECISION VARIABLE BOUNDS
# ================================================================

xl = np.array([
    df["Vc"].min(),
    df["Feed"].min(),
    df["Depth"].min()
])

xu = np.array([
    df["Vc"].max(),
    df["Feed"].max(),
    df["Depth"].max()
])

print("\nDecision-variable lower bounds:")
print(xl)

print("\nDecision-variable upper bounds:")
print(xu)


# ================================================================
# SECTION 34 — CAUSAL NSGA-II PROBLEM
# ================================================================

class CausalMachiningProblem(Problem):

    def __init__(self):

        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=3,
            xl=xl,
            xu=xu
        )

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs
    ):

        objectives = []
        constraints = []

        for i in range(len(x)):

            Vc = x[i, 0]
            Feed = x[i, 1]
            Depth = x[i, 2]

            result = causal_intervention_predict(
                Vc,
                Feed,
                Depth
            )

            predicted_time = (
                result["Time"].iloc[0]
            )

            predicted_temperature = (
                result["Temperature"].iloc[0]
            )

            predicted_wear = (
                result["Tool Wear"].iloc[0]
            )

            predicted_ra = (
                result["Ra"].iloc[0]
            )

            predicted_rz = (
                result["Rz"].iloc[0]
            )

            # ------------------------------------------------
            # Objectives
            # ------------------------------------------------

            objectives.append([
                predicted_ra,
                predicted_rz
            ])

            # ------------------------------------------------
            # Constraints
            #
            # g <= 0 is feasible in pymoo
            # ------------------------------------------------

            time_min, time_max = state_limits["Time"]

            temp_min, temp_max = state_limits[
                "Temperature"
            ]

            wear_min, wear_max = state_limits[
                "Tool Wear"
            ]

            g_time_low = (
                time_min - predicted_time
            )

            g_time_high = (
                predicted_time - time_max
            )

            g_wear_low = (
                wear_min - predicted_wear
            )

            g_wear_high = (
                predicted_wear - wear_max
            )

            g_temp_low = (
                temp_min - predicted_temperature
            )

            g_temp_high = (
                predicted_temperature - temp_max
            )

            # Combine each pair into a single constraint
            constraints.append([
                max(
                    g_time_low,
                    g_time_high
                ),

                max(
                    g_temp_low,
                    g_temp_high
                ),

                max(
                    g_wear_low,
                    g_wear_high
                )
            ])

        out["F"] = np.array(
            objectives
        )

        out["G"] = np.array(
            constraints
        )


# ================================================================
# SECTION 35 — RUN NSGA-II
# ================================================================

problem = CausalMachiningProblem()

algorithm = NSGA2(
    pop_size=100
)

print(
    "\nRunning causal NSGA-II..."
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 100),
    seed=RANDOM_STATE,
    save_history=True,
    verbose=True
)

print(
    "\nNSGA-II optimization completed."
)


# ================================================================
# SECTION 36 — EXTRACT PARETO SOLUTIONS
# ================================================================

pareto_decisions = res.X

pareto_objectives = res.F

pareto_df = pd.DataFrame(
    pareto_decisions,
    columns=[
        "Vc",
        "Feed",
        "Depth"
    ]
)

pareto_df["Ra"] = pareto_objectives[:, 0]

pareto_df["Rz"] = pareto_objectives[:, 1]


# ================================================================
# SECTION 37 — REMOVE DUPLICATE SOLUTIONS
# ================================================================

# Round numerical values before duplicate checking
duplicate_check = pareto_df[
    [
        "Vc",
        "Feed",
        "Depth",
        "Ra",
        "Rz"
    ]
].round(6)

unique_mask = ~duplicate_check.duplicated()

pareto_unique_df = pareto_df[
    unique_mask
].reset_index(drop=True)

print(
    "\nOriginal Pareto solutions:",
    len(pareto_df)
)

print(
    "Unique Pareto solutions:",
    len(pareto_unique_df)
)

display(
    pareto_unique_df.head(20)
)


# ================================================================
# SECTION 38 — PREDICT ENDOGENOUS STATES FOR PARETO SOLUTIONS
# ================================================================

pareto_results = []

for _, row in pareto_unique_df.iterrows():

    result = causal_intervention_predict(
        row["Vc"],
        row["Feed"],
        row["Depth"]
    )

    pareto_results.append(
        result.iloc[0].to_dict()
    )


pareto_full_df = pd.DataFrame(
    pareto_results
)

print(
    "\n================ CAUSALLY RECONSTRUCTED PARETO SET ================\n"
)

display(
    pareto_full_df.head(20)
)


# ================================================================
# SECTION 39 — PARETO FRONT
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    pareto_full_df["Ra"],
    pareto_full_df["Rz"],
    s=45,
    alpha=0.8
)

plt.xlabel("Surface roughness Ra (µm)")
plt.ylabel("Maximum roughness height Rz (µm)")
plt.title(
    "Causality-Guided NSGA-II Pareto Front"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 40 — PARETO RANGE
# ================================================================

print(
    "\n================ PARETO RANGE ================\n"
)

for col in [
    "Ra",
    "Rz",
    "Vc",
    "Feed",
    "Depth",
    "Time",
    "Temperature",
    "Tool Wear"
]:

    print(
        f"{col:15s}: "
        f"{pareto_full_df[col].min():.6f} "
        f"to "
        f"{pareto_full_df[col].max():.6f}"
    )


# ================================================================
# SECTION 41 — MODEL ERROR VS PARETO RANGE
# ================================================================

ra_rmse = np.sqrt(
    mean_squared_error(
        y_ra,
        ra_pred
    )
)

rz_rmse = np.sqrt(
    mean_squared_error(
        y_rz,
        rz_pred
    )
)

ra_pareto_range = (
    pareto_full_df["Ra"].max()
    -
    pareto_full_df["Ra"].min()
)

rz_pareto_range = (
    pareto_full_df["Rz"].max()
    -
    pareto_full_df["Rz"].min()
)

print(
    "\n================ PARETO RANGE VS MODEL ERROR ================\n"
)

print(
    f"Ra RMSE: {ra_rmse:.6f}"
)

print(
    f"Ra Pareto range: {ra_pareto_range:.6f}"
)

print(
    f"Rz RMSE: {rz_rmse:.6f}"
)

print(
    f"Rz Pareto range: {rz_pareto_range:.6f}"
)

print(
    "\nRa range / RMSE:",
    ra_pareto_range / ra_rmse
)

print(
    "Rz range / RMSE:",
    rz_pareto_range / rz_rmse
)


# ================================================================
# SECTION 42 — BOOTSTRAP XGBOOST MODELS
# ================================================================

"""
Bootstrap uncertainty is generated for the Ra and Rz response
models.

The bootstrap samples preserve the original sample size and
resample observations with replacement.
"""

N_BOOTSTRAP = 100

bootstrap_ra_models = []
bootstrap_rz_models = []

rng = np.random.default_rng(
    RANDOM_STATE
)

for b in range(N_BOOTSTRAP):

    sample_indices = rng.choice(
        len(df),
        size=len(df),
        replace=True
    )

    boot_df = df.iloc[
        sample_indices
    ].copy()

    model_ra_b = create_xgb()

    model_rz_b = create_xgb()

    model_ra_b.fit(
        boot_df[
            [
                "Vc",
                "Feed",
                "Tool Wear"
            ]
        ],
        boot_df["Ra"]
    )

    model_rz_b.fit(
        boot_df[
            [
                "Ra",
                "Vc",
                "Tool Wear"
            ]
        ],
        boot_df["Rz"]
    )

    bootstrap_ra_models.append(
        model_ra_b
    )

    bootstrap_rz_models.append(
        model_rz_b
    )

print(
    f"{N_BOOTSTRAP} bootstrap model pairs generated."
)


# ================================================================
# SECTION 43 — BOOTSTRAP PREDICTION FUNCTION
# ================================================================

def bootstrap_predict_response(
    Vc,
    Feed,
    Depth,
    n_models=None
):

    if n_models is None:
        n_models = N_BOOTSTRAP

    # First obtain causal state variables
    causal_result = causal_intervention_predict(
        Vc,
        Feed,
        Depth
    )

    predicted_time = (
        causal_result["Time"].iloc[0]
    )

    predicted_temperature = (
        causal_result["Temperature"].iloc[0]
    )

    predicted_wear = (
        causal_result["Tool Wear"].iloc[0]
    )

    ra_predictions = []
    rz_predictions = []

    for i in range(
        min(
            n_models,
            len(bootstrap_ra_models)
        )
    ):

        # Ra
        ra_input = pd.DataFrame({
            "Vc": [Vc],
            "Feed": [Feed],
            "Tool Wear": [predicted_wear]
        })

        ra_value = (
            bootstrap_ra_models[i]
            .predict(ra_input)[0]
        )

        # Rz
        rz_input = pd.DataFrame({
            "Ra": [ra_value],
            "Vc": [Vc],
            "Tool Wear": [predicted_wear]
        })

        rz_value = (
            bootstrap_rz_models[i]
            .predict(rz_input)[0]
        )

        ra_predictions.append(
            ra_value
        )

        rz_predictions.append(
            rz_value
        )

    return {
        "Time": predicted_time,
        "Temperature": predicted_temperature,
        "Tool Wear": predicted_wear,
        "Ra_mean": np.mean(
            ra_predictions
        ),
        "Ra_std": np.std(
            ra_predictions
        ),
        "Ra_lower": np.percentile(
            ra_predictions,
            2.5
        ),
        "Ra_upper": np.percentile(
            ra_predictions,
            97.5
        ),
        "Rz_mean": np.mean(
            rz_predictions
        ),
        "Rz_std": np.std(
            rz_predictions
        ),
        "Rz_lower": np.percentile(
            rz_predictions,
            2.5
        ),
        "Rz_upper": np.percentile(
            rz_predictions,
            97.5
        )
    }


# ================================================================
# SECTION 44 — PROPAGATE UNCERTAINTY TO PARETO SOLUTIONS
# ================================================================
uncertainty_results = []

for _, row in pareto_full_df.iterrows():
    uncertainty = bootstrap_predict_response(
        row["Vc"], row["Feed"], row["Depth"]
    )
    uncertainty_results.append(uncertainty)

uncertainty_df = pd.DataFrame(uncertainty_results)

# --- FIX: drop duplicated state columns before concat ---
uncertainty_df = uncertainty_df.drop(
    columns=["Time", "Temperature", "Tool Wear"],
    errors="ignore"
)

pareto_uncertainty_df = pd.concat(
    [
        pareto_full_df.reset_index(drop=True),
        uncertainty_df.reset_index(drop=True),
    ],
    axis=1,
)

# Sanity check: no duplicate column names
assert not pareto_uncertainty_df.columns.duplicated().any(), \
    pareto_uncertainty_df.columns[pareto_uncertainty_df.columns.duplicated()].tolist()

print("\n================ PARETO UNCERTAINTY ================\n")
display(pareto_uncertainty_df.head(20))

# ================================================================
# SECTION 45 — UNCERTAINTY PLOT FOR Ra
# ================================================================

plt.figure(figsize=(9, 6))

x_axis = np.arange(
    len(pareto_uncertainty_df)
)

plt.errorbar(
    x_axis,
    pareto_uncertainty_df["Ra_mean"],
    yerr=[
        pareto_uncertainty_df["Ra_mean"]
        -
        pareto_uncertainty_df["Ra_lower"],

        pareto_uncertainty_df["Ra_upper"]
        -
        pareto_uncertainty_df["Ra_mean"]
    ],
    fmt="o",
    capsize=3
)

plt.xlabel(
    "Pareto solution index"
)

plt.ylabel(
    "Predicted Ra (µm)"
)

plt.title(
    "Bootstrap Prediction Uncertainty of Pareto Solutions"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 46 — UNCERTAINTY PLOT FOR Rz
# ================================================================

plt.figure(figsize=(9, 6))

plt.errorbar(
    x_axis,
    pareto_uncertainty_df["Rz_mean"],
    yerr=[
        pareto_uncertainty_df["Rz_mean"]
        -
        pareto_uncertainty_df["Rz_lower"],

        pareto_uncertainty_df["Rz_upper"]
        -
        pareto_uncertainty_df["Rz_mean"]
    ],
    fmt="o",
    capsize=3
)

plt.xlabel(
    "Pareto solution index"
)

plt.ylabel(
    "Predicted Rz (µm)"
)

plt.title(
    "Bootstrap Prediction Uncertainty of Pareto Solutions"
)

plt.tight_layout()
plt.show()


# ================================================================
# SECTION 47 — TOPSIS FUNCTION
# ================================================================

def topsis(
    data,
    weights
):

    """
    TOPSIS for two cost-type objectives:
        Ra → minimize
        Rz → minimize
    """

    matrix = data[
        ["Ra", "Rz"]
    ].values.astype(float)

    weights = np.array(
        weights,
        dtype=float
    )

    # Normalize weights
    weights = (
        weights /
        weights.sum()
    )

    # Vector normalization
    denominator = np.sqrt(
        (matrix ** 2).sum(axis=0)
    )

    normalized_matrix = (
        matrix /
        denominator
    )

    # Weighted normalized matrix
    weighted_matrix = (
        normalized_matrix *
        weights
    )

    # Both objectives are cost criteria
    ideal_best = (
        weighted_matrix.min(axis=0)
    )

    ideal_worst = (
        weighted_matrix.max(axis=0)
    )

    distance_best = np.sqrt(
        (
            (
                weighted_matrix
                -
                ideal_best
            ) ** 2
        ).sum(axis=1)
    )

    distance_worst = np.sqrt(
        (
            (
                weighted_matrix
                -
                ideal_worst
            ) ** 2
        ).sum(axis=1)
    )

    closeness = (
        distance_worst /
        (
            distance_best
            +
            distance_worst
            +
            1e-12
        )
    )

    ranking = (
        pd.Series(
            closeness
        )
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    return closeness, ranking


# ================================================================
# SECTION 48 — TOPSIS WEIGHT SENSITIVITY
# ================================================================

weight_scenarios = [

    (0.20, 0.80),
    (0.30, 0.70),
    (0.40, 0.60),
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20)
]

topsis_results = []

for w_ra, w_rz in weight_scenarios:

    closeness, ranking = topsis(
        pareto_uncertainty_df,
        [w_ra, w_rz]
    )

    best_index = np.argmax(
        closeness
    )

    topsis_results.append({

        "Ra Weight": w_ra,

        "Rz Weight": w_rz,

        "Best Solution Index":
            best_index,

        "Best Vc":
            pareto_uncertainty_df.loc[
                best_index,
                "Vc"
            ],

        "Best Feed":
            pareto_uncertainty_df.loc[
                best_index,
                "Feed"
            ],

        "Best Depth":
            pareto_uncertainty_df.loc[
                best_index,
                "Depth"
            ],

        "Best Ra":
            pareto_uncertainty_df.loc[
                best_index,
                "Ra"
            ],

        "Best Rz":
            pareto_uncertainty_df.loc[
                best_index,
                "Rz"
            ]
    })


topsis_sensitivity_df = pd.DataFrame(
    topsis_results
)

print(
    "\n================ TOPSIS WEIGHT SENSITIVITY ================\n"
)

display(
    topsis_sensitivity_df
)


# ================================================================
# SECTION 49 — TOPSIS RANK STABILITY
# ================================================================

rank_matrix = []

for w_ra, w_rz in weight_scenarios:

    closeness, ranking = topsis(
        pareto_uncertainty_df,
        [w_ra, w_rz]
    )

    rank_matrix.append(
        ranking.values
    )


rank_matrix = np.array(
    rank_matrix
)

rank_stability_df = pd.DataFrame(
    rank_matrix.T,
    columns=[
        f"Ra={w1:.1f}, Rz={w2:.1f}"
        for w1, w2 in weight_scenarios
    ]
)

rank_stability_df[
    "Mean Rank"
] = rank_stability_df.mean(axis=1)

rank_stability_df[
    "Rank SD"
] = rank_stability_df.std(axis=1)

rank_stability_df[
    "Max Rank"
] = rank_stability_df.max(axis=1)

rank_stability_df[
    "Min Rank"
] = rank_stability_df.min(axis=1)

rank_stability_df[
    "Rank Range"
] = (
    rank_stability_df["Max Rank"]
    -
    rank_stability_df["Min Rank"]
)

print(
    "\n================ TOPSIS RANK STABILITY ================\n"
)

display(
    rank_stability_df.sort_values(
        "Mean Rank"
    ).head(20)
)


# ================================================================
# SECTION 50 — MOST STABLE TOPSIS SOLUTION
# ================================================================

most_stable_index = (
    rank_stability_df[
        "Rank SD"
    ].idxmin()
)

print(
    "\nMost rank-stable Pareto solution:"
)

display(
    pareto_uncertainty_df.iloc[
        most_stable_index
    ].to_frame().T
)


# ================================================================
# SECTION 51 — NSGA-II SENSITIVITY ANALYSIS
# ================================================================

nsga_scenarios = [

    (50, 50),

    (100, 50),

    (100, 100),

    (150, 100),

    (100, 150)
]


nsga_sensitivity_results = []


for population, generations in nsga_scenarios:

    print(
        f"\nRunning NSGA-II: "
        f"Population={population}, "
        f"Generations={generations}"
    )

    algorithm_sensitivity = NSGA2(
        pop_size=population
    )

    result_sensitivity = minimize(
        problem,
        algorithm_sensitivity,
        termination=(
            "n_gen",
            generations
        ),
        seed=RANDOM_STATE,
        verbose=False
    )

    F_sensitivity = (
        result_sensitivity.F
    )

    X_sensitivity = (
        result_sensitivity.X
    )

    # Remove duplicates
    temp_df = pd.DataFrame(
        X_sensitivity,
        columns=[
            "Vc",
            "Feed",
            "Depth"
        ]
    )

    temp_df["Ra"] = (
        F_sensitivity[:, 0]
    )

    temp_df["Rz"] = (
        F_sensitivity[:, 1]
    )

    temp_df = temp_df.round(6)

    temp_df = (
        temp_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    nsga_sensitivity_results.append({

        "Population":
            population,

        "Generations":
            generations,

        "Total Pareto Solutions":
            len(temp_df),

        "Unique Pareto Solutions":
            len(temp_df),

        "Ra Min":
            temp_df["Ra"].min(),

        "Ra Max":
            temp_df["Ra"].max(),

        "Ra Range":
            (
                temp_df["Ra"].max()
                -
                temp_df["Ra"].min()
            ),

        "Rz Min":
            temp_df["Rz"].min(),

        "Rz Max":
            temp_df["Rz"].max(),

        "Rz Range":
            (
                temp_df["Rz"].max()
                -
                temp_df["Rz"].min()
            )
    })


nsga_sensitivity_df = pd.DataFrame(
    nsga_sensitivity_results
)

print(
    "\n================ NSGA-II SENSITIVITY ================\n"
)

display(
    nsga_sensitivity_df
)


# ================================================================
# SECTION 52 — FEASIBILITY CHECK
# ================================================================

def check_state_feasibility(row):

    feasible = True

    for variable in [
        "Time",
        "Temperature",
        "Tool Wear"
    ]:

        lower, upper = (
            state_limits[variable]
        )

        value = row[variable]

        if (
            value < lower
            or
            value > upper
        ):

            feasible = False

    return feasible


pareto_uncertainty_df[
    "Feasible"
] = pareto_uncertainty_df.apply(
    check_state_feasibility,
    axis=1
)

print(
    "\n================ FEASIBILITY CHECK ================\n"
)

print(
    "Feasible solutions:",
    pareto_uncertainty_df[
        "Feasible"
    ].sum()
)

print(
    "Infeasible solutions:",
    (
        ~pareto_uncertainty_df[
            "Feasible"
        ]
    ).sum()
)


# ================================================================
# SECTION 53 — FEASIBLE PARETO SOLUTIONS ONLY
# ================================================================

feasible_pareto_df = (
    pareto_uncertainty_df[
        pareto_uncertainty_df["Feasible"]
    ]
    .reset_index(drop=True)
)

print(
    "\nFeasible Pareto solutions:",
    len(feasible_pareto_df)
)

display(
    feasible_pareto_df.head(20)
)


# ================================================================
# SECTION 54 — FINAL TOPSIS ON FEASIBLE SOLUTIONS
# ================================================================

if len(feasible_pareto_df) > 0:

    final_closeness, final_ranking = topsis(
        feasible_pareto_df,
        [0.50, 0.50]
    )

    feasible_pareto_df[
        "TOPSIS Closeness"
    ] = final_closeness

    feasible_pareto_df[
        "TOPSIS Rank"
    ] = final_ranking

    final_solution_index = (
        feasible_pareto_df[
            "TOPSIS Closeness"
        ].idxmax()
    )

    final_solution = (
        feasible_pareto_df.loc[
            final_solution_index
        ]
    )

    print(
        "\n================ FINAL TOPSIS SOLUTION ================\n"
    )

    display(
        final_solution.to_frame().T
    )

else:

    print(
        "\nNo feasible Pareto solution was found."
    )


# ================================================================
# SECTION 55 — FINAL OPTIMAL MACHINING CONDITION
# ================================================================

if len(feasible_pareto_df) > 0:

    print(
        "\n================================================"
    )

    print(
        "FINAL RECOMMENDED MACHINING CONDITION"
    )

    print(
        "================================================"
    )

    print(
        f"Vc           = "
        f"{final_solution['Vc']:.6f}"
    )

    print(
        f"Feed         = "
        f"{final_solution['Feed']:.6f}"
    )

    print(
        f"Depth        = "
        f"{final_solution['Depth']:.6f}"
    )

    print(
        f"Time         = "
        f"{final_solution['Time']:.6f}"
    )

    print(
        f"Temperature  = "
        f"{final_solution['Temperature']:.6f}"
    )

    print(
        f"Tool Wear    = "
        f"{final_solution['Tool Wear']:.6f}"
    )

    print(
        f"Predicted Ra = "
        f"{final_solution['Ra']:.6f} µm"
    )

    print(
        f"Predicted Rz = "
        f"{final_solution['Rz']:.6f} µm"
    )

    print(
        f"TOPSIS score = "
        f"{final_solution['TOPSIS Closeness']:.6f}"
    )


# ================================================================
# SECTION 56 — OBJECTIVE TRADE-OFF PLOT WITH FINAL SOLUTION
# ================================================================

if len(feasible_pareto_df) > 0:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        feasible_pareto_df["Ra"],
        feasible_pareto_df["Rz"],
        alpha=0.75,
        label="Pareto solutions"
    )

    plt.scatter(
        final_solution["Ra"],
        final_solution["Rz"],
        marker="*",
        s=250,
        label="TOPSIS solution"
    )

    plt.xlabel("Ra (µm)")
    plt.ylabel("Rz (µm)")
    plt.title(
        "Feasible Pareto Front and TOPSIS Compromise Solution"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# SECTION 57 — MODEL PERFORMANCE SUMMARY
# ================================================================

model_summary = []

for target, model, X, y in [

    (
        "Time",
        time_model,
        X_time,
        y_time
    ),

    (
        "Temperature",
        temperature_model,
        X_temperature,
        y_temperature
    ),

    (
        "Tool Wear",
        wear_model,
        X_wear,
        y_wear
    ),

    (
        "Ra",
        ra_model,
        X_ra,
        y_ra
    ),

    (
        "Rz",
        rz_model,
        X_rz,
        y_rz
    )
]:

    prediction = model.predict(X)

    model_summary.append({

        "Target":
            target,

        "R2":
            r2_score(
                y,
                prediction
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y,
                    prediction
                )
            ),

        "MAE":
            mean_absolute_error(
                y,
                prediction
            )
    })


model_summary_df = pd.DataFrame(
    model_summary
)

print(
    "\n================ STRUCTURAL MODEL PERFORMANCE ================\n"
)

display(
    model_summary_df
)


# ================================================================
# SECTION 58 — FEATURE IMPORTANCE SUMMARY
# ================================================================

def get_xgb_importance(
    model,
    feature_names
):

    importance = model.feature_importances_

    return pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importance
    }).sort_values(
        "Importance",
        ascending=False
    )


importance_ra_df = get_xgb_importance(
    ra_model,
    X_ra.columns
)

importance_rz_df = get_xgb_importance(
    rz_model,
    X_rz.columns
)

print(
    "\n================ XGBOOST IMPORTANCE: Ra ================\n"
)

display(
    importance_ra_df
)

print(
    "\n================ XGBOOST IMPORTANCE: Rz ================\n"
)

display(
    importance_rz_df
)


# ================================================================
# SECTION 59 — SAVE ALL RESULTS TO EXCEL
# ================================================================

output_filename = (
    "Causal_Machining_Optimization_Results.xlsx"
)

with pd.ExcelWriter(
    output_filename,
    engine="openpyxl"
) as writer:

    # Original data
    df.to_excel(
        writer,
        sheet_name="Original_Data",
        index=False
    )

    # Descriptive statistics
    df[
        required_columns
    ].describe().T.to_excel(
        writer,
        sheet_name="Descriptive_Statistics"
    )

    # Correlation
    corr.to_excel(
        writer,
        sheet_name="Correlation"
    )

    # Model performance
    model_summary_df.to_excel(
        writer,
        sheet_name="Structural_Model"
        ,
        index=False
    )

    # CV
    cv_results_df.to_excel(
        writer,
        sheet_name="10Fold_CV",
        index=False
    )

    # Baselines
    baseline_results_df.to_excel(
        writer,
        sheet_name="Ra_Baselines",
        index=False
    )

    baseline_rz_results_df.to_excel(
        writer,
        sheet_name="Rz_Baselines",
        index=False
    )

    # Feature importance
    importance_ra_df.to_excel(
        writer,
        sheet_name="Ra_Importance",
        index=False
    )

    importance_rz_df.to_excel(
        writer,
        sheet_name="Rz_Importance",
        index=False
    )

    # SHAP interaction
    interaction_df_ra.to_excel(
        writer,
        sheet_name="SHAP_Interactions"
    )

    # Pareto
    pareto_full_df.to_excel(
        writer,
        sheet_name="Pareto_Front",
        index=False
    )

    # Uncertainty
    pareto_uncertainty_df.to_excel(
        writer,
        sheet_name="Pareto_Uncertainty",
        index=False
    )

    # Feasible Pareto
    feasible_pareto_df.to_excel(
        writer,
        sheet_name="Feasible_Pareto",
        index=False
    )

    # TOPSIS sensitivity
    topsis_sensitivity_df.to_excel(
        writer,
        sheet_name="TOPSIS_Sensitivity",
        index=False
    )

    # Rank stability
    rank_stability_df.to_excel(
        writer,
        sheet_name="Rank_Stability",
        index=False
    )

    # NSGA sensitivity
    nsga_sensitivity_df.to_excel(
        writer,
        sheet_name="NSGA_Sensitivity",
        index=False
    )

print(
    "\nExcel results saved as:"
)

print(
    output_filename
)


# ================================================================
# SECTION 60 — DOWNLOAD RESULTS
# ================================================================

files.download(
    output_filename
)


# ================================================================
# SECTION 61 — FINAL SUMMARY
# ================================================================

print(
    "\n============================================================"
)

print(
    "ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "============================================================"
)

print(
    f"Dataset size: {len(df)} observations"
)

print(
    f"Unique Pareto solutions: "
    f"{len(pareto_unique_df)}"
)

print(
    f"Feasible Pareto solutions: "
    f"{len(feasible_pareto_df)}"
)

if len(feasible_pareto_df) > 0:

    print(
        "\nTOPSIS compromise solution:"
    )

    print(
        f"Vc = {final_solution['Vc']:.4f}"
    )

    print(
        f"Feed = {final_solution['Feed']:.4f}"
    )

    print(
        f"Depth = {final_solution['Depth']:.4f}"
    )

    print(
        f"Ra = {final_solution['Ra']:.4f} µm"
    )

    print(
        f"Rz = {final_solution['Rz']:.4f} µm"
    )

    print(
        f"Time = {final_solution['Time']:.4f}"
    )

    print(
        f"Temperature = "
        f"{final_solution['Temperature']:.4f}"
    )

    print(
        f"Tool Wear = "
        f"{final_solution['Tool Wear']:.4f}"
    )

print(
    "\nResults exported to:"
)

print(
    output_filename
)

print(
    "============================================================"
)
