"""
================================================================================
 Jet Engine Health Monitoring System  --  NASA C-MAPSS Turbofan Dataset
 train_model.py  :  full ML pipeline (data -> model -> prediction -> decision)
================================================================================

Pipeline stages (each in its own function, as required):
    data_loading()   -> read train/test/RUL files, assign column names
    preprocess()     -> compute RUL, drop constant sensors, scale features
    train_model()    -> RandomForestRegressor on the training set
    predict()        -> RUL prediction on the (last cycle of each) test engine
    evaluate_model() -> MAE + RMSE
    decision_logic() -> NORMAL / WARNING / CRITICAL classification
    visualization()  -> 4 high-quality EDA / result plots
    failure_case()   -> find and explain the worst prediction

The trained model + scaler + feature list are saved with joblib so the
Streamlit app can load them instantly (no retraining in the UI).

Run:
    python train_model.py
    python train_model.py --dataset FD003     # try another sub-dataset
================================================================================
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
# Folder that contains train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Column names (per the C-MAPSS readme: unit, cycle, 3 op-settings, 21 sensors)
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# RUL is clipped at this value. This is standard practice for C-MAPSS:
# at the start of life the engine is healthy and the "true" RUL is not really
# observable, so we model RUL as flat (=cap) until degradation starts. This
# turns the target into a piece-wise-linear curve and greatly improves accuracy.
RUL_CAP = 125

# Decision thresholds (cycles of remaining useful life)
WARNING_THRESHOLD = 100   # RUL > 100  -> NORMAL
CRITICAL_THRESHOLD = 50   # RUL > 50   -> WARNING ; else CRITICAL


# --------------------------------------------------------------------------- #
#  1. DATA LOADING                                                            #
# --------------------------------------------------------------------------- #
def data_loading(dataset="FD001", data_dir=DATA_DIR):
    """Load the train, test and ground-truth RUL files for one sub-dataset.

    Returns
    -------
    train_df : DataFrame  -- every operational cycle of every training engine
    test_df  : DataFrame  -- every operational cycle of every test engine
    rul_true : ndarray    -- true RUL at the LAST cycle of each test engine
    """
    train_path = os.path.join(data_dir, f"train_{dataset}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset}.txt")

    for p in (train_path, test_path, rul_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Put train_{dataset}.txt / test_{dataset}.txt / RUL_{dataset}.txt "
                f"into the '{data_dir}' folder."
            )

    # Files are space-separated; trailing spaces create 2 phantom columns -> drop them.
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None).iloc[:, :26]
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None).iloc[:, :26]
    train_df.columns = ALL_COLS
    test_df.columns = ALL_COLS

    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].values

    print(f"[data_loading] dataset={dataset}")
    print(f"  train: {train_df.shape[0]} rows, "
          f"{train_df['unit'].nunique()} engines")
    print(f"  test : {test_df.shape[0]} rows, "
          f"{test_df['unit'].nunique()} engines")
    print(f"  true RUL vector length: {len(rul_true)}")
    return train_df, test_df, rul_true


# --------------------------------------------------------------------------- #
#  2. PREPROCESSING  (RUL target, feature selection, scaling)                 #
# --------------------------------------------------------------------------- #
def add_rul(df):
    """Add the RUL column:  RUL = max_cycle_of_engine - current_cycle."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["cycle"]
    return df


def preprocess(train_df, test_df, rul_cap=RUL_CAP):
    """Compute RUL, drop constant sensors, scale features.

    Returns a dictionary holding everything downstream stages need.
    """
    # --- target -----------------------------------------------------------
    train_df = add_rul(train_df)
    # Clip RUL (piece-wise-linear target, see RUL_CAP note above)
    train_df["RUL"] = train_df["RUL"].clip(upper=rul_cap)

    # --- drop constant / near-constant sensors ----------------------------
    # In single-condition datasets several sensors never move and carry no
    # information; we remove any feature whose std (on the training set) ~ 0.
    candidate_features = SETTING_COLS + SENSOR_COLS
    stds = train_df[candidate_features].std()
    dropped = stds[stds < 1e-6].index.tolist()
    feature_cols = [c for c in candidate_features if c not in dropped]
    print(f"[preprocess] dropped {len(dropped)} constant features: {dropped}")
    print(f"[preprocess] using {len(feature_cols)} features")

    # --- scaling -----------------------------------------------------------
    # RandomForest is scale-invariant, but we fit a StandardScaler anyway so
    # the exact same transform is reused in the app and so the pipeline stays
    # model-agnostic (you could swap in an SVR / NN without other changes).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    y_train = train_df["RUL"].values

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
    }


def build_test_matrix(test_df, feature_cols, scaler):
    """Take the LAST cycle of every test engine (that is where the provided
    ground-truth RUL is defined) and scale it the same way as training."""
    last_rows = test_df.groupby("unit").last().reset_index()
    X_test = scaler.transform(last_rows[feature_cols])
    return last_rows, X_test


# --------------------------------------------------------------------------- #
#  3. MODEL TRAINING                                                          #
# --------------------------------------------------------------------------- #
def train_model(X_train, y_train):
    """Train a RandomForestRegressor."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    print("[train_model] fitting RandomForestRegressor ...")
    model.fit(X_train, y_train)
    print("[train_model] done.")
    return model


# --------------------------------------------------------------------------- #
#  4. PREDICTION                                                              #
# --------------------------------------------------------------------------- #
def predict(model, X, rul_cap=RUL_CAP):
    """Predict RUL and clip to the valid range [0, rul_cap]."""
    preds = model.predict(X)
    return np.clip(preds, 0, rul_cap)


# --------------------------------------------------------------------------- #
#  5. EVALUATION                                                              #
# --------------------------------------------------------------------------- #
def evaluate_model(y_true, y_pred):
    """Compute and print MAE and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("\n========== MODEL EVALUATION ==========")
    print(f"  MAE  : {mae:6.2f} cycles")
    print(f"  RMSE : {rmse:6.2f} cycles")
    print("======================================\n")
    return {"MAE": mae, "RMSE": rmse}


# --------------------------------------------------------------------------- #
#  6. DECISION LOGIC                                                          #
# --------------------------------------------------------------------------- #
def decision_logic(rul):
    """Map a single RUL value to a maintenance status."""
    if rul > WARNING_THRESHOLD:
        return "NORMAL"
    elif rul > CRITICAL_THRESHOLD:
        return "WARNING"
    else:
        return "CRITICAL"


def apply_decision(rul_array):
    """Vectorised version of decision_logic for an array of RULs."""
    return np.array([decision_logic(r) for r in rul_array])


# --------------------------------------------------------------------------- #
#  7. FAILURE CASE                                                            #
# --------------------------------------------------------------------------- #
def failure_case(units, y_true, y_pred):
    """Find the worst prediction and explain why it is a failure."""
    abs_err = np.abs(y_pred - y_true)
    i = int(np.argmax(abs_err))
    pred_status = decision_logic(y_pred[i])
    true_status = decision_logic(y_true[i])
    print("========== FAILURE CASE (worst prediction) ==========")
    print(f"  Engine unit      : {int(units[i])}")
    print(f"  True RUL         : {y_true[i]:.0f} cycles  -> status {true_status}")
    print(f"  Predicted RUL    : {y_pred[i]:.1f} cycles  -> status {pred_status}")
    print(f"  Absolute error   : {abs_err[i]:.1f} cycles")
    direction = "LATE (dangerous)" if y_pred[i] > y_true[i] else "EARLY (conservative)"
    print(f"  Prediction is    : {direction}")
    print("  Why it failed    : at its last recorded cycle this engine's sensor")
    print("    signature looked healthier (or noisier) than its real wear state,")
    print("    so the forest averaged toward a longer life. Late predictions like")
    print("    this are the costly ones in PHM — the engine fails sooner than")
    print("    forecast. The C-MAPSS scoring function penalises them more heavily.")
    print("=====================================================\n")
    return {
        "unit": int(units[i]),
        "true_rul": float(y_true[i]),
        "pred_rul": float(y_pred[i]),
        "abs_error": float(abs_err[i]),
        "direction": direction,
    }


# --------------------------------------------------------------------------- #
#  8. VISUALISATION  (4 high-quality plots)                                   #
# --------------------------------------------------------------------------- #
def visualization(train_df, last_rows, y_true, y_pred, feature_cols,
                  plot_dir=PLOT_DIR, dataset="FD001"):
    """Generate the four required plots and save them as PNGs."""
    os.makedirs(plot_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")
    saved = {}

    # --- Plot 1: sensor trend vs cycle for one engine ---------------------
    # sensor_4 (HPC outlet temperature) shows a strong degradation trend.
    sensor = "sensor_4" if "sensor_4" in feature_cols else feature_cols[0]
    eng = train_df[train_df["unit"] == 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eng["cycle"], eng[sensor], color="#6c2bd9", lw=1.8)
    ax.set_title(f"Sensor trend over engine life  (engine #1, {sensor})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Operational cycle")
    ax.set_ylabel(f"{sensor} reading")
    fig.tight_layout()
    p1 = os.path.join(plot_dir, "plot1_sensor_trend.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    saved["sensor_trend"] = p1

    # --- Plot 2: early vs late cycle comparison ---------------------------
    # Distribution of a degradation sensor for early-life vs late-life cycles.
    early = train_df[train_df["RUL"] >= 100][sensor]
    late = train_df[train_df["RUL"] <= 20][sensor]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(early, bins=40, alpha=0.7, label="Early life (RUL >= 100)",
            color="#2bb673")
    ax.hist(late, bins=40, alpha=0.7, label="Late life (RUL <= 20)",
            color="#d1495b")
    ax.set_title(f"Early vs late cycle distribution  ({sensor})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{sensor} reading")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(plot_dir, "plot2_early_vs_late.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    saved["early_vs_late"] = p2

    # --- Plot 3: RUL vs cycle for a few engines ---------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for u in [1, 2, 3, 4, 5]:
        e = train_df[train_df["unit"] == u]
        ax.plot(e["cycle"], e["RUL"], lw=1.6, label=f"engine {u}")
    ax.set_title("Remaining Useful Life vs cycle  (RUL capped at %d)" % RUL_CAP,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Operational cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(plot_dir, "plot3_rul_vs_cycle.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    saved["rul_vs_cycle"] = p3

    # --- Plot 4: predicted vs actual RUL ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#6c2bd9", edgecolor="white")
    lim = max(y_true.max(), y_pred.max()) + 5
    ax.plot([0, lim], [0, lim], "--", color="#333", lw=1.5, label="perfect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Predicted vs actual RUL (test set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(plot_dir, "plot4_pred_vs_actual.png")
    fig.savefig(p4, dpi=130)
    plt.close(fig)
    saved["pred_vs_actual"] = p4

    print(f"[visualization] saved {len(saved)} plots to {plot_dir}")
    return saved


# --------------------------------------------------------------------------- #
#  MAIN                                                                       #
# --------------------------------------------------------------------------- #
def main(dataset="FD001"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. load
    train_df, test_df, rul_true = data_loading(dataset)

    # 2. preprocess
    pp = preprocess(train_df, test_df)
    feature_cols = pp["feature_cols"]
    scaler = pp["scaler"]

    # 3. train
    model = train_model(pp["X_train"], pp["y_train"])

    # 4. predict on the last cycle of each test engine
    last_rows, X_test = build_test_matrix(test_df, feature_cols, scaler)
    y_pred = predict(model, X_test)
    y_true = np.clip(rul_true, 0, RUL_CAP)  # clip ground truth the same way

    # 5. evaluate
    metrics = evaluate_model(y_true, y_pred)

    # 6. decision logic on every test engine
    statuses = apply_decision(y_pred)
    uniq, counts = np.unique(statuses, return_counts=True)
    print("[decision_logic] status distribution on test fleet:")
    for s, c in zip(uniq, counts):
        print(f"    {s:8s}: {c} engines")
    print()

    # 7. failure case
    fc = failure_case(last_rows["unit"].values, y_true, y_pred)

    # 8. visualisation
    plots = visualization(pp["train_df"], last_rows, y_true, y_pred,
                          feature_cols, dataset=dataset)

    # --- save everything the app needs (no retraining in the UI) ----------
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dataset": dataset,
        "rul_cap": RUL_CAP,
        "thresholds": {"warning": WARNING_THRESHOLD,
                       "critical": CRITICAL_THRESHOLD},
        "metrics": metrics,
    }
    model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    joblib.dump(bundle, model_path, compress=3)  # compress -> small file
    print(f"[main] model bundle saved -> {model_path}")

    return metrics, fc, plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL model")
    parser.add_argument("--dataset", default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"],
                        help="which C-MAPSS sub-dataset to use")
    args = parser.parse_args()
    main(args.dataset)
