"""
================================================================================
 Jet Engine Health Monitoring System  --  Streamlit dashboard (app.py)
================================================================================
Loads the pre-trained model bundle (models/rf_model.joblib) produced by
train_model.py and serves an interactive dashboard. No retraining happens in
the UI, so predictions are instant.

Run:
    streamlit run app.py
================================================================================
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
#  Paths / constants                                                          #
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "data")

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

STATUS_COLORS = {"NORMAL": "#2bb673", "WARNING": "#f0a202", "CRITICAL": "#d1495b"}


# --------------------------------------------------------------------------- #
#  Cached loaders (run once, then reused -> fast UI)                          #
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data(dataset):
    path = os.path.join(DATA_DIR, f"test_{dataset}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = ALL_COLS
    return df


def decision_logic(rul, warning, critical):
    if rul > warning:
        return "NORMAL"
    elif rul > critical:
        return "WARNING"
    return "CRITICAL"


# --------------------------------------------------------------------------- #
#  Page setup                                                                 #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Jet Engine Health Monitoring System",
                   page_icon="✈️", layout="wide")

st.title("✈️ Jet Engine Health Monitoring System")
st.caption("NASA C-MAPSS turbofan dataset · RandomForest RUL prediction")

bundle = load_bundle()
if bundle is None:
    st.error("Model not found. Run `python train_model.py` first to create "
             "models/rf_model.joblib.")
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
dataset = bundle["dataset"]
rul_cap = bundle["rul_cap"]
warning = bundle["thresholds"]["warning"]
critical = bundle["thresholds"]["critical"]

test_df = load_test_data(dataset)
engine_ids = sorted(test_df["unit"].unique())

# --------------------------------------------------------------------------- #
#  Sidebar                                                                    #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    st.markdown(f"**Dataset:** `{dataset}`")
    st.markdown(f"**Model MAE:** {bundle['metrics']['MAE']:.2f} cycles")
    st.markdown(f"**Model RMSE:** {bundle['metrics']['RMSE']:.2f} cycles")
    st.divider()

    mode = st.radio("Input mode", ["Select engine ID", "Random sample"])
    if mode == "Select engine ID":
        engine_id = st.selectbox("Engine ID", engine_ids)
    else:
        if st.button("🎲 Pick random engine"):
            st.session_state["rand_engine"] = int(np.random.choice(engine_ids))
        engine_id = st.session_state.get("rand_engine", engine_ids[0])
        st.markdown(f"Selected engine: **{engine_id}**")

    st.divider()
    predict_clicked = st.button("🔮 Predict Engine Health", type="primary",
                                use_container_width=True)

# --------------------------------------------------------------------------- #
#  Main panel                                                                 #
# --------------------------------------------------------------------------- #
engine_data = test_df[test_df["unit"] == engine_id].sort_values("cycle")
last_row = engine_data.iloc[[-1]]

col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader(f"Engine #{engine_id}")
    st.metric("Cycles recorded", int(engine_data["cycle"].max()))

    if predict_clicked:
        X = scaler.transform(last_row[feature_cols])
        rul = float(np.clip(model.predict(X)[0], 0, rul_cap))
        status = decision_logic(rul, warning, critical)
        color = STATUS_COLORS[status]

        st.metric("Predicted RUL", f"{rul:.0f} cycles")
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;background:{color};"
            f"color:white;font-size:22px;font-weight:700;text-align:center'>"
            f"STATUS: {status}</div>",
            unsafe_allow_html=True,
        )
        if status == "CRITICAL":
            st.error("Immediate maintenance recommended.")
        elif status == "WARNING":
            st.warning("Schedule maintenance soon.")
        else:
            st.success("Engine operating within safe limits.")
    else:
        st.info("Choose an engine and press **Predict Engine Health**.")

with col_right:
    st.subheader("Sensor trend over the engine's recorded life")
    sensor = "sensor_4" if "sensor_4" in feature_cols else feature_cols[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(engine_data["cycle"], engine_data[sensor],
            color="#6c2bd9", lw=1.8)
    ax.set_title(f"{sensor} vs cycle  (engine #{engine_id})", fontweight="bold")
    ax.set_xlabel("Operational cycle")
    ax.set_ylabel(f"{sensor} reading")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with st.expander("Show raw sensor table for this engine"):
    st.dataframe(engine_data[INDEX_COLS + feature_cols].reset_index(drop=True),
                 use_container_width=True)
