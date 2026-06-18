# ✈️ Jet Engine Health Monitoring System (NASA C-MAPSS)

A complete, demo-ready prognostics system: it predicts the **Remaining Useful
Life (RUL)** of aircraft turbofan engines from sensor data, classifies their
health (NORMAL / WARNING / CRITICAL), and serves everything through an
interactive Streamlit dashboard.

**Pipeline:** Data → Model → Prediction → Decision → UI

---

## 1. Project structure

```
jet_engine_health/
├── train_model.py        # full ML pipeline (run this first)
├── app.py                # Streamlit dashboard (loads the saved model)
├── requirements.txt
├── README.md
├── data/                 # put the C-MAPSS .txt files here
│   ├── train_FD001.txt   test_FD001.txt   RUL_FD001.txt
│   └── ... (FD002, FD003, FD004)
├── models/
│   └── rf_model.joblib   # created by train_model.py
└── outputs/              # 4 PNG plots created by train_model.py
    ├── plot1_sensor_trend.png
    ├── plot2_early_vs_late.png
    ├── plot3_rul_vs_cycle.png
    └── plot4_pred_vs_actual.png
```

---

## 2. Install

```bash
pip install -r requirements.txt
```

(Python 3.9+ recommended.)

---

## 3. Run locally

**Step 1 — train the model & generate plots (run once):**

```bash
python train_model.py                 # uses FD001 by default
python train_model.py --dataset FD003 # optionally try another sub-dataset
```

This prints MAE / RMSE, the status distribution, a failure-case analysis, and
saves `models/rf_model.joblib` + the four plots in `outputs/`.

**Step 2 — launch the dashboard:**

```bash
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501).
Pick an engine (or a random sample) and press **Predict Engine Health**.

---

## 4. Run in Google Colab

```python
# in a Colab cell
!pip install -q streamlit pandas numpy scikit-learn matplotlib joblib
# upload the data/ files, then:
!python train_model.py --dataset FD001
```

To view the Streamlit app from Colab, use a tunnel (e.g. `localtunnel`):

```python
!npm install -g localtunnel
!streamlit run app.py &>/content/log.txt &
!npx localtunnel --port 8501
```

The training script alone (`train_model.py`) runs fully in Colab and prints all
metrics + saves the plots, so the app is optional there.

---

## 5. What each pipeline stage does

| Function           | Purpose                                                        |
|--------------------|----------------------------------------------------------------|
| `data_loading()`   | Read train/test/RUL, assign the 26 C-MAPSS column names        |
| `preprocess()`     | RUL = max_cycle − cycle (clipped at 125), drop constant sensors, scale |
| `train_model()`    | Fit `RandomForestRegressor` (200 trees)                         |
| `predict()`        | Predict RUL on the last cycle of each test engine, clip to [0,125] |
| `evaluate_model()` | Compute & print MAE and RMSE                                    |
| `decision_logic()` | RUL>100 → NORMAL · RUL>50 → WARNING · else CRITICAL             |
| `failure_case()`   | Find the worst prediction and explain why it failed            |
| `visualization()`  | 4 titled, labelled plots                                        |

---

## 6. Results (FD001 baseline)

| Metric | Value (cycles) |
|--------|----------------|
| MAE    | ≈ 11.9         |
| RMSE   | ≈ 16.9         |

These are solid baseline numbers for a RandomForest on FD001.

### Why RUL is capped at 125
At the start of its life an engine is healthy, so its "true" RUL isn't really
observable from sensors. Modelling RUL as flat (= the cap) until degradation
begins turns the target into a piece-wise-linear curve and is standard practice
for C-MAPSS — it noticeably lowers RMSE.

---

## 7. Ideas to push it further
- Add rolling-window / lag features per sensor (big accuracy gain).
- Try gradient boosting (XGBoost / LightGBM) or an LSTM for sequence modelling.
- For FD002 / FD004 (6 operating conditions): normalise sensors *per operating
  condition* before training.
- Add the official C-MAPSS asymmetric scoring function (penalises late
  predictions more) as an extra evaluation metric.
