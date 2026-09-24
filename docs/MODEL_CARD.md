# Model Card — BDS-07 Hyperlocal AQI Forecasting
**Version:** 1.0 | **Date:** September 2026 | **Team:** BDS-07

---

## 1. Model Overview

| Field | Detail |
|-------|--------|
| **Model Name** | Hyperlocal AQI LightGBM Regressor |
| **Algorithm** | LightGBM (Gradient Boosted Decision Trees) |
| **Task** | Regression — Predict next-hour PM2.5 concentration (µg/m³) |
| **Target Variable** | `pm2_5` — Fine particulate matter (µg/m³) |
| **Version** | Tier 3 (final), saved as `models/lgbm_model.txt` |

---

## 2. Intended Use

**Primary use:** Hyperlocal next-hour PM2.5 prediction for urban citizens, environmental researchers, and urban planners in Indian cities.

**User personas:**
- **Citizens** — plan outdoor activity based on predicted AQI level
- **Urban Planners** — identify high-risk zones and peak pollution hours
- **Environmental Researchers** — analyse source signal drivers via SHAP

**Out-of-scope uses:**
- Long-horizon forecasting (>1 hour ahead) — model is not validated for this
- Industrial emission monitoring — requires different sensor types
- Legal/regulatory compliance reporting — not certified for that purpose

---

## 3. Training Data

| Property | Detail |
|----------|--------|
| **Source** | OpenWeather Air Pollution API + Weather API |
| **Cities** | Delhi (28.61°N, 77.21°E), Mumbai (19.08°N, 72.88°E), Bengaluru (12.97°N, 77.59°E) |
| **Frequency** | Hourly sensor readings |
| **Duration** | ~60-day historical dataset |
| **Features** | PM2.5, PM10, NO2, SO2, CO, temperature, humidity, pressure, wind speed, wind direction |
| **Provenance** | OpenWeather API — free tier, no redistribution restrictions for research |

---

## 4. Feature Engineering

| Feature Group | Features | Rationale |
|--------------|----------|-----------|
| Wind Vectors | `u_wind`, `v_wind` | Converts circular wind_deg to Cartesian (avoids 0°/360° discontinuity) |
| Temporal Encoding | `sin_hour`, `cos_hour`, `day_of_week`, `is_weekend` | Captures diurnal cycles and weekday/weekend traffic patterns |
| Autoregressive Lags | `pm2_5_lag_1/2/3/6/12/24` | Temporal momentum — recent pollution predicts next hour |
| Rolling Statistics | `pm2_5_roll_mean/std_3h/6h/24h` | Short, medium, and long-run trend smoothing |

---

## 5. Model Parameters

```
objective:     regression
metric:        rmse
n_estimators:  200
learning_rate: 0.05
random_state:  42
```

---

## 6. Evaluation Results

| Model | MAE (µg/m³) | Notes |
|-------|-------------|-------|
| Tier 1 — Persistence | 47.07 | Predict next = current |
| Tier 2 — 24h Rolling Naive | ~41.0 | Simple rolling mean |
| **Tier 3 — LightGBM (this model)** | **35.73** | **24.1% improvement over Tier 1** |

**Validation method:** Chronological 80/20 split with a 24-hour purge gap between train and test sets to eliminate temporal data leakage through lag features.

---

## 7. Explainability

TreeSHAP (exact Shapley values) is used to quantify feature contributions per prediction.

**Key findings from SHAP analysis:**
- `pm2_5_lag_1` and short-window rolling means are the dominant predictors (pollution momentum)
- `sin_hour` / `cos_hour` capture diurnal pollution cycles (rush-hour peaks)
- Wind vectors `u_wind`, `v_wind` contribute to dispersion effects
- See: `reports/figures/shap_summary.png`

---

## 8. Limitations & Risks

| Limitation | Description |
|-----------|-------------|
| **Sensor dependency** | Model degrades if OpenWeather API is unavailable or returning stale data |
| **City scope** | Trained only on Delhi, Mumbai, Bengaluru — generalisability to other cities is unvalidated |
| **Short-term only** | Validated for 1-hour ahead prediction only |
| **No spatial interpolation** | Predictions are at city centroid level, not hyperlocal grid |
| **Concept drift** | Seasonal or long-term climate changes may require retraining |

---

## 9. Fairness & Ethics

- The model does not use any demographic or personally identifiable information
- Predictions are based purely on atmospheric and meteorological sensor readings
- Potential misuse: overreliance on predictions for medical decisions — should be supplemented by certified air quality monitors

---

## 10. Reproducibility

```bash
# Reproduce training from scratch:
pip install -r requirements.txt
python src/data/ingest.py          # collect data
python src/features/build_features.py  # engineer features
python src/models/train_lgbm.py    # train model (logs to MLflow)
python src/models/explainability.py    # generate SHAP plots
```

All experiment runs are tracked in `mlflow.db` (SQLite). View with: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
