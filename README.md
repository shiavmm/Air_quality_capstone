# BDS-07 | Hyperlocal Air Quality Forecasting
### Sensor Fusion + Explainable Source Signals | T.Y. B.Sc. Data Science — Semester V

A locally-runnable, end-to-end ML prototype that ingests live air quality and weather data from the OpenWeather API, trains a LightGBM forecasting model with full SHAP explainability, and serves predictions via a FastAPI backend and Streamlit dashboard.

---

## Project Structure

```
bds-07-air-quality/
├── src/
│   ├── data/           → Ingestion + data quality validation
│   ├── features/       → Feature engineering (wind vectors, lags, rolling stats)
│   ├── models/         → Baselines, LightGBM training, SHAP explainability
│   ├── analysis/       → Trend & seasonality analysis (statsmodels)
│   ├── api/            → FastAPI prediction microservice
│   └── dashboard/      → Streamlit interactive dashboard
├── models/             → Saved LightGBM booster (.txt)
├── data/               → raw/ and processed/ datasets (git-ignored)
├── reports/figures/    → SHAP summary plots
├── docs/               → Design docs, model card, data dictionary
├── tests/              → pytest unit + integration tests
├── mlflow.db           → MLflow experiment tracking (SQLite)
└── requirements.txt
```

---

## Setup (Clean Machine)

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone & Install
```bash
git clone https://github.com/shiavmm/Air_quality_capstone.git
cd bds-07-air-quality
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Copy the example env file
cp .env.example .env

# Open .env and paste your OpenWeather API key
# Get a free key at: https://openweathermap.org/api
OPENWEATHER_API_KEY=your_actual_key_here
```

---

## Running the Pipeline

### Step 1 — Ingest Live Data
```bash
python src/data/ingest.py
```
Fetches live PM2.5, PM10, NO2, temperature, humidity, wind for Delhi, Mumbai, Bengaluru.

### Step 2 — Feature Engineering
```bash
python src/features/build_features.py
```
Generates wind vectors, autoregressive lags (t-1 … t-24), and rolling statistics. Saves to `data/processed/processed_features.parquet`.

### Step 3 — Evaluate Baselines
```bash
python src/models/baselines.py
```
Runs Tier 1 (Persistence) and Tier 2 (24h Rolling Naive) baseline models.

### Step 4 — Train LightGBM Model
```bash
python src/models/train_lgbm.py
```
Trains LightGBM regressor, logs metrics to MLflow, saves model to `models/lgbm_model.txt`.

### Step 5 — SHAP Explainability
```bash
python src/models/explainability.py
```
Generates global SHAP feature attribution plot at `reports/figures/shap_summary.png`.

### Step 6 — Trend & Seasonality Analysis
```bash
python src/analysis/trend_seasonality.py
```
Runs STL decomposition on PM2.5 time series (trend, seasonal, residual components).

### Step 7 — Start the API Server
```bash
# From the project root
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI available at: http://127.0.0.1:8000/docs

### Step 8 — Launch the Dashboard
```bash
# In a new terminal (API must be running)
streamlit run src/dashboard/app.py
```
Dashboard opens at: http://localhost:8501

---

## Run Tests
```bash
pytest tests/ -v
```

---

## View MLflow Experiment Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Opens at: http://127.0.0.1:5000

---

## Model Performance

| Model | MAE | RMSE |
|-------|-----|------|
| Tier 1 — Persistence Baseline | 47.07 | — |
| Tier 2 — 24h Rolling Naive | — | — |
| **Tier 3 — LightGBM (ours)** | **35.73** | — |
| MAE Reduction vs Tier 1 | **24.1%** | — |

Validation: Chronological 80/20 split with 24-hour purge gap (zero data leakage).

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | LightGBM, scikit-learn |
| MLOps | MLflow (SQLite) |
| Explainability | SHAP (TreeSHAP) |
| Trend Analysis | statsmodels (STL) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Geospatial | GeoPandas, Folium |
| Data Source | OpenWeather API (Air Pollution + Weather) |

---

## Security Notes
- API keys are stored in `.env` (git-ignored) — never committed to the repository
- See `.env.example` for required environment variables
