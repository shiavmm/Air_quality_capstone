import os
import sys
import time
import unicodedata
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parents[2]))

app = FastAPI(
    title="Hyperlocal AQI Engine",
    description="Production Sensor Fusion API (True Time-Series Lags + SHAP)",
    version="3.2.0",
)

MODEL_PATH = "models/lgbm_model.txt"
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    import warnings
    warnings.warn("OPENWEATHER_API_KEY missing! Add it to your .env file. Live fetch will fail.", stacklevel=1)

if os.path.exists(MODEL_PATH):
    model = lgb.Booster(model_file=MODEL_PATH)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Warning: Could not initialize TreeExplainer: {e}")
        explainer = None
else:
    model = None
    explainer = None


class CoordinatePayload(BaseModel):
    lat: float
    lon: float


class SensorPayload(BaseModel):
    lat: float = 19.0760
    lon: float = 72.8777
    temp_celsius: float = 28.5
    humidity: float = 65.0
    pressure: float = 1013.25
    wind_speed: float = 5.2
    wind_deg: float = 180.0
    pm10: float = 50.0
    no2: float = 20.0
    so2: float = 10.0
    co: float = 500.0
    pm2_5_lag_1: float = 45.0
    pm2_5_lag_2: float = 44.0
    pm2_5_lag_3: float = 43.5
    pm2_5_lag_6: float = 42.0
    pm2_5_lag_12: float = 40.0
    pm2_5_lag_24: float = 38.0
    pm2_5_roll_mean_3h: float = 44.1
    pm2_5_roll_std_3h: float = 1.2
    pm2_5_roll_mean_6h: float = 43.0
    pm2_5_roll_std_6h: float = 2.1
    pm2_5_roll_mean_24h: float = 41.5
    pm2_5_roll_std_24h: float = 3.8
    sin_hour: float = 0.0
    cos_hour: float = 1.0
    u_wind: float = 0.0
    v_wind: float = 5.2
    day_of_week: int = 3
    hour: int = 12
    is_weekend: int = 0


@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}


@app.post("/fetch_coords")
def fetch_live_coordinate_data(payload: CoordinatePayload):
    lat, lon = payload.lat, payload.lon

    if not API_KEY:
        raise HTTPException(status_code=503, detail="OPENWEATHER_API_KEY not configured on the server.")

    try:
        # Current Weather
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        w_resp = requests.get(w_url, timeout=10).json()
        if str(w_resp.get("cod", 200)) != "200":
            raise HTTPException(
                status_code=502,
                detail=f"OpenWeatherMap Weather API error ({w_resp.get('cod')}): {w_resp.get('message', 'Unknown error')}. Check your OPENWEATHER_API_KEY.",
            )

        # Current Air Pollution
        a_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        a_resp = requests.get(a_url, timeout=10).json()
        if "list" not in a_resp or len(a_resp["list"]) == 0:
            raise HTTPException(
                status_code=502,
                detail=f"OpenWeatherMap Air Pollution API error: {a_resp.get('message', a_resp)}. Check your OPENWEATHER_API_KEY.",
            )

        # Past 24 Hours Air Pollution History for True Lags
        end_time = int(time.time())
        start_time = end_time - (25 * 3600)  # 25 hours back
        h_url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_time}&end={end_time}&appid={API_KEY}"
        h_resp = requests.get(h_url, timeout=10).json()

        pm25_series = []
        if "list" in h_resp and len(h_resp["list"]) > 0:
            pm25_series = [
                item["components"]["pm2_5"] for item in h_resp["list"]
            ]

        # If historical list is shorter than 25, pad with current PM2.5 value naturally
        current_pm25 = float(a_resp["list"][0]["components"]["pm2_5"])
        while len(pm25_series) < 25:
            pm25_series.insert(0, current_pm25)

        # Compute True Rolling & Lag Stats
        s = pd.Series(pm25_series)
        lag_1 = float(s.iloc[-1])
        lag_2 = float(s.iloc[-2])
        lag_3 = float(s.iloc[-3])
        lag_6 = float(s.iloc[-6])
        lag_12 = float(s.iloc[-12])
        lag_24 = float(s.iloc[-24])

        roll_mean_3h = float(s.tail(3).mean())
        roll_std_3h = float(s.tail(3).std(ddof=0))
        roll_mean_6h = float(s.tail(6).mean())
        roll_std_6h = float(s.tail(6).std(ddof=0))
        roll_mean_24h = float(s.tail(24).mean())
        roll_std_24h = float(s.tail(24).std(ddof=0))

        components = a_resp["list"][0].get("components", {})
        return {
            "lat": lat,
            "lon": lon,
            "temp": float(w_resp["main"]["temp"]),
            "humidity": float(w_resp["main"]["humidity"]),
            "pressure": float(w_resp["main"].get("pressure", 1013.25)),
            "wind_speed": float(w_resp["wind"]["speed"]),
            "wind_deg": float(w_resp["wind"].get("deg", 180.0)),
            "pm2_5": current_pm25,
            "pm10": float(components.get("pm10", 50.0)),
            "no2": float(components.get("no2", 20.0)),
            "so2": float(components.get("so2", 10.0)),
            "co": float(components.get("co", 500.0)),
            "location_name": unicodedata.normalize("NFKD", str(w_resp.get("name", "Grid Station"))).encode("ascii", "ignore").decode("ascii") or "Grid Station",
            "lags": {
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_6": lag_6,
                "lag_12": lag_12,
                "lag_24": lag_24,
                "roll_mean_3h": roll_mean_3h,
                "roll_std_3h": roll_std_3h,
                "roll_mean_6h": roll_mean_6h,
                "roll_std_6h": roll_std_6h,
                "roll_mean_24h": roll_mean_24h,
                "roll_std_24h": roll_std_24h,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"API Fetch Error: {str(e)}"
        )


@app.post("/predict")
def predict_air_quality(payload: SensorPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model file missing!")

    data_dict = payload.dict()
    features = model.feature_name()

    # Fallback defaults for any expected model features
    defaults = {
        "lat": 19.0760, "lon": 72.8777, "pressure": 1013.25,
        "no2": 20.0, "so2": 10.0, "co": 500.0, "pm10": 50.0,
        "temp_celsius": 28.5, "humidity": 65.0, "wind_speed": 5.0, "wind_deg": 180.0,
    }
    for f in features:
        if f not in data_dict:
            data_dict[f] = defaults.get(f, 0.0)

    input_data = pd.DataFrame([data_dict])
    X_infer = input_data[features]

    prediction = model.predict(X_infer)[0]

    # Compute Local SHAP Attributions for this specific point
    shap_contributions = []
    base_val = 55.99
    if explainer is not None:
        try:
            shap_res = explainer(X_infer)
            base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 55.99
            vals = shap_res.values[0]
            shap_contributions = [
                {
                    "feature": f,
                    "shap_value": round(float(v), 3),
                    "feature_value": round(float(X_infer[f].iloc[0]), 2),
                }
                for f, v in sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)
            ]
        except Exception as e:
            print(f"Warning: Local SHAP calculation error: {e}")

    return {
        "predicted_pm2_5": round(float(prediction), 2),
        "base_value": round(float(base_val), 2),
        "shap_contributions": shap_contributions,
        "status": "SUCCESS",
    }