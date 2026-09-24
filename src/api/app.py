import os
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))

app = FastAPI(
    title="Hyperlocal AQI Engine",
    description="Production Sensor Fusion API (True Time-Series Lags)",
    version="3.1.0",
)

MODEL_PATH = "models/lgbm_model.txt"
API_KEY = "388a10b4ef72d7f8d6ee5cfaae5aefc0"

if os.path.exists(MODEL_PATH):
    model = lgb.Booster(model_file=MODEL_PATH)
else:
    model = None


class CoordinatePayload(BaseModel):
    lat: float
    lon: float


class SensorPayload(BaseModel):
    temp_celsius: float
    humidity: float
    wind_speed: float
    wind_deg: float
    pm2_5_lag_1: float
    pm2_5_lag_2: float
    pm2_5_lag_3: float
    pm2_5_lag_6: float
    pm2_5_lag_12: float
    pm2_5_lag_24: float
    pm2_5_roll_mean_3h: float
    pm2_5_roll_std_3h: float
    pm2_5_roll_mean_6h: float
    pm2_5_roll_std_6h: float
    pm2_5_roll_mean_24h: float
    pm2_5_roll_std_24h: float
    sin_hour: float
    cos_hour: float
    u_wind: float
    v_wind: float
    day_of_week: int
    hour: int
    is_weekend: int
    pm10: float


@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}


@app.post("/fetch_coords")
def fetch_live_coordinate_data(payload: CoordinatePayload):
    lat, lon = payload.lat, payload.lon

    try:
        # Current Weather
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        w_resp = requests.get(w_url, timeout=5).json()

        # Current Air Pollution
        a_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        a_resp = requests.get(a_url, timeout=5).json()

        # Past 24 Hours Air Pollution History for True Lags
        end_time = int(time.time())
        start_time = end_time - (25 * 3600)  # 25 hours back
        h_url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_time}&end={end_time}&appid={API_KEY}"
        h_resp = requests.get(h_url, timeout=5).json()

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

        return {
            "temp": float(w_resp["main"]["temp"]),
            "humidity": float(w_resp["main"]["humidity"]),
            "wind_speed": float(w_resp["wind"]["speed"]),
            "wind_deg": float(w_resp["wind"].get("deg", 180.0)),
            "pm2_5": current_pm25,
            "pm10": float(a_resp["list"][0]["components"]["pm10"]),
            "location_name": w_resp.get("name", "Grid Station"),
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
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"API Fetch Error: {str(e)}"
        )


@app.post("/predict")
def predict_air_quality(payload: SensorPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model file missing!")

    input_data = pd.DataFrame([payload.dict()])
    features = model.feature_name()
    X_infer = input_data[features]

    prediction = model.predict(X_infer)[0]
    return {"predicted_pm2_5": round(float(prediction), 2), "status": "SUCCESS"}