"""
Generate a synthetic 60-day historical dataset for model training.
Used when the OpenWeather historical API is unavailable or returns no data.

Run:
    python src/data/generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd

CITIES = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "base_pm25": 85.0, "base_temp": 33.0},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "base_pm25": 48.0, "base_temp": 29.0},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "base_pm25": 35.0, "base_temp": 24.0},
}

def generate_city_data(city: str, meta: dict, n_hours: int = 1440) -> pd.DataFrame:
    """Generate realistic synthetic hourly air quality data for one city."""
    rng = np.random.default_rng(hash(city) % (2**31))
    timestamps = pd.date_range("2024-04-01", periods=n_hours, freq="1h")
    hours = timestamps.hour.values

    # Diurnal PM2.5 pattern: higher at rush hours (7-9am, 5-8pm), lower at night
    diurnal = (
        8 * np.sin(2 * np.pi * (hours - 6) / 24) +
        5 * np.sin(2 * np.pi * (hours - 17) / 12)
    )

    # Long-run weekly trend + noise
    t = np.arange(n_hours)
    trend = 10 * np.sin(2 * np.pi * t / (24 * 7))
    noise = rng.normal(0, 6, n_hours)

    pm25 = np.clip(meta["base_pm25"] + diurnal + trend + noise, 2, 400)
    pm10 = np.clip(pm25 * 1.8 + rng.normal(0, 10, n_hours), 5, 600)
    no2  = np.clip(20 + 8 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 3, n_hours), 0, 200)
    so2  = np.clip(5 + rng.normal(0, 1.5, n_hours), 0, 50)
    co   = np.clip(200 + rng.normal(0, 30, n_hours), 0, 10000)
    temp = meta["base_temp"] + 5 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 1.5, n_hours)
    humidity = np.clip(65 + 15 * np.sin(2 * np.pi * (hours + 6) / 24) + rng.normal(0, 5, n_hours), 10, 100)
    pressure = 1013 + rng.normal(0, 3, n_hours)
    wind_speed = np.clip(3 + rng.exponential(2, n_hours), 0, 25)
    wind_deg = rng.uniform(0, 360, n_hours)

    return pd.DataFrame({
        "timestamp":    timestamps,
        "city":         city,
        "lat":          meta["lat"],
        "lon":          meta["lon"],
        "pm2_5":        np.round(pm25, 2),
        "pm10":         np.round(pm10, 2),
        "no2":          np.round(no2, 2),
        "so2":          np.round(so2, 2),
        "co":           np.round(co, 2),
        "temp_celsius": np.round(temp, 2),
        "humidity":     np.round(humidity, 2),
        "pressure":     np.round(pressure, 2),
        "wind_speed":   np.round(wind_speed, 2),
        "wind_deg":     np.round(wind_deg, 2),
    })


def generate_all(n_hours: int = 1440):
    """Generate and save synthetic dataset for all cities."""
    dfs = [generate_city_data(city, meta, n_hours) for city, meta in CITIES.items()]
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/historical_sensor_fusion.csv"
    df.to_csv(out, index=False)
    print(f"[SUCCESS] Generated {len(df)} rows ({n_hours}h x {len(CITIES)} cities) -> {out}")
    return df


if __name__ == "__main__":
    generate_all(n_hours=1440)   # 60 days of hourly data per city
