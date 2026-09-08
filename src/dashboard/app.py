import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Hyperlocal AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ True Hyperlocal AQI Forecasting Engine")
st.markdown(
    "Micro-Zone Coordinate Ingestion (OpenWeather API + LightGBM Model)"
)

API_BASE = "http://127.0.0.1:8000"

MICRO_ZONES = {
    "Custom GPS Coordinates": {"lat": 19.0760, "lon": 72.8777},
    "Bandra West (Mumbai)": {"lat": 19.0596, "lon": 72.8295},
    "Andheri East (Mumbai)": {"lat": 19.1136, "lon": 72.8697},
    "BKC Commercial Zone (Mumbai)": {"lat": 19.0657, "lon": 72.8683},
    "Colaba (South Mumbai)": {"lat": 18.9067, "lon": 72.8147},
    "Majiwada Junction (Thane)": {"lat": 19.2183, "lon": 72.9781},
    "Ghodbunder Road (Thane)": {"lat": 19.2678, "lon": 72.9642},
    "Palghar Industrial Belt": {"lat": 19.6966, "lon": 72.7699},
    "Virar West": {"lat": 19.4559, "lon": 72.8080},
    "Vasai East": {"lat": 19.3919, "lon": 72.8397},
}

st.sidebar.header("📍 Hyperlocal Location Controls")
selected_zone = st.sidebar.selectbox(
    "Select Ward / Micro-Zone Preset", list(MICRO_ZONES.keys())
)

default_lat = MICRO_ZONES[selected_zone]["lat"]
default_lon = MICRO_ZONES[selected_zone]["lon"]

lat = st.sidebar.number_input(
    "Latitude (°N)", value=default_lat, format="%.4f"
)
lon = st.sidebar.number_input(
    "Longitude (°E)", value=default_lon, format="%.4f"
)

if "temp" not in st.session_state:
    st.session_state.temp = 28.5
    st.session_state.humidity = 65.0
    st.session_state.wind_speed = 5.2
    st.session_state.pm2_5 = 45.0
    st.session_state.pm10 = 85.0
    st.session_state.location_name = "Default Grid"

if st.sidebar.button("📡 Fetch Live Point Metrics"):
    try:
        res = requests.post(
            f"{API_BASE}/fetch_coords", json={"lat": lat, "lon": lon}
        )
        if res.status_code == 200:
            data = res.json()
            st.session_state.temp = float(data["temp"])
            st.session_state.humidity = float(data["humidity"])
            st.session_state.wind_speed = float(data["wind_speed"])
            st.session_state.pm2_5 = float(data["pm2_5"])
            st.session_state.pm10 = float(data["pm10"])
            st.session_state.location_name = data["location_name"]
            st.sidebar.success(f"Loaded live metrics for GPS ({lat}, {lon})!")
        else:
            st.sidebar.error("Failed to fetch live point data.")
    except Exception as e:
        st.sidebar.error(f"Backend connection error: {e}")

st.sidebar.header("🎛️ Manual Adjustments")
temp_celsius = st.sidebar.slider(
    "Temperature (°C)", 10.0, 50.0, float(st.session_state.temp)
)
humidity = st.sidebar.slider(
    "Humidity (%)", 10.0, 100.0, float(st.session_state.humidity)
)
wind_speed = st.sidebar.slider(
    "Wind Speed (m/s)", 0.0, 30.0, float(st.session_state.wind_speed)
)
pm2_5_lag_1 = st.sidebar.number_input(
    "Current PM2.5 (Lag 1h)", value=float(st.session_state.pm2_5)
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Hyperlocal Forecast: {selected_zone}")
    st.caption(
        f"Target Coordinates: **{lat}°N, {lon}°E** | Station Match: **{st.session_state.location_name}**"
    )

    now = pd.Timestamp.now()
    hour = now.hour
    day_of_week = now.dayofweek
    rad = np.radians(180.0)

    payload = {
        "temp_celsius": temp_celsius,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_deg": 180.0,
        "pm2_5_lag_1": pm2_5_lag_1,
        "pm2_5_lag_2": pm2_5_lag_1 * 0.98,
        "pm2_5_lag_3": pm2_5_lag_1 * 0.96,
        "pm2_5_lag_6": pm2_5_lag_1 * 0.95,
        "pm2_5_lag_12": pm2_5_lag_1 * 0.92,
        "pm2_5_lag_24": pm2_5_lag_1 * 0.90,
        "pm2_5_roll_mean_3h": pm2_5_lag_1,
        "pm2_5_roll_std_3h": 2.5,
        "pm2_5_roll_mean_6h": pm2_5_lag_1 * 0.97,
        "pm2_5_roll_std_6h": 3.1,
        "pm2_5_roll_mean_24h": pm2_5_lag_1 * 0.93,
        "pm2_5_roll_std_24h": 5.4,
        "sin_hour": float(np.sin(2 * np.pi * hour / 24.0)),
        "cos_hour": float(np.cos(2 * np.pi * hour / 24.0)),
        "u_wind": float(-wind_speed * np.sin(rad)),
        "v_wind": float(-wind_speed * np.cos(rad)),
        "day_of_week": day_of_week,
        "hour": hour,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "pm10": float(st.session_state.pm10),
    }

    if st.button("🚀 Generate Hyperlocal Forecast"):
        try:
            res = requests.post(f"{API_BASE}/predict", json=payload)
            if res.status_code == 200:
                pred = res.json()["predicted_pm2_5"]
                st.metric(
                    label="Forecasted Micro-Zone PM2.5 (µg/m³)",
                    value=f"{pred} µg/m³",
                    delta=round(pred - pm2_5_lag_1, 2),
                    delta_color="inverse",
                )
                st.success(
                    "Inference generated successfully for coordinates!"
                )
            else:
                st.error("API failed to process request.")
        except Exception as e:
            st.error(f"Cannot connect to API server: {e}")

with col2:
    st.subheader("Global Feature Attribution (SHAP)")
    shap_img_path = "reports/figures/shap_summary.png"
    if os.path.exists(shap_img_path):
        st.image(
            shap_img_path,
            caption="TreeSHAP Global Feature Importances",
            use_container_width=True,
        )