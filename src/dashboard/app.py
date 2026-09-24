import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.analysis.geo_map import build_aqi_map
from src.analysis.trend_seasonality import (
    run_stl_decomposition,
    compute_summary_stats,
    plot_stl_decomposition,
    plot_diurnal_profile,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperlocal AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"]  { font-size: 1.6rem; font-weight: 700; }
  [data-testid="stMetricLabel"]  { font-size: 0.8rem; color: #aaa; }
  .block-container                { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🌫️ Hyperlocal AQI Forecasting Engine")
st.markdown("**BDS-07 | Sensor Fusion Pipeline** · LightGBM + SHAP + STL Decomposition")

API_BASE = "http://127.0.0.1:8000"

# ──────────────────────────────────────────────────────────────────────────────
# Micro-zone presets
# ──────────────────────────────────────────────────────────────────────────────
MICRO_ZONES = {
    "Custom GPS Coordinates":        {"lat": 19.0760, "lon": 72.8777},
    "Bandra West (Mumbai)":          {"lat": 19.0596, "lon": 72.8295},
    "Andheri East (Mumbai)":         {"lat": 19.1136, "lon": 72.8697},
    "BKC Commercial Zone (Mumbai)":  {"lat": 19.0657, "lon": 72.8683},
    "Colaba (South Mumbai)":         {"lat": 18.9067, "lon": 72.8147},
    "Majiwada Junction (Thane)":     {"lat": 19.2183, "lon": 72.9781},
    "Ghodbunder Road (Thane)":       {"lat": 19.2678, "lon": 72.9642},
    "Palghar Industrial Belt":       {"lat": 19.6966, "lon": 72.7699},
    "Virar West":                    {"lat": 19.4559, "lon": 72.8080},
    "Vasai East":                    {"lat": 19.3919, "lon": 72.8397},
}

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — location + fetch
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📍 Location Controls")

# Initialize coordinates in session state if missing
if "lat_val" not in st.session_state:
    st.session_state.lat_val = MICRO_ZONES["Virar West"]["lat"]
    st.session_state.lon_val = MICRO_ZONES["Virar West"]["lon"]
    st.session_state.preset_box = "Virar West"

def on_preset_change():
    zone = st.session_state.preset_box
    if zone in MICRO_ZONES:
        st.session_state.lat_val = MICRO_ZONES[zone]["lat"]
        st.session_state.lon_val = MICRO_ZONES[zone]["lon"]

selected_zone = st.sidebar.selectbox(
    "Select Micro-Zone Preset",
    list(MICRO_ZONES.keys()),
    index=list(MICRO_ZONES.keys()).index(st.session_state.preset_box) if st.session_state.preset_box in MICRO_ZONES else 0,
    key="preset_box",
    on_change=on_preset_change,
)

lat = st.sidebar.number_input("Latitude (°N)", key="lat_val", format="%.4f")
lon = st.sidebar.number_input("Longitude (°E)", key="lon_val", format="%.4f")

# Session-state defaults
if "lags" not in st.session_state:
    st.session_state.temp          = 28.5
    st.session_state.humidity      = 65.0
    st.session_state.pressure      = 1013.25
    st.session_state.wind_speed    = 5.2
    st.session_state.wind_deg      = 180.0
    st.session_state.pm2_5         = 45.0
    st.session_state.pm10          = 85.0
    st.session_state.no2           = 20.0
    st.session_state.so2           = 10.0
    st.session_state.co            = 500.0
    st.session_state.location_name = "Default Grid"
    st.session_state.lags = {
        "lag_1": 45.0, "lag_2": 44.0, "lag_3": 43.5,
        "lag_6": 42.0, "lag_12": 40.0, "lag_24": 38.0,
        "roll_mean_3h": 44.1, "roll_std_3h": 1.2,
        "roll_mean_6h": 43.0, "roll_std_6h": 2.1,
        "roll_mean_24h": 41.5, "roll_std_24h": 3.8,
    }

if st.sidebar.button("📡 Fetch Live Point Metrics"):
    try:
        res = requests.post(f"{API_BASE}/fetch_coords", json={"lat": lat, "lon": lon})
        if res.status_code == 200:
            data = res.json()
            st.session_state.temp          = float(data["temp"])
            st.session_state.humidity      = float(data["humidity"])
            st.session_state.pressure      = float(data.get("pressure", 1013.25))
            st.session_state.wind_speed    = float(data["wind_speed"])
            st.session_state.wind_deg      = float(data["wind_deg"])
            st.session_state.pm2_5         = float(data["pm2_5"])
            st.session_state.pm10          = float(data["pm10"])
            st.session_state.no2           = float(data.get("no2", 20.0))
            st.session_state.so2           = float(data.get("so2", 10.0))
            st.session_state.co            = float(data.get("co", 500.0))
            st.session_state.location_name = data["location_name"]
            st.session_state.lags          = data["lags"]
            st.sidebar.success(f"✅ Loaded live metrics for ({lat:.4f}, {lon:.4f})")
        else:
            detail = res.json().get("detail", "Unknown error") if res.headers.get("content-type", "").startswith("application/json") else res.text
            st.sidebar.error(f"Failed to fetch live data: {detail}")
    except Exception as e:
        st.sidebar.error(f"Backend error: {e}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Manual Adjustments")
temp_celsius = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, float(st.session_state.temp))
humidity     = st.sidebar.slider("Humidity (%)",      10.0, 100.0, float(st.session_state.humidity))
wind_speed   = st.sidebar.slider("Wind Speed (m/s)",  0.0,  30.0,  float(st.session_state.wind_speed))
pm2_5_lag_1  = st.sidebar.number_input("Current PM2.5 (Lag 1h)", value=float(st.session_state.pm2_5))

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Forecast & Explainability",
    "📈 Trend & Seasonality",
    "🗺️ Geospatial AQI Map",
    "📊 Model Performance",
])

def plot_local_shap_chart(shap_contribs, base_val, pred_val):
    import matplotlib.pyplot as plt

    top_items = list(shap_contribs[:9])
    top_items.reverse()  # ascending for horizontal bars

    feat_labels = {
        "pm10": "PM10 Coarse Dust",
        "temp_celsius": "Temperature",
        "humidity": "Humidity",
        "wind_speed": "Wind Speed",
        "wind_deg": "Wind Direction",
        "pressure": "Atmospheric Pressure",
        "pm2_5_lag_1": "PM2.5 (1h Lag)",
        "pm2_5_lag_2": "PM2.5 (2h Lag)",
        "pm2_5_lag_24": "PM2.5 (24h Lag)",
        "pm2_5_roll_mean_3h": "Rolling Mean 3h",
        "pm2_5_roll_mean_24h": "Rolling Mean 24h",
        "u_wind": "Zonal Wind (U)",
        "v_wind": "Meridional Wind (V)",
        "sin_hour": "Diurnal Cycle (sin)",
        "cos_hour": "Diurnal Cycle (cos)",
        "hour": "Hour of Day",
        "day_of_week": "Day of Week",
        "lat": "Latitude",
        "lon": "Longitude",
        "no2": "NO2 Concentration",
        "so2": "SO2 Concentration",
        "co": "CO Concentration",
    }

    names = [f"{feat_labels.get(item['feature'], item['feature'])} = {item['feature_value']}" for item in top_items]
    values = [item['shap_value'] for item in top_items]
    colors = ['#EF5350' if v >= 0 else '#26A69A' for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    bars = ax.barh(names, values, color=colors, height=0.55, edgecolor='#333', linewidth=0.8)
    ax.axvline(0, color='#888', linewidth=1.0, linestyle='--')

    max_abs = max([abs(v) for v in values] + [1.0])
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    for bar, val in zip(bars, values):
        xpos = val + (max_abs * 0.04 if val >= 0 else -max_abs * 0.04)
        ha = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}",
                va='center', ha=ha, color='white', fontsize=7.5, fontweight='bold')

    ax.set_title(f"Local SHAP Attribution: Baseline {base_val:.1f} -> Forecast {pred_val:.1f} ug/m3",
                 color='white', fontsize=9.5, fontweight='bold', pad=8)
    ax.tick_params(colors='#aaa', labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.set_xlabel("SHAP Impact (ug/m3)  [Green = Cleaner, Red = Dirtier]", color='#888', fontsize=7.5)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Forecast & Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Hyperlocal Forecast: {selected_zone}")
        st.caption(
            f"Coordinates: **{lat:.4f}°N, {lon:.4f}°E** | Station: **{st.session_state.location_name}**"
        )

        now         = pd.Timestamp.now()
        hour        = now.hour
        day_of_week = now.dayofweek
        rad         = np.radians(st.session_state.wind_deg)
        lags        = st.session_state.lags

        payload = {
            "lat":                float(lat),
            "lon":                float(lon),
            "temp_celsius":       float(temp_celsius),
            "humidity":           float(humidity),
            "pressure":           float(st.session_state.get("pressure", 1013.25)),
            "wind_speed":         float(wind_speed),
            "wind_deg":           float(st.session_state.wind_deg),
            "pm10":               float(st.session_state.pm10),
            "no2":                float(st.session_state.get("no2", 20.0)),
            "so2":                float(st.session_state.get("so2", 10.0)),
            "co":                 float(st.session_state.get("co", 500.0)),
            "pm2_5_lag_1":        float(pm2_5_lag_1),
            "pm2_5_lag_2":        float(lags["lag_2"]),
            "pm2_5_lag_3":        float(lags["lag_3"]),
            "pm2_5_lag_6":        float(lags["lag_6"]),
            "pm2_5_lag_12":       float(lags["lag_12"]),
            "pm2_5_lag_24":       float(lags["lag_24"]),
            "pm2_5_roll_mean_3h": float(lags["roll_mean_3h"]),
            "pm2_5_roll_std_3h":  float(lags["roll_std_3h"]),
            "pm2_5_roll_mean_6h": float(lags["roll_mean_6h"]),
            "pm2_5_roll_std_6h":  float(lags["roll_std_6h"]),
            "pm2_5_roll_mean_24h":float(lags["roll_mean_24h"]),
            "pm2_5_roll_std_24h": float(lags["roll_std_24h"]),
            "sin_hour":           float(np.sin(2 * np.pi * hour / 24.0)),
            "cos_hour":           float(np.cos(2 * np.pi * hour / 24.0)),
            "u_wind":             float(-wind_speed * np.sin(rad)),
            "v_wind":             float(-wind_speed * np.cos(rad)),
            "day_of_week":        int(day_of_week),
            "hour":               int(hour),
            "is_weekend":         1 if day_of_week >= 5 else 0,
        }

        # Auto-compute first forecast or run on button click
        run_prediction = st.button("🚀 Generate Hyperlocal Forecast") or ("latest_prediction" not in st.session_state)

        if run_prediction:
            try:
                res = requests.post(f"{API_BASE}/predict", json=payload)
                if res.status_code == 200:
                    resp_json = res.json()
                    pred = resp_json["predicted_pm2_5"]
                    st.session_state.latest_prediction = pred
                    st.session_state.latest_shap = resp_json.get("shap_contributions", [])
                    st.session_state.base_value = resp_json.get("base_value", 55.99)
                else:
                    detail = res.json().get("detail", "Unknown error") if res.headers.get("content-type", "").startswith("application/json") else res.text
                    st.error(f"API prediction error: {detail}")
            except Exception as e:
                st.error(f"Cannot connect to API server: {e}")

        # Display current forecast if available
        if "latest_prediction" in st.session_state:
            pred = st.session_state.latest_prediction
            st.metric(
                label=f"Forecasted PM2.5 for {selected_zone} (µg/m³)",
                value=f"{pred} µg/m³",
                delta=round(pred - pm2_5_lag_1, 2),
                delta_color="inverse",
            )

            # AQI health band
            if pred <= 30:
                st.success("🟢 **Good** — Safe for outdoor activities.")
            elif pred <= 60:
                st.info("🟡 **Satisfactory** — Minor discomfort to sensitive groups.")
            elif pred <= 90:
                st.warning("🟠 **Moderate** — Discomfort to people with lung/heart disease.")
            elif pred <= 120:
                st.error("🔴 **Poor** — Breathing discomfort on prolonged exposure.")
            else:
                st.error("🚨 **Severe** — Avoid outdoor exposure.")

            # CSV export
            report_df = pd.DataFrame([{
                "Timestamp":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Location":        st.session_state.location_name,
                "MicroZone":       selected_zone,
                "Latitude":        lat,
                "Longitude":       lon,
                "Predicted_PM2.5": pred,
                "Current_PM2.5":   pm2_5_lag_1,
                "Temperature_C":   temp_celsius,
                "Humidity_pct":    humidity,
                "Wind_Speed_ms":   wind_speed,
            }])
            st.download_button(
                "📥 Export Forecast Audit Report (CSV)",
                data=report_df.to_csv(index=False),
                file_name=f"AQI_Forecast_{int(pd.Timestamp.now().timestamp())}.csv",
                mime="text/csv",
            )

    with col2:
        st.subheader("Model Explainability (SHAP)")
        shap_mode = st.radio("Attribution Scope", ["Dynamic Local (This Location)", "Global Summary"], horizontal=True)

        if shap_mode == "Dynamic Local (This Location)":
            if st.session_state.get("latest_shap"):
                fig = plot_local_shap_chart(
                    st.session_state.latest_shap,
                    st.session_state.get("base_value", 55.99),
                    st.session_state.get("latest_prediction", pm2_5_lag_1),
                )
                st.pyplot(fig)
                st.caption("ℹ️ Local TreeSHAP reveals exactly which features pushed PM2.5 up (red) or pulled it down (green) for this exact micro-zone forecast.")
            else:
                st.info("Generating local SHAP attribution...")
        else:
            shap_img = "reports/figures/shap_summary.png"
            if os.path.exists(shap_img):
                st.image(shap_img, caption="TreeSHAP Global Feature Importances", use_container_width=True)
            else:
                st.info("SHAP plot not found. Run `python src/models/explainability.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trend & Seasonality
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 PM2.5 Trend & Seasonality Analysis (STL Decomposition)")
    st.markdown(
        "Uses **statsmodels STL** (Seasonal-Trend decomposition via LOESS) "
        "to separate PM2.5 into long-run trend, 24-hour diurnal seasonality, and residual noise."
    )

    data_path = "data/raw/historical_sensor_fusion.csv"

    if not os.path.exists(data_path):
        st.warning(
            "⚠️ Historical data not found at `data/raw/historical_sensor_fusion.csv`.\n\n"
            "Run `python src/data/ingest.py` to ingest data first."
        )
    else:
        df_hist = pd.read_csv(data_path, parse_dates=["timestamp"])
        cities  = df_hist["city"].unique().tolist() if "city" in df_hist.columns else []

        if not cities:
            st.error("No city column found in dataset.")
        else:
            selected_city = st.selectbox("Select City", cities, key="stl_city")
            group = (
                df_hist[df_hist["city"] == selected_city]
                .set_index("timestamp")["pm2_5"]
                .dropna()
                .resample("1h").mean()
                .ffill(limit=3)
                .dropna()
            )

            if len(group) < 48:
                st.warning(f"Not enough data for {selected_city} (need ≥ 48 hourly rows).")
            else:
                with st.spinner("Running STL decomposition..."):
                    try:
                        decomp = run_stl_decomposition(group, period=24)
                        stats  = compute_summary_stats(decomp)

                        # Summary metrics
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Mean PM2.5",      f"{stats['mean_pm25']} µg/m³")
                        c2.metric("Trend Range",      f"{stats['trend_range']} µg/m³")
                        c3.metric("Seasonal Amplitude", f"{stats['seasonal_amplitude']} µg/m³")
                        c4.metric("Residual Std",     f"{stats['residual_std']}")
                        c5.metric("Peak Pollution Hour", f"{stats['peak_hour']:02d}:00")

                        st.markdown("---")

                        # Generate and display plots
                        stl_path     = f"reports/figures/stl_{selected_city.lower()}.png"
                        diurnal_path = f"reports/figures/diurnal_{selected_city.lower()}.png"

                        plot_stl_decomposition(decomp, selected_city, stl_path)
                        plot_diurnal_profile(group, selected_city, diurnal_path)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(stl_path, caption=f"STL Components — {selected_city}", use_container_width=True)
                        with col_b:
                            st.image(diurnal_path, caption=f"Diurnal Profile — {selected_city}", use_container_width=True)

                    except Exception as e:
                        st.error(f"STL analysis failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Geospatial AQI Map
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🗺️ Geospatial AQI Map — Live City Pollution Levels")
    st.markdown(
        "Colour-coded markers scaled by PM2.5 concentration. "
        "Click any marker for detailed readings. Heatmap overlay shows pollution gradient."
    )

    # Build city data from session state (or last known values)
    CITY_COORDS = {
        "Delhi":     {"lat": 28.6139, "lon": 77.2090},
        "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    }

    data_path_raw = "data/raw/historical_sensor_fusion.csv"
    map_data = []

    if os.path.exists(data_path_raw):
        df_raw = pd.read_csv(data_path_raw, parse_dates=["timestamp"])
        # Get latest reading per city
        latest = (
            df_raw.sort_values("timestamp")
            .groupby("city")
            .last()
            .reset_index()
        )
        for _, row in latest.iterrows():
            city = row["city"]
            coords = CITY_COORDS.get(city, {"lat": row.get("lat", 20.0), "lon": row.get("lon", 78.0)})
            map_data.append({
                "city":     city,
                "lat":      coords["lat"],
                "lon":      coords["lon"],
                "pm2_5":    float(row.get("pm2_5", 0)),
                "temp":     round(float(row.get("temp_celsius", row.get("temp", 25))), 1),
                "humidity": float(row.get("humidity", 60)),
            })
    else:
        # Fallback placeholder data when no ingested data yet
        st.info("💡 No ingested data found. Showing sample placeholder readings. Run `python src/data/ingest.py` for live data.")
        map_data = [
            {"city": "Delhi",     "lat": 28.6139, "lon": 77.2090, "pm2_5": 95.0,  "temp": 32, "humidity": 55},
            {"city": "Mumbai",    "lat": 19.0760, "lon": 72.8777, "pm2_5": 48.0,  "temp": 29, "humidity": 78},
            {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "pm2_5": 32.5,  "temp": 24, "humidity": 65},
        ]

    # Add active micro-zone to map data so it updates dynamically with location changes
    active_entry = {
        "city": f"📍 {selected_zone} (Active)",
        "lat": float(lat),
        "lon": float(lon),
        "pm2_5": float(st.session_state.pm2_5),
        "predicted_pm2_5": st.session_state.get("latest_prediction", None),
        "temp": round(float(temp_celsius), 1),
        "humidity": round(float(humidity), 1),
        "is_selected": True,
    }
    map_data = [active_entry] + [d for d in map_data if "(Active)" not in d.get("city", "")]

    view_col1, _ = st.columns([2, 1])
    with view_col1:
        map_view_mode = st.radio(
            "Map Perspective",
            [f"🎯 Focus on Active Micro-Zone ({selected_zone})", "🇮🇳 Overview (All Monitored Cities)"],
            horizontal=True,
            key="map_perspective_toggle",
        )

    if "Focus on Active" in map_view_mode:
        center_coords = (float(lat), float(lon))
        map_zoom = 11
    else:
        center_coords = (20.5937, 78.9629)
        map_zoom = 5

    # Render the Folium map
    aqi_map = build_aqi_map(map_data, center=center_coords, zoom=map_zoom)
    st_folium(aqi_map, width=None, height=520, returned_objects=[])

    # City summary table
    st.markdown("#### Latest Readings by Location")
    if map_data:
        summary_df = pd.DataFrame(map_data)[["city", "pm2_5", "temp", "humidity"]]
        summary_df.columns = ["Location", "PM2.5 (µg/m³)", "Temp (°C)", "Humidity (%)"]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Model Evaluation & Baseline Comparison")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Algorithm",       "LightGBM")
    m2.metric("MAE",             "35.73 µg/m³")
    m3.metric("Tier 1 Baseline", "47.07 µg/m³",  delta="-24.1%", delta_color="inverse")
    m4.metric("Tier 2 Baseline", "~41 µg/m³",    delta="-~13%",  delta_color="inverse")
    m5.metric("Train/Test Split","80% / 20%")

    st.markdown("---")

    # Baseline comparison table
    st.markdown("#### Tiered Baseline Comparison")
    comparison_df = pd.DataFrame({
        "Model":       ["Tier 1 — Persistence", "Tier 2 — 24h Rolling Naive", "Tier 3 — LightGBM (Ours)"],
        "MAE (µg/m³)": ["47.07", "41.00 (approx)", "35.73"],
        "Approach":    [
            "Predict next hour = current hour (no learning)",
            "Predict next hour = rolling 24h mean (simple stats)",
            "LightGBM with lag features, wind vectors, diurnal encoding",
        ],
        "Improvement": ["—", "~13% vs Tier 1", "✅ 24.1% vs Tier 1"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Validation Strategy")
    st.markdown("""
- **Split Type:** Chronological 80/20 (no random shuffling — preserves time ordering)
- **Purge Gap:** 24-hour buffer between train and test set (eliminates temporal leakage through lag features)
- **Explainability:** TreeSHAP exact Shapley value attribution per feature
- **Experiment Tracking:** MLflow (SQLite backend) — all runs logged to `mlflow.db`
    """)

    st.markdown("#### Feature Engineering Summary")
    feat_df = pd.DataFrame({
        "Feature Group":   ["Wind Vectors", "Temporal Encoding", "Autoregressive Lags", "Rolling Statistics"],
        "Features":        ["u_wind, v_wind", "sin_hour, cos_hour, day_of_week, is_weekend",
                            "pm2_5_lag_1/2/3/6/12/24", "roll_mean/std (3h, 6h, 24h)"],
        "Rationale":       [
            "Converts circular wind direction to continuous Cartesian components",
            "Captures diurnal pollution cycles (rush hours, night dip)",
            "Provides temporal momentum — previous pollution predicts next",
            "Captures short/medium/long-run pollution trend momentum",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)