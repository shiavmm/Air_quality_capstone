import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    page_title="Mumbai Air Quality Dashboard — BDS-07",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# AQI helpers
# ──────────────────────────────────────────────────────────────────────────────

AQI_BANDS = [
    (0,   30,  "#55a868", "Good",         "Air quality is satisfactory, and air pollution poses little or no risk."),
    (30,  60,  "#a7c957", "Satisfactory", "Air quality is acceptable. Minor concern for unusually sensitive people."),
    (60,  90,  "#f4a261", "Moderate",     "Members of sensitive groups may experience health effects."),
    (90,  120, "#e76f51", "Poor",         "Everyone may begin to experience health effects."),
    (120, 250, "#9b2226", "Severe",       "Health alert: everyone may experience serious health effects."),
    (250, 999, "#3d0814", "Hazardous",    "Health emergency. The entire population is likely to be affected."),
]

def get_aqi_info(pm25_val):
    """Return (color, label, description, band_index) for a PM2.5 value."""
    for i, (lo, hi, color, label, desc) in enumerate(AQI_BANDS):
        if lo <= pm25_val < hi:
            return color, label, desc, i
    return "#3d0814", "Hazardous", "Health emergency.", 5

def pm25_to_aqi_us(pm25):
    """Approximate US AQI from PM2.5 concentration using EPA breakpoints."""
    bp = [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    return min(500, round(pm25 * 1.5))

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Default light theme with layout enhancements
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ───── Typography ───── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1rem 2rem 3rem 2rem !important; max-width: 1400px; }

/* ───── Tabs ───── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #f0f2f6;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #e0e3e8;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #555;
    font-weight: 600;
    padding: 10px 20px;
    font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%) !important;
    color: white !important;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(26, 115, 232, 0.25);
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ───── Metric cards ───── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important; text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

div[data-testid="metric-container"] {
    background: #f8f9fb;
    border: 1px solid #e0e3e8;
    border-radius: 12px;
    padding: 16px 18px;
}

/* ───── Buttons ───── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a73e8, #4285f4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(26, 115, 232, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(26, 115, 232, 0.3) !important;
}

/* ───── DataFrame ───── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ───── Download button ───── */
.stDownloadButton > button {
    border-radius: 10px !important;
}

/* ───── Custom card classes ───── */
.pollutant-card {
    background: #f8f9fb;
    border: 1px solid #e0e3e8;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.pollutant-card:hover {
    border-color: #1a73e8;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}
.health-card {
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-subtext {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}
.aqi-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f44336;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(244,67,54,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(244,67,54,0); }
    100% { box-shadow: 0 0 0 0 rgba(244,67,54,0); }
}
.info-card {
    background: #f8f9fb;
    border: 1px solid #e0e3e8;
    border-radius: 16px;
    padding: 20px 24px 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size: 2rem;'>🌬️</span><br>
        <span style='font-size: 1.1rem; font-weight: 800; color: #1a73e8;'>AQI Engine</span><br>
        <span style='font-size: 0.7rem; color: #6b7280;'>BDS-07 · Hyperlocal Forecasting</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### 📍 Location Controls")

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
            st.session_state.data_source   = data.get("data_source", "owm")
            st.session_state.station_name  = data.get("station_name", "")
            st.session_state.lags          = data["lags"]
            source_label = f" — Station: {st.session_state.station_name}" if st.session_state.station_name else ""
            st.sidebar.success(f"✅ Loaded live metrics for ({lat:.4f}, {lon:.4f}){source_label}")
        else:
            detail = res.json().get("detail", "Unknown error") if res.headers.get("content-type", "").startswith("application/json") else res.text
            st.sidebar.error(f"Failed to fetch live data: {detail}")
    except Exception as e:
        st.sidebar.error(f"Backend error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🎛️ Manual Adjustments")
temp_celsius = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, float(st.session_state.temp))
humidity     = st.sidebar.slider("Humidity (%)",      10.0, 100.0, float(st.session_state.humidity))
wind_speed   = st.sidebar.slider("Wind Speed (m/s)",  0.0,  30.0,  float(st.session_state.wind_speed))
pm2_5_lag_1  = st.sidebar.number_input("Current PM2.5 (Lag 1h)", value=float(st.session_state.pm2_5))

# ──────────────────────────────────────────────────────────────────────────────
# Load historical data (used across multiple tabs)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/raw/historical_sensor_fusion.csv"
df_hist = None
if os.path.exists(DATA_PATH):
    df_hist = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# ══════════════════════════════════════════════════════════════════════════════
# HEADER — Site branding + live badge
# ══════════════════════════════════════════════════════════════════════════════
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:4px;'>
        <span style='font-size:2.5rem;'>🌬️</span>
        <div>
            <span style='font-size:1.8rem; font-weight:800; color:#1a1a2e;'>Air Quality Dashboard</span><br>
            <span style='font-size:0.85rem; color:#6b7280;'>BDS-07 · Hyperlocal AQI Forecasting Engine · Sensor Fusion Pipeline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with header_col2:
    st.markdown(f"""
    <div style='text-align:right; padding-top:12px;'>
        <span class='aqi-badge' style='background:rgba(26,115,232,0.1); color:#1a73e8;'>
            <span class='live-dot'></span> LIVE DATA
        </span><br>
        <span style='font-size:0.75rem; color:#6b7280; margin-top:6px; display:inline-block;'>
            Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard",
    "🚀 Forecast & SHAP",
    "📈 Trend & Seasonality",
    "🗺️ Geospatial Map",
    "📊 Model Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD (aqi.in-inspired)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Compute current AQI values ──
    current_pm25 = float(st.session_state.pm2_5)
    current_pm10 = float(st.session_state.pm10)
    current_no2  = float(st.session_state.get("no2", 20.0))
    current_so2  = float(st.session_state.get("so2", 10.0))
    current_co   = float(st.session_state.get("co", 500.0))
    current_temp = float(st.session_state.temp)
    current_hum  = float(st.session_state.humidity)
    current_wind = float(st.session_state.wind_speed)

    aqi_value = pm25_to_aqi_us(current_pm25)
    aqi_color, aqi_label, aqi_desc, aqi_band_idx = get_aqi_info(current_pm25)

    # ─────────────────────────────────────────────────
    # HERO CARD — Real-time Air Pollution Level
    # ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class='section-header'>
        <span class='live-dot'></span> Real-time Air Pollution Level
    </div>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([2, 1])
    with hero_left:
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {aqi_color}18 0%, {aqi_color}08 100%);
            border: 1px solid {aqi_color}40;
            border-radius: 20px;
            padding: 36px 40px;
            position: relative;
            overflow: hidden;
        '>
            <div style='display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:20px;'>
                <div>
                    <div style='font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>
                        {selected_zone} · Air Quality Index
                    </div>
                    <div style='font-size:5rem; font-weight:900; color:{aqi_color}; line-height:1; margin-bottom:8px;'>
                        {aqi_value}
                    </div>
                    <span class='aqi-badge' style='background:{aqi_color}22; color:{aqi_color};'>
                        {aqi_label.upper()}
                    </span>
                    <p style='color:#6b7280; margin-top:16px; font-size:0.88rem; max-width:420px; line-height:1.6;'>
                        {aqi_desc}
                    </p>
                </div>
                <div style='text-align:center; min-width:140px;'>
                    <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;'>PM2.5 Concentration</div>
                    <div style='font-size:3rem; font-weight:800; color:#1a1a2e;'>{current_pm25:.0f}</div>
                    <div style='font-size:0.85rem; color:#6b7280;'>µg/m³</div>
                </div>
            </div>
            <!-- AQI Scale Bar -->
            <div style='margin-top:28px;'>
                <div style='display:flex; border-radius:8px; overflow:hidden; height:10px;'>
                    <div style='flex:1; background:#55a868;'></div>
                    <div style='flex:1; background:#a7c957;'></div>
                    <div style='flex:1; background:#f4a261;'></div>
                    <div style='flex:1; background:#e76f51;'></div>
                    <div style='flex:1; background:#9b2226;'></div>
                    <div style='flex:1; background:#3d0814;'></div>
                </div>
                <div style='display:flex; justify-content:space-between; margin-top:6px; font-size:0.65rem; color:#6b7280;'>
                    <span>Good</span><span>Satisfactory</span><span>Moderate</span><span>Poor</span><span>Severe</span><span>Hazardous</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        # Weather info card
        st.markdown(f"""
        <div style='
            background: #f8f9fb;
            border: 1px solid #e0e3e8;
            border-radius: 20px;
            padding: 28px 24px;
            height: 100%;
        '>
            <div style='font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:20px;'>
                🌤️ Weather Conditions
            </div>
            <div style='display:flex; flex-direction:column; gap:18px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#6b7280;'>🌡️ Temperature</span>
                    <span style='font-weight:700; font-size:1.2rem; color:#1a1a2e;'>{current_temp:.1f}°C</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#6b7280;'>💧 Humidity</span>
                    <span style='font-weight:700; font-size:1.2rem; color:#1a1a2e;'>{current_hum:.0f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#6b7280;'>💨 Wind Speed</span>
                    <span style='font-weight:700; font-size:1.2rem; color:#1a1a2e;'>{current_wind:.1f} m/s</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#6b7280;'>🧭 Wind Dir</span>
                    <span style='font-weight:700; font-size:1.2rem; color:#1a1a2e;'>{float(st.session_state.wind_deg):.0f}°</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#6b7280;'>📊 Pressure</span>
                    <span style='font-weight:700; font-size:1.2rem; color:#1a1a2e;'>{float(st.session_state.pressure):.0f} hPa</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────
    # MAJOR AIR POLLUTANTS — Grid cards
    # ─────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>🧪 Major Air Pollutants</div>
    """, unsafe_allow_html=True)
    _ds = st.session_state.get("data_source", "owm")
    _sn = st.session_state.get("station_name", "")
    if _ds == "waqi+owm" and _sn:
        _src_text = f"Values are from <b>{_sn}</b> ground station (WAQI) fused with OpenWeatherMap weather data."
    else:
        _src_text = "Values are from the latest sensor fusion pipeline reading."
    st.markdown(f"""
    <div class='section-subtext'>
        Real-time concentrations of key air pollutants at the selected micro-zone.
        {_src_text}
    </div>
    """, unsafe_allow_html=True)

    pollutants = [
        ("PM2.5",  current_pm25, "µg/m³", "#1a73e8", "Fine particulate matter < 2.5µm. Penetrates lungs & bloodstream.", 60),
        ("PM10",   current_pm10, "µg/m³", "#2e7d32", "Inhalable coarse particles < 10µm. Causes respiratory issues.", 100),
        ("NO₂",    current_no2,  "µg/m³", "#e65100", "Nitrogen Dioxide. Traffic & combustion byproduct.", 40),
        ("SO₂",    current_so2,  "µg/m³", "#7b1fa2", "Sulfur Dioxide. Industrial emissions & fossil fuel burning.", 20),
        ("CO",     current_co,   "µg/m³", "#c62828", "Carbon Monoxide. Incomplete combustion product.", 2000),
    ]

    pcols = st.columns(5)
    for i, (name, val, unit, color, tooltip, threshold) in enumerate(pollutants):
        status = "🟢" if val < threshold * 0.5 else ("🟡" if val < threshold else "🔴")
        pct = min(100, (val / threshold) * 100)
        with pcols[i]:
            st.markdown(f"""
            <div class='pollutant-card'>
                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>
                    {name} {status}
                </div>
                <div style='font-size:2.2rem; font-weight:800; color:{color};'>{val:.1f}</div>
                <div style='font-size:0.75rem; color:#6b7280; margin-bottom:12px;'>{unit}</div>
                <div style='background:#e0e3e8; border-radius:4px; height:6px; overflow:hidden;'>
                    <div style='width:{pct:.0f}%; height:100%; background:{color}; border-radius:4px; transition: width 0.8s ease;'></div>
                </div>
                <div style='font-size:0.6rem; color:#6b7280; margin-top:6px;'>WHO Limit: {threshold} {unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────
    # HISTORICAL AQI GRAPH — 24h / 7d bar chart
    # ─────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>📊 AQI Graph — Historical Air Quality Data</div>
    <div class='section-subtext'>
        Visualize how air quality has changed over the past 24 hours and across the historical record.
        Data sourced from the sensor fusion pipeline.
    </div>
    """, unsafe_allow_html=True)

    if df_hist is not None:
        graph_city = st.selectbox("Select City for Historical Graph", df_hist["city"].unique().tolist(), key="hist_graph_city")
        df_city = df_hist[df_hist["city"] == graph_city].copy()
        df_city = df_city.set_index("timestamp").sort_index()

        graph_period = st.radio("Time Period", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"], horizontal=True, key="hist_period")

        now = pd.Timestamp.now()
        if graph_period == "Last 24 Hours":
            df_graph = df_city[df_city.index >= now - pd.Timedelta(hours=24)]
        elif graph_period == "Last 7 Days":
            df_graph = df_city[df_city.index >= now - pd.Timedelta(days=7)]
        elif graph_period == "Last 30 Days":
            df_graph = df_city[df_city.index >= now - pd.Timedelta(days=30)]
        else:
            df_graph = df_city

        # AQI bar chart with color coding
        df_graph_reset = df_graph.reset_index()
        df_graph_reset["aqi"] = df_graph_reset["pm2_5"].apply(pm25_to_aqi_us)
        df_graph_reset["color"] = df_graph_reset["pm2_5"].apply(lambda v: get_aqi_info(v)[0])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=df_graph_reset["timestamp"],
            y=df_graph_reset["aqi"],
            marker_color=df_graph_reset["color"].tolist(),
            hovertemplate="<b>%{x}</b><br>AQI: %{y}<br>PM2.5: %{customdata:.1f} µg/m³<extra></extra>",
            customdata=df_graph_reset["pm2_5"],
        ))

        # Add AQI band horizontal lines
        for lo, hi, color, label, _ in AQI_BANDS[:4]:
            aqi_lo = pm25_to_aqi_us(lo) if lo > 0 else 0
            aqi_hi = pm25_to_aqi_us(hi)
            fig_hist.add_hline(y=aqi_hi, line_dash="dot", line_color=color, opacity=0.4,
                               annotation_text=label, annotation_font_color=color, annotation_font_size=10)

        fig_hist.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=50, r=20, t=40, b=50),
            xaxis=dict(title="", gridcolor="#f0f0f0"),
            yaxis=dict(title="AQI (US EPA)", gridcolor="#f0f0f0"),
            font=dict(family="Inter", color="#333"),
            title=dict(text=f"Air Quality Index — {graph_city}", font=dict(size=16, color="#1a1a2e")),
            bargap=0.15,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("💡 Historical data not found. Run `python src/data/ingest.py` to ingest sensor data.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────
    # AQI TRENDS — Annual Air Quality Changes
    # ─────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>📈 AQI Trends — Annual Air Quality Changes</div>
    <div class='section-subtext'>
        Compare average pollutant concentrations across cities and observe seasonal patterns.
        Identify long-term trends in air quality degradation or improvement.
    </div>
    """, unsafe_allow_html=True)

    if df_hist is not None:
        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            # City-wise comparison bar chart
            city_avg = df_hist.groupby("city")[["pm2_5", "pm10", "no2", "so2"]].mean().round(1)

            fig_compare = go.Figure()
            colors_city = ["#1a73e8", "#2e7d32", "#e65100"]
            for i, city in enumerate(city_avg.index):
                fig_compare.add_trace(go.Bar(
                    name=city,
                    x=city_avg.columns.tolist(),
                    y=city_avg.loc[city].values.tolist(),
                    marker_color=colors_city[i % len(colors_city)],
                    marker_line=dict(width=0),
                ))

            fig_compare.update_layout(
                template="plotly_white",
                height=350,
                margin=dict(l=50, r=20, t=40, b=50),
                barmode="group",
                title=dict(text="Average Pollutant Levels by City", font=dict(size=14, color="#1a1a2e")),
                xaxis=dict(gridcolor="#f0f0f0"),
                yaxis=dict(title="µg/m³", gridcolor="#f0f0f0"),
                font=dict(family="Inter", color="#333"),
                legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with trend_col2:
            # Daily trend line chart
            daily_avg = df_hist.groupby([df_hist["timestamp"].dt.date, "city"])["pm2_5"].mean().reset_index()
            daily_avg.columns = ["date", "city", "pm2_5"]
            daily_avg["date"] = pd.to_datetime(daily_avg["date"])

            fig_daily = go.Figure()
            for i, city in enumerate(daily_avg["city"].unique()):
                city_data = daily_avg[daily_avg["city"] == city]
                fig_daily.add_trace(go.Scatter(
                    x=city_data["date"],
                    y=city_data["pm2_5"],
                    name=city,
                    mode="lines+markers",
                    line=dict(width=2.5, color=colors_city[i % len(colors_city)]),
                    marker=dict(size=4),
                    fill="tonexty" if i > 0 else "tozeroy",
                    fillcolor=f"rgba({','.join(str(int(colors_city[i % len(colors_city)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.08)",
                ))

            fig_daily.update_layout(
                template="plotly_white",
                height=350,
                margin=dict(l=50, r=20, t=40, b=50),
                title=dict(text="Daily PM2.5 Trend", font=dict(size=14, color="#1a1a2e")),
                xaxis=dict(title="", gridcolor="#f0f0f0"),
                yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="#f0f0f0"),
                font=dict(family="Inter", color="#333"),
                legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_daily, use_container_width=True)

        # Monthly heatmap
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        df_hist_copy = df_hist.copy()
        df_hist_copy["day"] = df_hist_copy["timestamp"].dt.day
        df_hist_copy["month"] = df_hist_copy["timestamp"].dt.strftime("%b %Y")
        df_hist_copy["hour"] = df_hist_copy["timestamp"].dt.hour

        heatmap_city = st.selectbox("Select City for Heatmap", df_hist["city"].unique().tolist(), key="heatmap_city")
        heatmap_data = df_hist_copy[df_hist_copy["city"] == heatmap_city].groupby(["day", "hour"])["pm2_5"].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="hour", columns="day", values="pm2_5")

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"Day {d}" for d in heatmap_pivot.columns],
            y=[f"{h:02d}:00" for h in heatmap_pivot.index],
            colorscale=[
                [0, "#e8f5e9"],
                [0.2, "#a5d6a7"],
                [0.4, "#fff9c4"],
                [0.6, "#ffcc80"],
                [0.8, "#ef9a9a"],
                [1.0, "#c62828"],
            ],
            colorbar=dict(title="PM2.5"),
            hovertemplate="Day %{x}<br>Hour: %{y}<br>PM2.5: %{z:.1f} µg/m³<extra></extra>",
        ))
        fig_heat.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=60, r=20, t=50, b=50),
            title=dict(text=f"Pollution Heatmap — {heatmap_city} (Hour vs Day)", font=dict(size=14, color="#1a1a2e")),
            xaxis=dict(title="Day of Month", gridcolor="#f0f0f0"),
            yaxis=dict(title="Hour of Day", gridcolor="#f0f0f0", autorange="reversed"),
            font=dict(family="Inter", color="#333"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("💡 Historical data not found. Run `python src/data/ingest.py` to ingest sensor data.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────
    # HEALTH ADVISORY
    # ─────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>🏥 Health Advisory</div>
    <div class='section-subtext'>Recommended actions based on current air quality level.</div>
    """, unsafe_allow_html=True)

    health_col1, health_col2, health_col3 = st.columns(3)

    with health_col1:
        st.markdown(f"""
        <div class='health-card' style='background: rgba(46,125,50,0.06); border: 1px solid rgba(46,125,50,0.2);'>
            <div style='font-size:1.5rem; margin-bottom:8px;'>😷</div>
            <div style='font-weight:700; color:#2e7d32; margin-bottom:6px;'>General Population</div>
            <div style='font-size:0.82rem; color:#6b7280; line-height:1.6;'>
                {"Enjoy outdoor activities! Air quality is safe." if aqi_band_idx <= 1 else
                 "Reduce prolonged outdoor exertion. Consider wearing a mask outdoors." if aqi_band_idx <= 3 else
                 "Avoid outdoor activities. Use air purifiers indoors. Wear N95 masks if going out."}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with health_col2:
        st.markdown(f"""
        <div class='health-card' style='background: rgba(230,81,0,0.06); border: 1px solid rgba(230,81,0,0.2);'>
            <div style='font-size:1.5rem; margin-bottom:8px;'>🫁</div>
            <div style='font-weight:700; color:#e65100; margin-bottom:6px;'>Respiratory Conditions</div>
            <div style='font-size:0.82rem; color:#6b7280; line-height:1.6;'>
                {"No special precautions needed." if aqi_band_idx <= 0 else
                 "Keep rescue inhaler handy. Monitor symptoms closely." if aqi_band_idx <= 2 else
                 "Stay indoors with windows closed. Run air purifier. Contact doctor if symptoms worsen."}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with health_col3:
        st.markdown(f"""
        <div class='health-card' style='background: rgba(198,40,40,0.06); border: 1px solid rgba(198,40,40,0.2);'>
            <div style='font-size:1.5rem; margin-bottom:8px;'>👶</div>
            <div style='font-weight:700; color:#c62828; margin-bottom:6px;'>Children & Elderly</div>
            <div style='font-size:0.82rem; color:#6b7280; line-height:1.6;'>
                {"Safe for all outdoor activities including sports." if aqi_band_idx <= 1 else
                 "Limit outdoor play time. Avoid strenuous outdoor activities." if aqi_band_idx <= 3 else
                 "Keep children & elderly indoors at all times. Ensure proper ventilation with purification."}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Forecast & Explainability
# ══════════════════════════════════════════════════════════════════════════════
def plot_local_shap_chart(shap_contribs, base_val, pred_val):
    """Create an interactive Plotly SHAP waterfall-style chart."""
    top_items = list(shap_contribs[:9])
    top_items.reverse()

    feat_labels = {
        "pm10": "PM10 Coarse Dust", "temp_celsius": "Temperature", "humidity": "Humidity",
        "wind_speed": "Wind Speed", "wind_deg": "Wind Direction", "pressure": "Atmospheric Pressure",
        "pm2_5_lag_1": "PM2.5 (1h Lag)", "pm2_5_lag_2": "PM2.5 (2h Lag)", "pm2_5_lag_24": "PM2.5 (24h Lag)",
        "pm2_5_roll_mean_3h": "Rolling Mean 3h", "pm2_5_roll_mean_24h": "Rolling Mean 24h",
        "u_wind": "Zonal Wind (U)", "v_wind": "Meridional Wind (V)",
        "sin_hour": "Diurnal Cycle (sin)", "cos_hour": "Diurnal Cycle (cos)",
        "hour": "Hour of Day", "day_of_week": "Day of Week",
        "lat": "Latitude", "lon": "Longitude",
        "no2": "NO₂ Concentration", "so2": "SO₂ Concentration", "co": "CO Concentration",
    }

    names = [f"{feat_labels.get(item['feature'], item['feature'])} = {item['feature_value']}" for item in top_items]
    values = [item['shap_value'] for item in top_items]
    colors = ['#c62828' if v >= 0 else '#2e7d32' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#333"),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#999")
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=200, r=60, t=50, b=40),
        title=dict(
            text=f"SHAP: Baseline {base_val:.1f} → Forecast {pred_val:.1f} µg/m³",
            font=dict(size=14, color="#1a1a2e"),
        ),
        xaxis=dict(title="SHAP Impact (µg/m³)", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
        font=dict(family="Inter", color="#333"),
    )
    return fig


with tab2:
    st.markdown("""
    <div class='section-header'>🚀 Hyperlocal Forecast & Explainability</div>
    <div class='section-subtext'>
        Generate a real-time PM2.5 forecast for any micro-zone using our LightGBM model,
        then inspect per-feature SHAP attribution to understand what drives the prediction.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f"""
        <div class='info-card'>
            <span style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px;'>
                Forecasting for
            </span><br>
            <span style='font-size:1.2rem; font-weight:700; color:#1a73e8;'>{selected_zone}</span><br>
            <span style='font-size:0.8rem; color:#6b7280;'>
                {lat:.4f}°N, {lon:.4f}°E · {st.session_state.location_name}
            </span>
        </div>
        """, unsafe_allow_html=True)

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

        run_prediction = st.button("🚀 Generate Hyperlocal Forecast", type="primary") or ("latest_prediction" not in st.session_state)

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

        if "latest_prediction" in st.session_state:
            pred = st.session_state.latest_prediction
            pred_aqi = pm25_to_aqi_us(pred)
            pred_color, pred_label, _, _ = get_aqi_info(pred)

            fcst_cols = st.columns(3)
            with fcst_cols[0]:
                st.metric(
                    label="Forecasted PM2.5",
                    value=f"{pred:.1f} µg/m³",
                    delta=f"{pred - pm2_5_lag_1:+.1f} µg/m³",
                    delta_color="inverse",
                )
            with fcst_cols[1]:
                st.metric(label="Forecasted AQI", value=str(pred_aqi))
            with fcst_cols[2]:
                st.metric(label="Category", value=pred_label)

            # Health band
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
        st.markdown("""
        <div class='info-card'>
            <span style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px;'>
                Model Explainability
            </span><br>
            <span style='font-size:1.2rem; font-weight:700; color:#1a1a2e;'>SHAP Feature Attribution</span>
        </div>
        """, unsafe_allow_html=True)

        shap_mode = st.radio("Attribution Scope", ["Dynamic Local (This Location)", "Global Summary"], horizontal=True)

        if shap_mode == "Dynamic Local (This Location)":
            if st.session_state.get("latest_shap"):
                fig = plot_local_shap_chart(
                    st.session_state.latest_shap,
                    st.session_state.get("base_value", 55.99),
                    st.session_state.get("latest_prediction", pm2_5_lag_1),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("ℹ️ Local TreeSHAP reveals which features pushed PM2.5 up (🔴) or pulled it down (🟢) for this forecast.")
            else:
                st.info("Run a forecast to see local SHAP attribution.")
        else:
            shap_img = "reports/figures/shap_summary.png"
            if os.path.exists(shap_img):
                st.image(shap_img, caption="TreeSHAP Global Feature Importances", use_container_width=True)
            else:
                st.info("SHAP plot not found. Run `python src/models/explainability.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trend & Seasonality
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-header'>📈 PM2.5 Trend & Seasonality Analysis</div>
    <div class='section-subtext'>
        Uses <b>statsmodels STL</b> (Seasonal-Trend decomposition via LOESS) to separate PM2.5 into
        long-run trend, 24-hour diurnal seasonality, and residual noise.
    </div>
    """, unsafe_allow_html=True)

    if df_hist is None:
        st.warning(
            "⚠️ Historical data not found at `data/raw/historical_sensor_fusion.csv`.\n\n"
            "Run `python src/data/ingest.py` to ingest data first."
        )
    else:
        cities = df_hist["city"].unique().tolist()

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
# TAB 4 — Geospatial AQI Map
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='section-header'>🗺️ Geospatial AQI Map</div>
    <div class='section-subtext'>
        Colour-coded markers scaled by PM2.5 concentration. Click any marker for detailed readings.
        Heatmap overlay shows pollution gradient across monitored locations.
    </div>
    """, unsafe_allow_html=True)

    CITY_COORDS = {
        "Delhi":     {"lat": 28.6139, "lon": 77.2090},
        "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    }

    data_path_raw = "data/raw/historical_sensor_fusion.csv"
    map_data = []

    if os.path.exists(data_path_raw):
        df_raw = pd.read_csv(data_path_raw, parse_dates=["timestamp"])
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
        st.info("💡 No ingested data found. Showing sample placeholder readings.")
        map_data = [
            {"city": "Delhi",     "lat": 28.6139, "lon": 77.2090, "pm2_5": 95.0,  "temp": 32, "humidity": 55},
            {"city": "Mumbai",    "lat": 19.0760, "lon": 72.8777, "pm2_5": 48.0,  "temp": 29, "humidity": 78},
            {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "pm2_5": 32.5,  "temp": 24, "humidity": 65},
        ]

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

    aqi_map = build_aqi_map(map_data, center=center_coords, zoom=map_zoom)
    st_folium(aqi_map, width=None, height=520, returned_objects=[])

    st.markdown("#### Latest Readings by Location")
    if map_data:
        summary_df = pd.DataFrame(map_data)[["city", "pm2_5", "temp", "humidity"]]
        summary_df.columns = ["Location", "PM2.5 (µg/m³)", "Temp (°C)", "Humidity (%)"]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class='section-header'>📊 Model Evaluation & Baseline Comparison</div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Algorithm",       "LightGBM")
    m2.metric("MAE",             "35.73 µg/m³")
    m3.metric("Tier 1 Baseline", "47.07 µg/m³",  delta="-24.1%", delta_color="inverse")
    m4.metric("Tier 2 Baseline", "~41 µg/m³",    delta="-~13%",  delta_color="inverse")
    m5.metric("Train/Test Split","80% / 20%")

    st.markdown("---")

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