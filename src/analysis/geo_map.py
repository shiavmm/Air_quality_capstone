"""
Module: Geospatial AQI Map Builder
====================================
Builds a Folium interactive map showing real-time / latest PM2.5 levels
for monitored cities as colour-coded circle markers with popups.

Used by:
  - src/dashboard/app.py  (embedded in Streamlit via streamlit-folium)
  - Can be run standalone to generate a static HTML map
"""

import os
import sys
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

sys.path.append(str(Path(__file__).resolve().parents[2]))

# AQI category thresholds (PM2.5 µg/m³) — India CPCB scale
AQI_BANDS = [
    (0,   30,  "#00e676", "Good"),
    (30,  60,  "#ffee58", "Satisfactory"),
    (60,  90,  "#ff9800", "Moderate"),
    (90,  120, "#f44336", "Poor"),
    (120, 250, "#7b1fa2", "Severe"),
    (250, 9999,"#212121", "Hazardous"),
]


def pm25_to_color(pm25: float) -> tuple[str, str]:
    """Return (hex_color, category_label) for a given PM2.5 value."""
    for lo, hi, color, label in AQI_BANDS:
        if lo <= pm25 < hi:
            return color, label
    return "#212121", "Hazardous"


def build_aqi_map(
    city_data: list[dict],
    center: tuple[float, float] = (20.5937, 78.9629),  # centre of India
    zoom: int = 5,
) -> folium.Map:
    """
    Build an interactive Folium map from a list of city AQI readings.

    Parameters
    ----------
    city_data : list of dicts, each with keys:
        city, lat, lon, pm2_5, predicted_pm2_5 (optional), temp, humidity
    center : (lat, lon) for initial map centre
    zoom   : initial zoom level

    Returns
    -------
    folium.Map object (render with st_folium in Streamlit)
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )

    heat_data = []

    for row in city_data:
        pm25   = row.get("pm2_5", 0.0)
        pred   = row.get("predicted_pm2_5")
        color, category = pm25_to_color(pm25)
        city   = row.get("city", "Unknown")
        lat    = row["lat"]
        lon    = row["lon"]

        # Popup HTML
        pred_line = (
            f"<b>Predicted PM2.5:</b> {pred:.1f} µg/m³<br>"
            if pred is not None else ""
        )
        popup_html = f"""
        <div style='font-family:sans-serif; min-width:180px'>
            <h4 style='margin:0 0 6px 0; color:{color}'>{city}</h4>
            <b>Live PM2.5:</b> {pm25:.1f} µg/m³<br>
            {pred_line}
            <b>Category:</b>
            <span style='color:{color}; font-weight:bold'>{category}</span><br>
            <b>Temp:</b> {row.get('temp', '—')}°C &nbsp;
            <b>Humidity:</b> {row.get('humidity', '—')}%
        </div>
        """

        is_selected = row.get("is_selected", False)
        marker_color = "#00E5FF" if is_selected else color
        border_color = "#FFFFFF" if is_selected else color
        border_weight = 3 if is_selected else 1

        if is_selected:
            # Outer halo for the active micro-zone
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(18, min(pm25 / 4, 45)) + 8,
                color="#00E5FF",
                weight=2,
                fill=False,
                dash_array="5, 5",
            ).add_to(m)

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(14, min(pm25 / 5, 35)),   # radius scales with pollution
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85 if is_selected else 0.75,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{city}: {pm25:.1f} µg/m³ — {category} {'(ACTIVE ZONE)' if is_selected else ''}",
        ).add_to(m)

        # Label marker
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""<div style='
                    font-size:10px; font-weight:bold; color:white;
                    text-shadow:1px 1px 2px black;
                    white-space:nowrap;'>{city}</div>""",
                icon_size=(100, 20),
                icon_anchor=(50, -8),
            ),
        ).add_to(m)

        heat_data.append([lat, lon, pm25])

    # Optional heatmap overlay
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=40,
            blur=25,
            gradient={"0.2": "blue", "0.5": "yellow", "0.8": "orange", "1.0": "red"},
        ).add_to(m)

    # AQI legend
    legend_html = """
    <div style='
        position:fixed; bottom:30px; left:30px; z-index:1000;
        background:rgba(15,17,23,0.9); padding:12px 16px;
        border-radius:8px; border:1px solid #333;
        font-family:sans-serif; font-size:12px; color:white'>
        <b>PM2.5 AQI Scale (India CPCB)</b><br>
        <span style='color:#00e676'>●</span> Good (0–30)<br>
        <span style='color:#ffee58'>●</span> Satisfactory (30–60)<br>
        <span style='color:#ff9800'>●</span> Moderate (60–90)<br>
        <span style='color:#f44336'>●</span> Poor (90–120)<br>
        <span style='color:#7b1fa2'>●</span> Severe (120–250)<br>
        <span style='color:#212121; background:#ccc'>●</span> Hazardous (250+)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def save_static_map(city_data: list[dict], out_path: str = "reports/aqi_map.html"):
    """Save a standalone HTML map file."""
    m = build_aqi_map(city_data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    print(f"[GEO MAP] Saved static HTML map → {out_path}")


if __name__ == "__main__":
    # Demo with static sample data
    sample = [
        {"city": "Delhi",     "lat": 28.6139, "lon": 77.2090, "pm2_5": 95.0,  "temp": 32, "humidity": 55},
        {"city": "Mumbai",    "lat": 19.0760, "lon": 72.8777, "pm2_5": 48.0,  "temp": 29, "humidity": 78},
        {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "pm2_5": 32.5,  "temp": 24, "humidity": 65},
    ]
    save_static_map(sample)
