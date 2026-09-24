# Data Dictionary — BDS-07 Air Quality Dataset

## Raw Sensor Fusion Dataset (`data/raw/historical_sensor_fusion.csv`)

| Column | Type | Unit | Source | Description |
|--------|------|------|--------|-------------|
| `timestamp` | datetime | — | Derived | UTC timestamp of the sensor reading |
| `city` | string | — | Hardcoded | City name: Delhi, Mumbai, or Bengaluru |
| `lat` | float | °N | Hardcoded | Latitude of city centroid |
| `lon` | float | °E | Hardcoded | Longitude of city centroid |
| `pm2_5` | float | µg/m³ | OpenWeather Air Pollution API | Fine particulate matter ≤2.5µm — **primary forecast target** |
| `pm10` | float | µg/m³ | OpenWeather Air Pollution API | Coarse particulate matter ≤10µm |
| `no2` | float | µg/m³ | OpenWeather Air Pollution API | Nitrogen dioxide — traffic and combustion indicator |
| `so2` | float | µg/m³ | OpenWeather Air Pollution API | Sulphur dioxide — industrial emission indicator |
| `co` | float | µg/m³ | OpenWeather Air Pollution API | Carbon monoxide — incomplete combustion indicator |
| `temp_celsius` | float | °C | OpenWeather Weather API | Ambient temperature (converted from Kelvin) |
| `humidity` | float | % | OpenWeather Weather API | Relative humidity (0–100%) |
| `pressure` | float | hPa | OpenWeather Weather API | Atmospheric pressure |
| `wind_speed` | float | m/s | OpenWeather Weather API | Wind speed magnitude |
| `wind_deg` | float | ° | OpenWeather Weather API | Wind direction (meteorological — 0° = North, 90° = East) |

---

## Processed Features Dataset (`data/processed/processed_features.parquet`)

All raw columns are retained, plus the following engineered features:

### Temporal Features
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `hour` | int | 0–23 | Hour of day extracted from timestamp |
| `day_of_week` | int | 0–6 | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | int | 0 or 1 | 1 if Saturday or Sunday, else 0 |
| `sin_hour` | float | −1 to 1 | Sine encoding of hour — captures diurnal cycle continuity |
| `cos_hour` | float | −1 to 1 | Cosine encoding of hour — captures diurnal cycle continuity |

### Wind Vector Features
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `u_wind` | float | m/s | East-West wind component: `−wind_speed × sin(wind_deg_rad)` |
| `v_wind` | float | m/s | North-South wind component: `−wind_speed × cos(wind_deg_rad)` |

> Wind direction is converted to Cartesian (u, v) components to avoid the 0°/360° angular discontinuity that would confuse tree-based models.

### Autoregressive Lag Features
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `pm2_5_lag_1` | float | µg/m³ | PM2.5 reading 1 hour before current timestamp |
| `pm2_5_lag_2` | float | µg/m³ | PM2.5 reading 2 hours before |
| `pm2_5_lag_3` | float | µg/m³ | PM2.5 reading 3 hours before |
| `pm2_5_lag_6` | float | µg/m³ | PM2.5 reading 6 hours before |
| `pm2_5_lag_12` | float | µg/m³ | PM2.5 reading 12 hours before |
| `pm2_5_lag_24` | float | µg/m³ | PM2.5 reading 24 hours before (same time yesterday) |

### Rolling Statistics Features
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `pm2_5_roll_mean_3h` | float | µg/m³ | 3-hour rolling mean of PM2.5 (shifted by 1 to avoid leakage) |
| `pm2_5_roll_std_3h` | float | µg/m³ | 3-hour rolling standard deviation |
| `pm2_5_roll_mean_6h` | float | µg/m³ | 6-hour rolling mean |
| `pm2_5_roll_std_6h` | float | µg/m³ | 6-hour rolling standard deviation |
| `pm2_5_roll_mean_24h` | float | µg/m³ | 24-hour rolling mean |
| `pm2_5_roll_std_24h` | float | µg/m³ | 24-hour rolling standard deviation |

> All rolling statistics use `.shift(1)` to ensure they are computed using only past values — no leakage of the current observation into rolling windows.

---

## AQI Category Scale (India CPCB — PM2.5)

| PM2.5 (µg/m³) | Category | Health Implication |
|----------------|----------|-------------------|
| 0 – 30 | Good | Minimal impact |
| 30 – 60 | Satisfactory | Minor discomfort to sensitive people |
| 60 – 90 | Moderate | Discomfort to people with lung/heart disease |
| 90 – 120 | Poor | Breathing discomfort on prolonged exposure |
| 120 – 250 | Severe | Severe respiratory impact on general population |
| > 250 | Hazardous | Health emergency — avoid all outdoor exposure |

---

## Data Quality Rules Applied

| Rule | Implementation | File |
|------|---------------|------|
| Negative pollutant values → NaN | `x < 0 → np.nan` | `data_quality.py` |
| Humidity bounded to [0, 100] | `df.clip(0, 100)` | `data_quality.py` |
| Short sensor gaps (≤3h) → forward-fill | `.ffill(limit=3)` | `data_quality.py` |
| Long gaps (>3h) → remain NaN | No fill beyond limit | `data_quality.py` |
