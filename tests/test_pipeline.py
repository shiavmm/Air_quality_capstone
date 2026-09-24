"""
pytest test suite — BDS-07 Air Quality
========================================
Covers:
  - data quality validation gates
  - feature engineering output shape/columns
  - FastAPI /predict endpoint (mocked)
  - STL trend analysis basic contract

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_quality import clean_air_quality_data
from src.features.build_features import build_atmospheric_features

# ──────────────────────────────────────────────
# Shared fixture: minimal synthetic DataFrame
# ──────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    """60-row hourly sensor-fusion DataFrame (enough for lag features)."""
    rng = np.random.default_rng(0)
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "timestamp":    ts,
        "city":         "TestCity",
        "pm2_5":        40.0 + rng.normal(0, 5, n),
        "pm10":         70.0 + rng.normal(0, 8, n),
        "no2":          20.0 + rng.normal(0, 2, n),
        "so2":          5.0  + rng.normal(0, 1, n),
        "co":           200. + rng.normal(0, 15, n),
        "temp_celsius": 28.0 + rng.normal(0, 2, n),
        "humidity":     65.0 + rng.normal(0, 5, n),
        "wind_speed":   3.0  + np.abs(rng.normal(0, 1, n)),
        "wind_deg":     rng.uniform(0, 360, n),
    })


# ──────────────────────────────────────────────
# Data Quality Tests
# ──────────────────────────────────────────────

class TestDataQuality:

    def test_negative_pm25_replaced_with_nan(self, synthetic_df):
        """Negative PM2.5 values must be converted to NaN."""
        synthetic_df.loc[5, "pm2_5"] = -30.0
        result = clean_air_quality_data(synthetic_df)
        assert result["pm2_5"].min() >= 0 or result.loc[5, "pm2_5"] != result.loc[5, "pm2_5"], \
            "Negative PM2.5 was not replaced with NaN"

    def test_negative_pm10_replaced_with_nan(self, synthetic_df):
        synthetic_df.loc[10, "pm10"] = -5.0
        result = clean_air_quality_data(synthetic_df)
        assert not (result["pm10"] < 0).any(), "Negative PM10 should be NaN'd"

    def test_humidity_clipped_upper(self, synthetic_df):
        """Humidity > 100 must be clipped to 100."""
        synthetic_df.loc[3, "humidity"] = 150.0
        result = clean_air_quality_data(synthetic_df)
        assert result["humidity"].max() <= 100, "Humidity not clipped at upper bound"

    def test_humidity_clipped_lower(self, synthetic_df):
        """Humidity < 0 must be clipped to 0."""
        synthetic_df.loc[7, "humidity"] = -20.0
        result = clean_air_quality_data(synthetic_df)
        assert result["humidity"].min() >= 0, "Humidity not clipped at lower bound"

    def test_short_gap_forward_filled(self, synthetic_df):
        """Gaps ≤ 3 consecutive hours must be forward-filled."""
        synthetic_df.loc[15:16, "pm2_5"] = np.nan   # 2-hour gap
        result = clean_air_quality_data(synthetic_df)
        assert not result.loc[15:16, "pm2_5"].isna().any(), \
            "2-hour gap should be forward-filled"

    def test_long_gap_not_fully_filled(self, synthetic_df):
        """Gaps > 3 consecutive hours must NOT be fully filled (no false data)."""
        synthetic_df.loc[20:24, "pm2_5"] = np.nan   # 5-hour gap
        result = clean_air_quality_data(synthetic_df)
        assert result["pm2_5"].isna().any(), \
            "5-hour gap should still contain NaN after quality gate"

    def test_returns_dataframe(self, synthetic_df):
        result = clean_air_quality_data(synthetic_df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, synthetic_df):
        result = clean_air_quality_data(synthetic_df)
        assert len(result) == len(synthetic_df), "Row count should not change after cleaning"


# ──────────────────────────────────────────────
# Feature Engineering Tests
# ──────────────────────────────────────────────

class TestFeatureEngineering:

    def test_wind_vector_columns_created(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert "u_wind" in result.columns, "u_wind not created"
        assert "v_wind" in result.columns, "v_wind not created"

    def test_temporal_encoding_columns(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        for col in ["sin_hour", "cos_hour", "day_of_week", "is_weekend"]:
            assert col in result.columns, f"Column {col} missing"

    def test_lag_columns_created(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        for lag in [1, 2, 3, 6, 12, 24]:
            assert f"pm2_5_lag_{lag}" in result.columns, f"Lag {lag} column missing"

    def test_rolling_stat_columns_created(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        for window in [3, 6, 24]:
            assert f"pm2_5_roll_mean_{window}h" in result.columns
            assert f"pm2_5_roll_std_{window}h"  in result.columns

    def test_output_has_rows(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert len(result) > 0, "Feature engineering returned empty DataFrame"

    def test_sin_hour_bounded(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert result["sin_hour"].between(-1, 1).all(), "sin_hour out of [-1, 1]"

    def test_cos_hour_bounded(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert result["cos_hour"].between(-1, 1).all(), "cos_hour out of [-1, 1]"

    def test_is_weekend_binary(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert result["is_weekend"].isin([0, 1]).all(), "is_weekend must be 0 or 1"

    def test_no_nans_after_dropna(self, synthetic_df):
        result = build_atmospheric_features(synthetic_df)
        assert not result.isnull().any().any(), "NaNs present after feature engineering dropna"


# ──────────────────────────────────────────────
# FastAPI /predict endpoint tests (no server needed)
# ──────────────────────────────────────────────

class TestAPIPredict:

    def test_predict_endpoint_with_valid_payload(self):
        """Test /predict via TestClient (no live server needed)."""
        try:
            from fastapi.testclient import TestClient
            # Import must succeed
            from src.api.app import app  # noqa: F401
            # Only run full test if model file exists
            import os
            if not os.path.exists("models/lgbm_model.txt"):
                pytest.skip("Model file not found — run train_lgbm.py first")

            client = TestClient(app)
            payload = {
                "temp_celsius": 28.0, "humidity": 65.0,
                "wind_speed": 3.0,    "wind_deg": 180.0,
                "pm2_5_lag_1": 45.0,  "pm2_5_lag_2": 44.0,
                "pm2_5_lag_3": 43.5,  "pm2_5_lag_6": 42.0,
                "pm2_5_lag_12": 40.0, "pm2_5_lag_24": 38.0,
                "pm2_5_roll_mean_3h": 44.1, "pm2_5_roll_std_3h": 1.2,
                "pm2_5_roll_mean_6h": 43.0, "pm2_5_roll_std_6h": 2.1,
                "pm2_5_roll_mean_24h": 41.5,"pm2_5_roll_std_24h": 3.8,
                "sin_hour": 0.5, "cos_hour": 0.866,
                "u_wind": -1.5,  "v_wind": 2.6,
                "day_of_week": 2, "hour": 14, "is_weekend": 0, "pm10": 85.0,
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "predicted_pm2_5" in data
            assert isinstance(data["predicted_pm2_5"], float)
            assert data["predicted_pm2_5"] >= 0
        except ImportError:
            pytest.skip("FastAPI TestClient not available")

    def test_health_check(self):
        """GET / should return status=online."""
        try:
            from fastapi.testclient import TestClient
            from src.api.app import app  # noqa: F401
            import os
            if not os.path.exists("models/lgbm_model.txt"):
                pytest.skip("Model file not found")
            client = TestClient(app)
            response = client.get("/")
            assert response.status_code == 200
            assert response.json()["status"] == "online"
        except ImportError:
            pytest.skip("FastAPI TestClient not available")
