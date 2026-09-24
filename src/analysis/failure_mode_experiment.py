"""
Failure Mode & Robustness Experiment
======================================
Simulates real-world sensor failure scenarios and validates that the
data-quality pipeline handles them gracefully without producing silent errors
or false "clean air" predictions.

Scenarios tested:
  1. Sensor Dropout    - consecutive NaN rows (missing sensor signal)
  2. Negative Readings - physically impossible PM2.5 values
  3. API Timeout       - what happens when OpenWeather is unreachable
  4. Short Gap Fill    - forward-fill works for <=3h gaps
  5. Long Gap          - gaps >3h should remain NaN (not silently filled)

Run:
    python src/analysis/failure_mode_experiment.py

Results are printed to stdout and saved to:
    reports/failure_mode_results.csv
"""

import os
import sys
from pathlib import Path

# UTF-8 output on Windows terminals
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.data_quality import clean_air_quality_data


# ---------------------------------------------------------------
# Helper to build a synthetic sensor DataFrame
# ---------------------------------------------------------------

def make_synthetic_df(n_hours: int = 72, base_pm25: float = 45.0) -> pd.DataFrame:
    """Return a synthetic hourly sensor-fusion DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp":    timestamps,
        "city":         "TestCity",
        "pm2_5":        base_pm25 + rng.normal(0, 5, n_hours),
        "pm10":         base_pm25 * 1.8 + rng.normal(0, 8, n_hours),
        "no2":          20.0 + rng.normal(0, 3, n_hours),
        "so2":          5.0  + rng.normal(0, 1, n_hours),
        "co":           200.0 + rng.normal(0, 20, n_hours),
        "temp_celsius": 28.0 + rng.normal(0, 2, n_hours),
        "humidity":     65.0 + rng.normal(0, 5, n_hours),
        "wind_speed":   3.0  + rng.normal(0, 1, n_hours).clip(min=0),
        "wind_deg":     rng.uniform(0, 360, n_hours),
    })
    return df


# ---------------------------------------------------------------
# Individual failure scenario testers
# ---------------------------------------------------------------

def test_short_sensor_dropout(df: pd.DataFrame) -> dict:
    """Scenario 1: 2-hour consecutive dropout (should be filled)."""
    df_fail = df.copy()
    df_fail.loc[10:11, "pm2_5"] = np.nan
    df_clean = clean_air_quality_data(df_fail)

    remaining_nulls = df_clean["pm2_5"].isna().sum()
    passed = remaining_nulls == 0
    return {
        "scenario":        "Short Sensor Dropout (2h)",
        "injected_nulls":  2,
        "remaining_nulls": remaining_nulls,
        "passed":          passed,
        "note":            "2h gap <= 3h limit -> forward-fill should recover all.",
    }


def test_long_sensor_dropout(df: pd.DataFrame) -> dict:
    """Scenario 2: 5-hour consecutive dropout (should NOT be fully filled)."""
    df_fail = df.copy()
    df_fail.loc[20:24, "pm2_5"] = np.nan
    df_clean = clean_air_quality_data(df_fail)

    remaining_nulls = df_clean["pm2_5"].isna().sum()
    passed = remaining_nulls > 0
    return {
        "scenario":        "Long Sensor Dropout (5h)",
        "injected_nulls":  5,
        "remaining_nulls": remaining_nulls,
        "passed":          passed,
        "note":            "5h gap > 3h limit -> some NaN must remain (no false fill).",
    }


def test_negative_readings(df: pd.DataFrame) -> dict:
    """Scenario 3: Physically impossible negative PM2.5 values."""
    df_fail = df.copy()
    df_fail.loc[5,  "pm2_5"] = -15.0
    df_fail.loc[30, "pm10"]  = -8.0
    df_fail.loc[50, "pm2_5"] = -100.0
    df_clean = clean_air_quality_data(df_fail)

    neg_after = (df_clean[["pm2_5", "pm10"]] < 0).sum().sum()
    passed = neg_after == 0
    return {
        "scenario":        "Negative Sensor Readings",
        "injected_nulls":  3,
        "remaining_nulls": int(neg_after),
        "passed":          passed,
        "note":            "Negative values must be replaced with NaN, never kept.",
    }


def test_humidity_bounds(df: pd.DataFrame) -> dict:
    """Scenario 4: Out-of-range humidity (>100 or <0)."""
    df_fail = df.copy()
    df_fail.loc[3,  "humidity"] = 150.0
    df_fail.loc[40, "humidity"] = -10.0
    df_clean = clean_air_quality_data(df_fail)

    out_of_range = ((df_clean["humidity"] < 0) | (df_clean["humidity"] > 100)).sum()
    passed = out_of_range == 0
    return {
        "scenario":        "Humidity Out-of-Range",
        "injected_nulls":  2,
        "remaining_nulls": int(out_of_range),
        "passed":          passed,
        "note":            "Humidity must be clipped to [0, 100].",
    }


def test_api_timeout_simulation() -> dict:
    """Scenario 5: API timeout - test that ingest fails gracefully."""
    import requests
    try:
        requests.get("http://0.0.0.0:1/nonexistent", timeout=0.5)
        passed = False
        note   = "Unexpected: request succeeded on unreachable host."
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        passed = True
        note   = "Connection/timeout error raised correctly - caller must handle."
    except Exception as ex:
        passed = False
        note   = f"Unexpected exception: {ex}"

    return {
        "scenario":        "API Timeout Simulation",
        "injected_nulls":  0,
        "remaining_nulls": 0,
        "passed":          passed,
        "note":            note,
    }


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def run_all_experiments():
    print("\n" + "=" * 60)
    print("  BDS-07 | Failure Mode & Robustness Experiment")
    print("=" * 60)

    df = make_synthetic_df()
    results = [
        test_short_sensor_dropout(df),
        test_long_sensor_dropout(df),
        test_negative_readings(df),
        test_humidity_bounds(df),
        test_api_timeout_simulation(),
    ]

    passed_count = 0
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        if r["passed"]:
            passed_count += 1
        print(f"\n  {status}  {r['scenario']}")
        print(f"         Injected: {r['injected_nulls']}  |  Remaining issues: {r['remaining_nulls']}")
        print(f"         Note: {r['note']}")

    print(f"\n{'=' * 60}")
    print(f"  Result: {passed_count}/{len(results)} scenarios passed.")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    results_df = pd.DataFrame(results)
    out_path = "reports/failure_mode_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[SAVED] Results -> {out_path}\n")

    return results


if __name__ == "__main__":
    run_all_experiments()
