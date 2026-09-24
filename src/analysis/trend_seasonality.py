# -*- coding: utf-8 -*-
"""
Module: Trend & Seasonality Analysis
=====================================
Uses statsmodels STL (Seasonal-Trend decomposition using LOESS) to decompose
historical PM2.5 time-series into:
  - Trend component     -> long-run pollution trajectory
  - Seasonal component  -> diurnal (24-hour) cycle (rush hour, night dip)
  - Residual component  -> unexplained variation / anomalies

Run standalone:
    python src/analysis/trend_seasonality.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

sys.path.append(str(Path(__file__).resolve().parents[2]))


# ──────────────────────────────────────────────
# Core analysis functions
# ──────────────────────────────────────────────

def run_stl_decomposition(series: pd.Series, period: int = 24) -> dict:
    """
    Fit STL decomposition on a PM2.5 hourly time-series.

    Parameters
    ----------
    series : pd.Series  (index = DatetimeIndex, values = PM2.5 µg/m³)
    period : int        seasonal period in hours (24 = diurnal cycle)

    Returns
    -------
    dict with keys: trend, seasonal, resid, original
    """
    # STL requires at least 2 full periods
    if len(series) < 2 * period:
        raise ValueError(
            f"Series too short ({len(series)} rows) for STL with period={period}. "
            f"Need at least {2 * period} rows."
        )

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    return {
        "original": series,
        "trend":    pd.Series(result.trend,    index=series.index),
        "seasonal": pd.Series(result.seasonal, index=series.index),
        "resid":    pd.Series(result.resid,    index=series.index),
    }


def compute_summary_stats(decomp: dict) -> dict:
    """Return key summary statistics from a decomposition result."""
    trend    = decomp["trend"]
    seasonal = decomp["seasonal"]
    resid    = decomp["resid"]
    original = decomp["original"]

    return {
        "mean_pm25":             round(float(original.mean()), 2),
        "trend_range":           round(float(trend.max() - trend.min()), 2),
        "seasonal_amplitude":    round(float(seasonal.max() - seasonal.min()), 2),
        "residual_std":          round(float(resid.std()), 2),
        "peak_hour":             int(seasonal.groupby(seasonal.index.hour).mean().idxmax()),
        "trough_hour":           int(seasonal.groupby(seasonal.index.hour).mean().idxmin()),
    }


def plot_stl_decomposition(decomp: dict, city: str, out_path: str) -> None:
    """Save a 4-panel STL decomposition figure to disk."""
    fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
    fig.suptitle(
        f"PM2.5 STL Decomposition - {city}",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 1, hspace=0.55)
    panels = [
        ("original", "Observed PM2.5 (µg/m³)", "#4FC3F7"),
        ("trend",    "Trend Component",          "#81C784"),
        ("seasonal", "Seasonal Component (24h)", "#FFD54F"),
        ("resid",    "Residual",                 "#EF9A9A"),
    ]

    for i, (key, label, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#1a1d27")
        ax.plot(decomp[key].index, decomp[key].values, color=color, linewidth=0.9)
        ax.set_ylabel(label, color="white", fontsize=8)
        ax.tick_params(colors="grey", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if i == 0:
            ax.fill_between(
                decomp[key].index, decomp[key].values, alpha=0.15, color=color
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[STL] Figure saved -> {out_path}")


def plot_diurnal_profile(series: pd.Series, city: str, out_path: str) -> None:
    """Save an average diurnal (hour-of-day) PM2.5 profile."""
    hourly_mean = series.groupby(series.index.hour).mean()
    hourly_std  = series.groupby(series.index.hour).std()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(hourly_mean.index, hourly_mean.values, color="#4FC3F7", linewidth=2, marker="o", markersize=4)
    ax.fill_between(
        hourly_mean.index,
        hourly_mean - hourly_std,
        hourly_mean + hourly_std,
        alpha=0.2, color="#4FC3F7"
    )
    ax.axvspan(7, 10, alpha=0.1, color="#FFD54F", label="AM Rush")
    ax.axvspan(17, 20, alpha=0.1, color="#EF9A9A", label="PM Rush")
    ax.set_title(f"Average Diurnal PM2.5 Profile - {city}", color="white", fontsize=12)
    ax.set_xlabel("Hour of Day", color="grey")
    ax.set_ylabel("Mean PM2.5 (µg/m³)", color="grey")
    ax.tick_params(colors="grey")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Diurnal] Figure saved -> {out_path}")


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

def run_full_analysis(data_path: str = "data/raw/historical_sensor_fusion.csv"):
    """Run STL + diurnal analysis for each city in the dataset."""
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("  -> Run 'python src/data/ingest.py' first to collect data.")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    results = {}
    for city, group in df.groupby("city"):
        print(f"\n── Analysing {city} ({len(group)} records) ──")
        group = group.set_index("timestamp")["pm2_5"].dropna()

        # Resample to hourly (forward-fill gaps up to 3h)
        group = group.resample("1h").mean().ffill(limit=3).dropna()

        if len(group) < 48:
            print(f"  [SKIP] Not enough data for {city} (need ≥48 hourly rows).")
            continue

        try:
            decomp = run_stl_decomposition(group, period=24)
            stats  = compute_summary_stats(decomp)
            results[city] = {"decomp": decomp, "stats": stats}

            print(f"  Mean PM2.5      : {stats['mean_pm25']} µg/m³")
            print(f"  Trend range     : {stats['trend_range']} µg/m³")
            print(f"  Seasonal amp    : {stats['seasonal_amplitude']} µg/m³")
            print(f"  Residual std    : {stats['residual_std']}")
            print(f"  Peak hour       : {stats['peak_hour']:02d}:00")
            print(f"  Trough hour     : {stats['trough_hour']:02d}:00")

            # Save figures
            plot_stl_decomposition(
                decomp, city,
                f"reports/figures/stl_{city.lower().replace(' ', '_')}.png"
            )
            plot_diurnal_profile(
                group, city,
                f"reports/figures/diurnal_{city.lower().replace(' ', '_')}.png"
            )

        except ValueError as e:
            print(f"  [SKIP] {e}")

    return results


if __name__ == "__main__":
    run_full_analysis()
