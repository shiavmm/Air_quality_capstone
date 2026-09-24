# Failure Mode Analysis — BDS-07 Air Quality System

## Purpose
Documents how the system behaves under real-world failure conditions. Required under the capstone acceptance gate: *"Must include at least one failure-mode, security or robustness experiment."*

---

## Experiment Setup

**Script:** `src/analysis/failure_mode_experiment.py`
**Output:** `reports/failure_mode_results.csv`

Run the experiment:
```bash
python src/analysis/failure_mode_experiment.py
```

---

## Scenario 1 — Short Sensor Dropout (2-hour gap)

**What was simulated:** Two consecutive PM2.5 readings set to NaN, simulating a 2-hour sensor communication failure.

**Expected behaviour:** Data quality gate forward-fills the gap (within the 3-hour limit).

**Result:** ✅ PASS — 2-hour gap fully recovered by `ffill(limit=3)`.

**Justification:** A 2-hour dropout is a common real-world scenario (network glitch, sensor restart). Forward-filling is acceptable because PM2.5 changes gradually over short periods.

---

## Scenario 2 — Long Sensor Dropout (5-hour gap)

**What was simulated:** Five consecutive PM2.5 readings set to NaN, simulating an extended sensor failure.

**Expected behaviour:** Forward-fill should **not** fill the entire gap. NaN values must remain for hours beyond the 3-hour limit.

**Result:** ✅ PASS — NaN values correctly remain after the 3-hour forward-fill limit.

**Importance:** Without this limit, a long sensor failure would silently propagate stale data, producing false "clean air" predictions. This is the core safety behaviour the data quality gate must enforce.

---

## Scenario 3 — Negative Sensor Readings

**What was simulated:** PM2.5 values of `-15.0`, `-100.0`, and PM10 of `-8.0` were injected.

**Expected behaviour:** All negative pollutant readings must be replaced with NaN (physically impossible values).

**Result:** ✅ PASS — No negative values remain after cleaning.

**Justification:** Negative pollutant concentrations are physically impossible. They indicate sensor malfunction or API data errors. Keeping them would corrupt all downstream features (lags, rolling stats, model input).

---

## Scenario 4 — Humidity Out-of-Range

**What was simulated:** Humidity values of `150%` and `-10%` were injected.

**Expected behaviour:** Humidity is clipped to valid range [0, 100].

**Result:** ✅ PASS — Both out-of-range values corrected.

---

## Scenario 5 — API Timeout / Unreachable Host

**What was simulated:** HTTP request to an unreachable host (`0.0.0.0:1`) with a 0.5-second timeout.

**Expected behaviour:** A `ConnectionError` or `Timeout` exception is raised — the system must not crash silently.

**Result:** ✅ PASS — Exception raised correctly. The FastAPI `/fetch_coords` endpoint wraps this in an HTTP 502 response with error detail.

**Mitigation:** The dashboard displays a user-facing error banner rather than an unhandled crash.

---

## Summary Table

| Scenario | Injected Failures | Remaining Issues | Result |
|----------|-------------------|------------------|--------|
| Short Sensor Dropout (2h) | 2 NaN | 0 | ✅ PASS |
| Long Sensor Dropout (5h) | 5 NaN | >0 (expected) | ✅ PASS |
| Negative Readings | 3 bad values | 0 | ✅ PASS |
| Humidity Out-of-Range | 2 bad values | 0 | ✅ PASS |
| API Timeout | 1 bad connection | Exception raised | ✅ PASS |

---

## Known Limitations

| Limitation | Risk | Mitigation |
|-----------|------|-----------|
| Lag features computed on filled data | Filled values propagate through lag windows | Limit fill to 3h; flag predictions made on filled inputs |
| No sensor confidence score | Cannot distinguish reliable vs unreliable readings | Integrate sensor QC flags from data source when available |
| API key expiry | All ingestion fails | Store key in `.env`; add key validity check on startup |
| Model drift | PM2.5 distribution shifts over seasons | Schedule periodic retraining via MLflow experiment tracking |
