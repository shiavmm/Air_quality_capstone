# User Stories & Product Backlog — BDS-07

## Personas

| Persona | Goal | Pain Point |
|---------|------|-----------|
| **Priya** — Daily commuter, Mumbai | Know if today's air is safe before her morning jog | Public dashboards show city-wide AQI, not her neighbourhood |
| **Dr. Mehta** — Urban planner, Delhi | Identify pollution hotspot hours to schedule road-work | No hyperlocal temporal data to justify policy decisions |
| **Riya** — Environmental researcher | Understand which atmospheric factors drive pollution spikes | Black-box models without explainability |

---

## Epic 1 — Hyperlocal Forecast

### US-01 — Live PM2.5 Prediction
> *As a citizen, I want to enter my GPS coordinates and get a PM2.5 forecast for the next hour, so I can decide whether to go outdoors.*

**Acceptance criteria:**
- [ ] User can enter latitude/longitude in the dashboard
- [ ] System fetches live weather + air data for those coordinates
- [ ] LightGBM model returns a PM2.5 prediction within 5 seconds
- [ ] Result displays AQI category (Good / Satisfactory / Moderate / Poor / Severe)

**Status:** ✅ Done

### US-02 — Health Advisory Alert
> *As a citizen with asthma, I want a colour-coded health advisory alongside the forecast, so I know what precautions to take.*

**Acceptance criteria:**
- [ ] Dashboard colour-codes result: green / yellow / orange / red / purple
- [ ] Advisory text is displayed in plain language
- [ ] Alert is shown immediately below the prediction metric

**Status:** ✅ Done

### US-03 — Audit Export
> *As a researcher, I want to download a timestamped forecast record as CSV, so I can track predictions over time.*

**Acceptance criteria:**
- [ ] Download button appears after each forecast
- [ ] CSV includes: timestamp, coordinates, prediction, input features

**Status:** ✅ Done

---

## Epic 2 — Trend & Seasonality Analysis

### US-04 — Diurnal Pattern Visualisation
> *As an urban planner, I want to see the average PM2.5 by hour of day for a city, so I can identify peak pollution windows.*

**Acceptance criteria:**
- [ ] Trend & Seasonality tab shows a diurnal profile chart
- [ ] AM and PM rush-hour windows are highlighted
- [ ] Peak hour metric is displayed

**Status:** ✅ Done (new tab)

### US-05 — STL Decomposition
> *As a researcher, I want to separate the long-run pollution trend from seasonal patterns and noise, so I can study each independently.*

**Acceptance criteria:**
- [ ] 4-panel STL plot (observed / trend / seasonal / residual)
- [ ] Seasonal amplitude metric reported
- [ ] Works per-city on historical data

**Status:** ✅ Done (new tab)

---

## Epic 3 — Geospatial Overview

### US-06 — City-Level AQI Map
> *As a planner, I want to see a map comparing PM2.5 across Delhi, Mumbai, and Bengaluru, so I can identify the most polluted city at a glance.*

**Acceptance criteria:**
- [ ] Interactive map with colour-coded markers per city
- [ ] Marker size scales with PM2.5 value
- [ ] Click marker to see detailed city readings
- [ ] Heatmap overlay shows pollution gradient

**Status:** ✅ Done (new tab)

---

## Epic 4 — Data & ML Infrastructure

### US-07 — Reproducible Training Pipeline
> *As a developer, I want to re-run the entire pipeline from raw data to trained model with a single sequence of commands, so anyone can reproduce the results.*

**Acceptance criteria:**
- [ ] `requirements.txt` lists all dependencies
- [ ] README documents step-by-step run instructions
- [ ] MLflow logs all experiment parameters and metrics

**Status:** ✅ Done

### US-08 — Failure-Safe Data Quality
> *As a system operator, I want the pipeline to handle sensor dropouts and bad readings gracefully, without producing false predictions.*

**Acceptance criteria:**
- [ ] Negative pollutant values → NaN
- [ ] Short gaps (≤3h) → forward-filled
- [ ] Long gaps (>3h) → remain NaN (no silent fill)
- [ ] Results documented in failure mode analysis

**Status:** ✅ Done

---

## Misuse / Abuse Cases

| Case | Risk | Mitigation |
|------|------|-----------|
| Medical decision-making based solely on model output | Prediction error could cause harm | Dashboard includes advisory disclaimer |
| API key leakage via repo commit | Unauthorised API usage, billing exposure | Key stored in `.env` (git-ignored); `.env.example` provided |
| Predicting for cities far outside training distribution | Silent model degradation | Dashboard notes model is trained on Delhi/Mumbai/Bengaluru only |
| Overriding lag inputs manually with unrealistic values | Garbage-in-garbage-out prediction | Input fields have reasonable slider/number bounds |

---

## Out of Scope

- Real-time sensor hardware integration (IoT)
- Predictions beyond 1-hour horizon
- Legal/regulatory air quality certification
- Mobile app deployment
- Forecasting for cities outside India
