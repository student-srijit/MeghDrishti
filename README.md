# MeghDrishti

![Python](https://img.shields.io/badge/python-3.11-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB.svg?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6.svg?style=for-the-badge&logo=typescript&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![MapLibre](https://img.shields.io/badge/MapLibre-396CB2.svg?style=for-the-badge&logo=maplibre&logoColor=white)

<p align="center">
  <img src="logo.png" width="200px" alt="MeghDrishti Logo"/>
</p>

<p align="center">
  <b>Convective-scale nowcasting system (SIH 2026)</b><br/>
  Real-time 0-6h thunderstorm, hail, downburst and cloudburst prediction fusing IMD radar, INSAT satellite, and Blitzortung lightning data on a live GIS dashboard.
</p>

---

## Table of Contents

* [Overview](#overview)
* [Live Dashboard](#live-dashboard)
* [Architecture](#architecture)
* [Data Sources](#data-sources)
* [Features](#features)
* [Pipeline](#pipeline)
* [Hazard Detection](#hazard-detection)
* [Models](#models)
* [API Reference](#api-reference)
* [GIS Layers](#gis-layers)
* [SMS Alerting](#sms-alerting)
* [Technology Stack](#technology-stack)
* [Getting Started](#getting-started)
* [Configuration](#configuration)
* [Repository Structure](#repository-structure)
* [Limitations](#limitations)
* [License](#license)

---

## Overview

**MeghDrishti** ("Cloud Vision" in Sanskrit) is a full-stack, real-time convective storm nowcasting platform built for Smart India Hackathon 2026. It autonomously ingests multi-source geospatial data — IMD radar reflectivity, satellite thermal imagery, lightning strikes, and ECMWF weather grids — fuses them into a unified multi-channel raster, and runs rule-based hazard detection alongside two deep/statistical nowcasting models (pySTEPS and DGMR).

The system's **default hazard view detects real hail and lightning across all of India right now** — no fixed demo city, no synthetic storm — using live RainViewer radar data (IMD's own network, republished) and Blitzortung.org lightning strikes. A second per-region mode powers the Forecast and Replay pages with pySTEPS/DGMR model comparison across 10 major Indian cities.

---

## Live Dashboard

The React + TypeScript dashboard runs at `http://localhost:5173` and communicates with the FastAPI backend at `http://localhost:8000`.

<p align="center">
  <img src="assets/dashboard_hazards.png" width="48%" style="border-radius:10px; margin:1%;"/>
  <img src="assets/dashboard_forecast.png" width="48%" style="border-radius:10px; margin:1%;"/>
</p>
<p align="center">
  <img src="assets/dashboard_layers.png" width="48%" style="border-radius:10px; margin:1%;"/>
  <img src="assets/dashboard_replay.png" width="48%" style="border-radius:10px; margin:1%;"/>
</p>

---

## Architecture

```
                   +---------------------------------------+
                   |           Data Ingestion              |
                   |  +------------+  +-----------------+  |
                   |  | RainViewer |  | Blitzortung.org |  |
                   |  | (Radar dBZ)|  | (Lightning MQTT)|  |
                   |  +-----+------+  +-------+---------+  |
                   |  +-----+-----------------------------+ |
                   |  | Copernicus Sentinel-3 / EUMETSAT  | |
                   |  |        (Satellite IR TIR-1)       | |
                   |  +-----------------------------------+ |
                   |  +-----------------------------------+ |
                   |  |   ECMWF Open Data (No API Key)    | |
                   |  |  0.25 deg HRES, 4x/day, CC-BY-4  | |
                   |  +-----------------------------------+ |
                   +------------------+--------------------+
                                      |
                   +------------------v--------------------+
                   |            Fusion Engine              |
                   |   Multi-channel raster stack:         |
                   |   [tir1, wv, mwir, dBZ, lightning]   |
                   +---+---------------+---------------+---+
                       |               |               |
              +--------v------+ +------v------+ +------v------+
              | Rule-Based    | |  pySTEPS    | |    DGMR     |
              | Hazard Engine | | (LK Optical | | (DeepMind   |
              | Hail/Lightn.  | |  Flow+Extrap| |  Pretrained)|
              +--------+------+ +------+------+ +------+------+
                       |               |               |
                       +---------------v---------------+
                                       |
                       +---------------v-----------------+
                       |    FastAPI Backend (:8000)      |
                       |  /hazards  /forecast            |
                       |  /raw-layers  /weather-layers   |
                       |  /nowcast-frame  /storm-eta     |
                       +---------------+-----------------+
                                       |
                       +---------------v-----------------+
                       |  React + TypeScript (:5173)     |
                       |  MapLibre GL - Heatmap hazards  |
                       |  MOSDAC WMS - 23 GIS layers     |
                       |  pySTEPS vs DGMR toggle         |
                       +---------------------------------+
```

---

## Data Sources

MeghDrishti uses **five real, free data sources** as opt-in live paths (each behind a `USE_LIVE_*` flag in `.env`), falling back to synthetic mock data automatically if any source is unavailable:

| Layer | Real Source | API Key | Notes |
|---|---|---|---|
| **Radar Reflectivity** | RainViewer | None | IMD radar network, republished. Genuine greyscale-to-dBZ decode |
| **Lightning Strikes** | Blitzortung.org | None | Free community VLF network, MQTT-streamed |
| **Station Weather** | Tomorrow.io | Free tier | Temp/humidity/wind/precip at 10 demo stations |
| **Weather Grid** | ECMWF Open Data | None | 0.25 deg HRES, 4x/day, CC-BY-4.0 |
| **Satellite IR** | Copernicus Sentinel-3 SLSTR | Free CDSE | Verified live; polar orbit (~1-2 passes/day) |
| **Satellite IR (alt)** | EUMETSAT MSG SEVIRI | Free | Geostationary, 15min updates; wired, pending 403 fix |
| **Hail + Lightning (default)** | Real, all of India | None | `hazard_india.py`: RainViewer + Blitzortung, zero synthetic |

> **Radar velocity (downburst)** has no free public equivalent — no aggregator exposes raw Doppler volumetric scans — so it remains synthetic.

---

## Features

- **All-India Real Hazard View** — hail and lightning detected across India in real time via RainViewer + Blitzortung. Not scoped to a demo city.

- **Four Hazard Types** — hail (reflectivity + cold cloud top + lightning collocated), downburst (radial velocity couplet), cloudburst (pySTEPS rain rate >= 15 mm/hr), and lightning (IMD probability categories).

- **Multi-Model Nowcasting** — pySTEPS Lucas-Kanade optical flow (0-6h, calibrated mm/hr) and DeepMind DGMR (0-90min, real pretrained weights) with a live comparison toggle.

- **Multi-Source Fusion** — `[tir1, wv, mwir, reflectivity_dbz, lightning_prob]` stacked into one unified multi-channel raster per ingest cycle.

- **23 Real ISRO/MOSDAC GIS Layers** — 16 overlays (LULC, basins, drainage, landslide risk, rivers, roads, airports, district boundaries) + 7 base maps (Bhuvan, OSM, DEM, Natural Earth, Black Marble). Sourced live from MOSDAC CloudBurst DSS.

- **10 Selectable Demo Regions** — Pune, Delhi, Mumbai, Chennai, Kolkata, Bengaluru, Hyderabad, Ahmedabad, Jaipur, Guwahati — pre-warmed in the background so region switches are instant.

- **Storm-Arrival Countdown** — ETA derived from pySTEPS Lucas-Kanade motion field, not a separate model.

- **Weather Overlays** — real-time temperature, humidity, wind speed, pressure, and rain rate across the whole of India; lead-time animatable alongside the nowcast slider.

- **Convective Risk Index (P1)** — composite score overlaying cloudburst + hail + lightning contributions in a single magma-colormapped raster.

- **SMS Alerts** — Twilio-powered hazard alerts to configured numbers when a district hits the severity threshold, with per-district cooldown.

- **Background Pre-Warm** — all 10 region snapshots kept warm by a background thread so `/regions/{key}` switches apply instantly.

---

## Pipeline

End-to-end execution triggered on startup and every `INGEST_CYCLE_MINUTES` (default 15):

```
1. Ingestion (parallel, failure-isolated)
   +-- imd_nowcast.py         -> station temp/humidity/wind (Tomorrow.io or synthetic)
   +-- satellite_insat.py     -> satellite tir1 (Copernicus -> synthetic fallback)
   +-- radar_puller.py        -> radar reflectivity (RainViewer or synthetic Gaussian cell)

2. Lightning (independent of above)
   +-- blitzortung_lightning.py -> real VLF strikes over ~6s listen window

3. Fusion
   +-- fusion.py -> regrid + stack all channels into one multi-channel raster

4. Hazard Detection
   +-- hazard_india.py  -> all-India hail + lightning (default /hazards endpoint)
   +-- hazard.py        -> per-region hail/downburst on fused raster

5. Nowcasting
   +-- pysteps_baseline.py -> LK optical flow + semi-Lagrangian extrapolation
   +-- dgmr_nowcast.py     -> DeepMind DGMR (lazy-loaded on first /forecast?model=dgmr)

6. Weather Grid
   +-- weather_fields.py -> ECMWF Open Data regridded to India bbox (or synthetic)

7. Serve via FastAPI
   +-- api/main.py -> all endpoints, cached per cycle, CORS-enabled
```

---

## Hazard Detection

All hazard rules use physically motivated, documented thresholds from `nowcast/configs/settings.py`:

| Hazard | Rule | Threshold |
|---|---|---|
| **Hail** | Reflectivity AND TIR-1 AND lightning prob, all collocated | >= 55 dBZ, <= 210K, >= 0.30 |
| **Downburst** | Radial velocity delta (max-min in 5-cell window) | >= 25 m/s |
| **Cloudburst** | pySTEPS-extrapolated rain rate | >= 15 mm/hr (IMD "very heavy rain") |
| **Lightning** | IMD probability category | Cat11/Cat19 boundaries |

The all-India view uses `hazard_india.py` which runs hail and lightning rules over the full country-scale reflectivity grid — approximately 200x200 cells covering 68E-97.5E, 6.5N-37N.

Hazard points are **advected** at requested lead times using real ECMWF wind vectors (storms roughly follow the steering flow) — not re-detected at a future time.

---

## Models

### pySTEPS (Baseline, Section 4a)

Lucas-Kanade optical flow + semi-Lagrangian extrapolation on the synthetic reflectivity sequence. Produces **0-6h forecast at 10-minute steps** in calibrated mm/hr. Used to drive the cloudburst hazard rule and lead-time slider.

### DGMR (Stretch Goal, Section 4b)

DeepMind's pretrained **Skillful Precipitation Nowcasting GAN** (`openclimatefix/dgmr` on HuggingFace), run **zero-shot on CPU** (~3-11s/forecast). Real weights, no fine-tuning. Produces 0-90min relative intensity frames.

> **Important:** DGMR was trained on UK Met Office radar. Its output over Indian/synthetic input is **unitless 0-1 relative intensity**, not calibrated mm/hr. It is shown as a visual comparison layer only and never fed into the cloudburst hazard rule. This is the explicit domain-shift limitation the plan asks to disclose.

### SmaAt-UNet (Planned, Section 4b Option B)

Architecture implemented in `nowcast/models/smaat_unet.py` and training scaffold in `train_smaat.py`. Not yet fine-tuned on Indian radar data — DGMR zero-shot was built instead.

---

## API Reference

The FastAPI backend exposes the following endpoints at `http://localhost:8000`:

| Endpoint | Method | Description |
|---|---|---|
| `/hazards` | GET | Real hail + lightning across all India. `?lead_time=N` advects by ECMWF wind |
| `/hazards/region` | GET | Per-region demo: all 4 hazard types, pySTEPS-backed cloudburst |
| `/forecast` | GET | Nowcast summary. `?model=pysteps or dgmr or smaat` |
| `/nowcast-frame` | GET | Single PNG overlay frame. `?model=...&lead_time=N` |
| `/raw-layers` | GET | Satellite IR + radar reflectivity as base64 PNG overlays |
| `/weather-layers` | GET | Temp/humidity/wind/pressure/rainfall/risk grids. `?lead_time=N` |
| `/wind-vectors` | GET | Sparse wind arrow points for symbol rendering |
| `/storm-eta` | GET | Storm motion cells with bearing + speed from pySTEPS LK field |
| `/region-forecast` | GET | Point-sampled weather trend at `?lat=&lon=&lead_time=N` |
| `/regions` | GET | List selectable demo regions |
| `/regions/{key}` | POST | Switch active demo region (instant from pre-warm cache) |
| `/wms-proxy/bhuvan` | GET | CORS proxy for Bhuvan WMS |
| `/health` | GET | Health check |

Full interactive docs at `http://localhost:8000/docs`.

---

## GIS Layers

MeghDrishti integrates **23 genuine ISRO/NRSC/MOSDAC WMS layers**, sourced by driving MOSDAC's live CloudBurst DSS with a real browser and capturing each layer's actual WMS request:

**7 Base Maps:** Bhuvan Maps, OSM, DEM, LULC, Natural Earth, Black Marble, True Marble

**16 Overlay Layers:** LULC, River Basins, Drainage, Landslide Risk, Fire Risk, Rivers, Roads, Railways, Airports, Administrative Boundaries, Taluka Boundaries, District Population, and more

> Bhuvan's WMS server sends no CORS headers, so it is routed through a same-origin backend proxy at `/wms-proxy/bhuvan` — discovered and fixed during live browser verification with Playwright.

---

## SMS Alerting

When any district's real-time hazard rollup hits the configured severity threshold (`ALERT_MIN_SEVERITY`, default `"high"`), the system sends a Twilio SMS alert to all configured `ALERT_TO_NUMBERS`.

- Per-district cooldown (`ALERT_COOLDOWN_MINUTES`, default 60 min) prevents spam
- Alert send failures are isolated — a Twilio outage never affects hazard detection
- Configured via `.env`: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_TO_NUMBERS`

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend Runtime | Python 3.11 |
| API Framework | FastAPI + Uvicorn |
| Nowcasting (Baseline) | pySTEPS (Lucas-Kanade + Semi-Lagrangian) |
| Nowcasting (Deep) | DGMR (DeepMind, HuggingFace) |
| Radar Data | RainViewer (IMD network, greyscale-to-dBZ) |
| Lightning Data | Blitzortung.org (MQTT, community VLF) |
| Satellite Data | Copernicus Sentinel-3 SLSTR / EUMETSAT MSG |
| Weather Grid | ECMWF Open Data (0.25 deg HRES, free) |
| Station Weather | Tomorrow.io Realtime API |
| SMS Alerts | Twilio |
| Frontend | React 18 + TypeScript + Vite |
| Map Rendering | MapLibre GL JS |
| GIS Layers | ISRO/NRSC/MOSDAC WMS (23 layers) |
| Numerical | NumPy, SciPy, PyTorch |
| Visualization | Matplotlib, Pillow |

---

## Getting Started

### Prerequisites

- Python 3.11
- Node.js 18+
- A `.env` file (copy from `.env.example`)

### Backend

```bash
git clone https://github.com/AdityaShome/MeghDrishti.git
cd MeghDrishti

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env -- all USE_LIVE_* flags default to false (synthetic mode, no API calls)

# Start the backend
uvicorn nowcast.api.main:app --port 8000
```

> **Note:** Do not use `--reload` for normal use. The server writes timestamped data files to `nowcast/data/` on every ingest cycle, and uvicorn's reloader restarts the server on each write. If you need `--reload` while editing:
>
> ```bash
> uvicorn nowcast.api.main:app --port 8000 --reload --reload-exclude "nowcast/data/*"
> ```

### Frontend (Dashboard)

```bash
cd nowcast/dashboard
npm install
npm run dev
```

Opens at `http://localhost:5173` and proxies API calls to `http://localhost:8000`.

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`:

| Variable | Required | Description |
|---|---|---|
| `USE_LIVE_RADAR` | No | `true` to use real RainViewer reflectivity (no key needed) |
| `USE_LIVE_LIGHTNING` | No | `true` to use real Blitzortung.org lightning (no key needed) |
| `USE_LIVE_ECMWF` | No | `true` to use real ECMWF Open Data weather grid (no key needed) |
| `USE_LIVE_IMD` | No | `true` to use real Tomorrow.io station weather |
| `TOMORROW_API_KEY` | Conditional | Required if `USE_LIVE_IMD=true` |
| `USE_LIVE_SATELLITE` | No | `true` to use real Copernicus Sentinel-3 SLSTR |
| `COPERNICUS_CLIENT_ID` | Conditional | Required if `USE_LIVE_SATELLITE=true` |
| `COPERNICUS_CLIENT_SECRET` | Conditional | Required if `USE_LIVE_SATELLITE=true` |
| `EUMETSAT_CONSUMER_KEY` | Optional | EUMETSAT MSG SEVIRI (tried first if configured) |
| `EUMETSAT_CONSUMER_SECRET` | Optional | EUMETSAT MSG SEVIRI |
| `TWILIO_ACCOUNT_SID` | Optional | For SMS hazard alerts |
| `TWILIO_AUTH_TOKEN` | Optional | For SMS hazard alerts |
| `TWILIO_FROM_NUMBER` | Optional | Twilio-registered sender number |
| `ALERT_TO_NUMBERS` | Optional | Comma-separated recipient numbers |
| `ALERT_MIN_SEVERITY` | No | `low`, `moderate`, or `high` (default: `high`) |
| `ALERT_COOLDOWN_MINUTES` | No | Alert re-send cooldown per district (default: `60`) |
| `INGEST_CYCLE_MINUTES` | No | Background refresh cadence (default: `15`) |

> All `USE_LIVE_*` flags default to `false` — the system runs fully on synthetic data out of the box with no external network calls.

---

## Repository Structure

```text
.
+-- nowcast/
|   +-- alerts/
|   |   +-- sms_alerts.py               # Twilio SMS alert integration
|   +-- api/
|   |   +-- main.py                     # FastAPI entrypoint, all endpoints, caching
|   +-- configs/
|   |   +-- districts_india.py          # ~130 Indian district centroids + vectorized lookup
|   |   +-- settings.py                 # Hazard thresholds, regions, env flags
|   +-- dashboard/                      # React + TypeScript + Vite frontend
|   |   +-- src/
|   |   |   +-- components/             # HazardsPage, ForecastPage, ReplayPage, etc.
|   |   |   +-- map/                    # MapLibre GL layers (hazards, radar, WMS)
|   |   |   +-- lib/
|   |   |       +-- mosdacLayers.ts     # All 23 ISRO/MOSDAC WMS layer definitions
|   |   +-- legacy/
|   |       +-- index.html              # Original single-file HTML/JS (reference)
|   +-- ingestion/
|   |   +-- imd_nowcast.py              # Tomorrow.io station feed + Blitzortung
|   |   +-- blitzortung_lightning.py    # Real VLF lightning via MQTT
|   |   +-- rainviewer_radar.py         # Real all-India radar via RainViewer
|   |   +-- satellite_insat.py          # Satellite orchestrator (EUMETSAT -> Copernicus)
|   |   +-- eumetsat_satellite.py       # EUMETSAT MSG SEVIRI (geostationary)
|   |   +-- copernicus_satellite.py     # Copernicus Sentinel-3 SLSTR (verified live)
|   |   +-- ecmwf_weather.py            # ECMWF Open Data HRES (no key)
|   |   +-- radar_puller.py             # Radar orchestrator
|   +-- models/
|   |   +-- hazard.py                   # Per-region hail/downburst rules on fused raster
|   |   +-- hazard_india.py             # All-India hail+lightning on country-scale grid
|   |   +-- pysteps_baseline.py         # LK optical flow + semi-Lagrangian, cloudburst
|   |   +-- dgmr_nowcast.py             # DeepMind DGMR zero-shot (real pretrained weights)
|   |   +-- smaat_unet.py               # SmaAt-UNet architecture (not yet fine-tuned)
|   |   +-- train_smaat.py              # Training scaffold for SmaAt-UNet
|   |   +-- eta.py                      # Storm ETA from pySTEPS LK motion field
|   +-- processing/
|       +-- fusion.py                   # Multi-channel raster stack + rolling buffer
|       +-- storm_track.py              # Canonical synthetic storm trajectory
|       +-- weather_fields.py           # ECMWF regrid + synthetic ambient fields
|       +-- synthetic_radar.py          # Gaussian cell mock for fallback
|       +-- backtest.py                 # Historical replay framework
|       +-- load_historical_replay.py   # Historical data loader
+-- test_backend.py                     # Integration tests for all API endpoints
+-- requirements.txt
+-- .env.example
+-- WRITEUP.md                          # One-page technical write-up
+-- project.md                          # Full project plan (SIH 2026)
```

---

## Limitations

- **Radar velocity** (downburst): no public free source exposes raw Doppler volumetric scans. Downburst stays fully synthetic even with all live flags enabled.
- **DGMR calibration**: domain shift from UK Met Office training data means output is unitless relative intensity, not mm/hr. Never used for the cloudburst hazard rule.
- **Satellite revisit**: Copernicus Sentinel-3 is polar-orbiting (~1-2 passes/day); frequent "no recent scene over this bbox" fallbacks to synthetic are expected, not bugs.
- **Processing scale**: pySTEPS extrapolation runs on a ~64x64 demo bbox, degrading after ~2 simulated hours as the storm exits. The all-India hazard view runs at RainViewer mosaic resolution.
- **Hazard thresholds**: textbook meteorological values, never validated against a real Indian convective event.
- **SmaAt-UNet**: architecture implemented, training scaffold ready, but no fine-tuning on Indian/SEVIR radar data yet.

---

## License

MIT License. See [LICENSE](LICENSE) for more information.

---

<p align="center">Built for <b>Smart India Hackathon 2026</b></p>