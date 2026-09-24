"""FastAPI service (section 5a).

Serves hazard GeoJSON and storm ETA computed from the latest ingestion
cycle. Inference is cached per file-write, recomputed only when a new
ingestion snapshot lands (not on every request) — matches the "cached, not
per-call" requirement in the plan.
"""
import base64
import glob
import io
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from nowcast.configs.settings import (
    IMD_DIR,
    INGEST_CYCLE_MINUTES,
    CLOUDBURST_RAIN_RATE_MM_HR,
    REGIONS,
    ALERT_MIN_SEVERITY,
    ALERT_COOLDOWN_MINUTES,
    get_active_region_key,
    get_region_name,
    set_active_region,
    override_active_region,
)
from nowcast.configs.districts_india import DISTRICTS
from nowcast.ingestion.imd_nowcast import pull as pull_imd
from nowcast.ingestion.satellite_insat import pull as pull_satellite
from nowcast.ingestion.radar_puller import pull as pull_radar
from nowcast.models.hazard import classify_station, hail_cells, downburst_cells
from nowcast.models import hazard_india
from nowcast.models.eta import storm_cells
from nowcast.models.pysteps_baseline import run_forecast, cloudburst_cells
from nowcast.processing.fusion import build_fused_frame, list_imd_timestamps, build_fused_frame_for_timestamp
from nowcast.processing import weather_fields
from nowcast.alerts import sms_alerts

_SEVERITY_RANK = {"low": 0, "moderate": 1, "high": 2}
_DISTRICT_CENTROIDS = {name: (lat, lon) for name, _state, lat, lon in DISTRICTS}

app = FastAPI(title="MeghDrishti Nowcast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {"records": [], "loaded_from": None}
_forecast_cache = {"data": None, "computed_at": 0}
_dgmr_cache = {"data": None, "computed_at": 0, "load_failed": False}
_fusion_cache = {"frame": None, "computed_at": 0}
_FORECAST_TTL_SECONDS = INGEST_CYCLE_MINUTES * 60
_lock = threading.Lock()

# key -> {"records", "loaded_from", "forecast", "fusion_frame", "warmed_at"} —
# a full ingest+forecast+fusion snapshot per demo region, kept warm in the
# background (_region_prewarm_loop) so switching regions via /regions/{key}
# can apply an already-computed snapshot instantly instead of a live
# 10-20s re-ingest (Tomorrow.io per station + RainViewer + Blitzortung's
# fixed listen window + pySTEPS). A snapshot older than
# _SNAPSHOT_FRESH_SECONDS is treated as not-yet-warmed and falls back to a
# live ingest, same as before this existed.
_region_snapshots = {}
_SNAPSHOT_FRESH_SECONDS = INGEST_CYCLE_MINUTES * 60 * 2

# Real hail+lightning across all of India (hazard_india.py) plus the raw
# all-India radar image /raw-layers serves — independent of the per-region
# demo system above, and sharing one RainViewer/Blitzortung fetch between
# both rather than fetching twice. Cached and refreshed in the background
# (_india_hazards_loop) rather than per-request: a fetch takes ~15s
# (RainViewer mosaic + Blitzortung's listen window), too slow to redo on
# every /hazards or /raw-layers call.
_india_hazards_cache = {"hazards": [], "reflectivity": None, "computed_at": 0, "error": None}
_INDIA_HAZARDS_TTL_SECONDS = 180

# district_key ("District, State") -> unix timestamp of the last SMS sent
# for that district — see _maybe_send_alerts. Purely in-memory, resets on
# restart; fine for a hackathon-timescale demo.
_alert_cooldowns = {}


def _district_risk_summary(hazards):
    """Aggregate the real, point-level hazard list into one risk rollup per
    district (judges think in districts, not grid cells) — count of
    hail/lightning hits and the highest severity seen, per district.
    Districts with zero hazards right now are simply absent from the
    output (an empty list is the true state, not something to pad out to
    all ~130 known centroids)."""
    by_district = {}
    for h in hazards:
        key = (h.get("district"), h.get("state"))
        if key not in by_district:
            lat, lon = _DISTRICT_CENTROIDS.get(h["district"], (h["lat"], h["lon"]))
            by_district[key] = {
                "district": h.get("district"),
                "state": h.get("state"),
                "lat": lat,
                "lon": lon,
                "hail_count": 0,
                "lightning_count": 0,
                "max_severity": "low",
            }
        entry = by_district[key]
        if h["type"] == "hail":
            entry["hail_count"] += 1
        elif h["type"] == "lightning":
            entry["lightning_count"] += 1
        if _SEVERITY_RANK[h["severity"]] > _SEVERITY_RANK[entry["max_severity"]]:
            entry["max_severity"] = h["severity"]

    summary = list(by_district.values())
    summary.sort(key=lambda d: (_SEVERITY_RANK[d["max_severity"]], d["hail_count"] + d["lightning_count"]), reverse=True)
    return summary


def _maybe_send_alerts(district_summary):
    """Twilio SMS to ALERT_TO_NUMBERS for any district whose rollup just hit
    ALERT_MIN_SEVERITY (default "high") and isn't still in its post-alert
    cooldown window — last-mile notification for farmers/local
    administration, called out explicitly in the problem statement. Never
    allowed to raise into the caller: a Twilio outage or misconfiguration
    should not affect hazard detection itself."""
    if not sms_alerts.configured():
        return
    now = time.time()
    threshold_rank = _SEVERITY_RANK[ALERT_MIN_SEVERITY]
    for entry in district_summary:
        if _SEVERITY_RANK[entry["max_severity"]] < threshold_rank:
            continue
        key = f"{entry['district']}, {entry['state']}"
        last_sent = _alert_cooldowns.get(key, 0)
        if now - last_sent < ALERT_COOLDOWN_MINUTES * 60:
            continue
        hazard_type = "hail" if entry["hail_count"] >= entry["lightning_count"] else "lightning"
        detail = f"{entry['hail_count']} hail + {entry['lightning_count']} lightning detection(s) nearby."
        body = sms_alerts.format_hazard_alert(entry["district"], entry["state"], hazard_type, entry["max_severity"], detail)
        try:
            sent = sms_alerts.send_sms(body)
            if sent:
                _alert_cooldowns[key] = now
                print(f"[api] sent hazard alert for {key} to {len(sent)} number(s)")
        except Exception as exc:
            print(f"[api] alert send failed for {key}: {exc}")


def _refresh_dgmr():
    """DGMR (section 4b) is loaded and run lazily, on first request only —
    it's a comparison/demo feature, not on the critical startup path, and
    weight download + CPU inference (~seconds) shouldn't slow down the
    primary pySTEPS-driven demo. Cached for the same TTL as pySTEPS."""
    if _dgmr_cache["load_failed"]:
        return None
    now = time.time()
    if _dgmr_cache["data"] is not None and now - _dgmr_cache["computed_at"] < _FORECAST_TTL_SECONDS:
        return _dgmr_cache["data"]
    try:
        from nowcast.models.dgmr_nowcast import run_forecast as dgmr_run_forecast

        with _lock:
            _dgmr_cache["data"] = dgmr_run_forecast()
            _dgmr_cache["computed_at"] = now
        return _dgmr_cache["data"]
    except Exception as exc:
        print(f"[api] DGMR unavailable, disabling for this process: {exc}")
        _dgmr_cache["load_failed"] = True
        return None


def _refresh_fusion():
    now = time.time()
    if _fusion_cache["frame"] is not None and now - _fusion_cache["computed_at"] < _FORECAST_TTL_SECONDS:
        return _fusion_cache["frame"]
    with _lock:
        _fusion_cache["frame"] = build_fused_frame()
        _fusion_cache["computed_at"] = now
    return _fusion_cache["frame"]


def _refresh_forecast():
    """pySTEPS forecast is expensive-ish (LK + extrapolation) — cache it for
    the same ingestion cycle rather than recomputing per request."""
    now = time.time()
    if _forecast_cache["data"] is not None and now - _forecast_cache["computed_at"] < _FORECAST_TTL_SECONDS:
        return _forecast_cache["data"]
    with _lock:
        _forecast_cache["data"] = run_forecast()
        _forecast_cache["computed_at"] = now
    return _forecast_cache["data"]


def _latest_snapshot_path():
    files = sorted(glob.glob(os.path.join(IMD_DIR, "*.json")))
    return files[-1] if files else None


def _refresh():
    path = _latest_snapshot_path()
    if path is None:
        return
    if path == _cache["loaded_from"]:
        return
    with open(path) as f:
        data = json.load(f)
    with _lock:
        _cache["records"] = [classify_station(r) for r in data["records"]]
        _cache["loaded_from"] = path


def _ingest_all():
    """Run all three independent pullers (2a/2b/2c) — one source's failure
    never blocks the others, matching the ingestion layer's failure-isolation
    requirement (section 2)."""
    for name, fn in (("imd", pull_imd), ("satellite", pull_satellite), ("radar", pull_radar)):
        try:
            fn()
        except Exception as exc:
            print(f"[api] {name} puller failed: {exc}")


def _warm_region(key):
    """Run a full ingest+forecast+fusion cycle for `key` and stash the
    result in _region_snapshots. Uses override_active_region — a
    thread-local switch — rather than mutating the persistent global
    active region: this runs for ~10-20s per region, and an earlier version
    that mutated the shared global (even with a save/restore dance) let a
    concurrent request land mid-warm and transiently see the wrong region's
    bbox, confirmed in testing. Thread-local isolation means concurrent
    requests on other threads are never affected, regardless of timing."""
    try:
        with override_active_region(key):
            _ingest_all()
            path = _latest_snapshot_path()
            if path is None:
                return
            with open(path) as f:
                data = json.load(f)
            records = [classify_station(r) for r in data["records"]]
            forecast = run_forecast()
            fusion_frame = build_fused_frame()
        with _lock:
            _region_snapshots[key] = {
                "records": records,
                "loaded_from": path,
                "forecast": forecast,
                "fusion_frame": fusion_frame,
                "warmed_at": time.time(),
            }
    except Exception as exc:
        print(f"[api] pre-warm failed for region '{key}': {exc}")


def _apply_snapshot(key):
    """Instantly point the live caches (what every endpoint actually reads)
    at a pre-warmed snapshot for `key`. Returns False if nothing warm
    enough exists yet, so the caller can fall back to a live ingest."""
    snap = _region_snapshots.get(key)
    if snap is None or (time.time() - snap["warmed_at"]) > _SNAPSHOT_FRESH_SECONDS:
        return False
    with _lock:
        _cache["records"] = snap["records"]
        _cache["loaded_from"] = snap["loaded_from"]
        _forecast_cache["data"] = snap["forecast"]
        _forecast_cache["computed_at"] = snap["warmed_at"]
        _fusion_cache["frame"] = snap["fusion_frame"]
        _fusion_cache["computed_at"] = snap["warmed_at"]
        # DGMR is a lazy, comparison-only feature (section 4b) — not part of
        # the pre-warm set, just invalidated so it recomputes for the new
        # region on next request instead of showing the old region's frame.
        _dgmr_cache["data"] = None
        _dgmr_cache["load_failed"] = False
    return True


def _region_prewarm_loop():
    """Keeps every region's snapshot warm so /regions/{key} is instant
    instead of a live re-ingest. One full pass over all REGIONS per
    INGEST_CYCLE_MINUTES — same cadence the old single-region loop used —
    then re-applies whichever region is currently active, so it keeps
    refreshing periodically exactly like before this existed."""
    while True:
        for key in REGIONS:
            _warm_region(key)
        _apply_snapshot(get_active_region_key())
        time.sleep(INGEST_CYCLE_MINUTES * 60)


def _refresh_india_hazards():
    try:
        from nowcast.ingestion.rainviewer_radar import fetch_india_reflectivity

        reflectivity = fetch_india_reflectivity(hazard_india.INDIA_GRID_SIZE)
        hazards = hazard_india.detect(reflectivity=reflectivity)
        with _lock:
            _india_hazards_cache["hazards"] = hazards
            _india_hazards_cache["reflectivity"] = reflectivity
            _india_hazards_cache["computed_at"] = time.time()
            _india_hazards_cache["error"] = None
    except Exception as exc:
        print(f"[api] all-India hazard detection failed: {exc}")
        with _lock:
            _india_hazards_cache["error"] = str(exc)


def _india_hazards_loop():
    while True:
        _refresh_india_hazards()
        time.sleep(_INDIA_HAZARDS_TTL_SECONDS)


@app.on_event("startup")
def startup():
    active = get_active_region_key()
    _warm_region(active)
    if not _apply_snapshot(active):
        # _warm_region itself failed (e.g. every live source down) — fall
        # back to the original startup path so the app still comes up.
        _ingest_all()
        _refresh()
        _refresh_fusion()
    _refresh_india_hazards()
    threading.Thread(target=_india_hazards_loop, daemon=True).start()
    t = threading.Thread(target=_region_prewarm_loop, daemon=True)
    t.start()


@app.get("/regions")
def regions():
    """Selectable demo regions (section: see settings.py REGIONS docstring) —
    the storm-scale grid is a fixed-size box that can be repositioned to any
    of these; real layers (RainViewer/Blitzortung/ECMWF) already cover
    wherever it's pointed."""
    return {
        "active": get_active_region_key(),
        "options": [{"key": k, "name": v["name"], "bbox": v["bbox"]} for k, v in REGIONS.items()],
    }


@app.post("/regions/{key}")
def set_region(key: str):
    try:
        set_active_region(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not _apply_snapshot(key):
        # Nothing pre-warmed yet for this region (e.g. requested before the
        # background pre-warm loop's first full pass finishes) — fall back
        # to a live ingest so the switch still works, just slower this once.
        with _lock:
            _forecast_cache["data"] = None
            _dgmr_cache["data"] = None
            _dgmr_cache["load_failed"] = False
            _fusion_cache["frame"] = None
            _cache["loaded_from"] = None
        _ingest_all()
        _refresh()
        _refresh_fusion()
        # Populate the snapshot cache too, so switching back to this region
        # later is instant instead of live every time.
        threading.Thread(target=_warm_region, args=(key,), daemon=True).start()

    return {"active": key, "name": get_region_name(), "instant": key in _region_snapshots}


@app.get("/hazards")
def hazards(lead_time: int = Query(0, ge=0, le=360, description="minutes; advects points by real ECMWF wind, see note")):
    """Real hail + lightning hazard points across all of India (see
    models/hazard_india.py), served from a background-refreshed cache
    (~3min cadence — a live fetch takes ~15s, too slow per-request).

    `lead_time` does NOT re-run detection at a future time — there's no
    real all-India forecast mechanism for hail/lightning (same reason
    cloudburst/downburst were dropped entirely, see hazard_india.py).
    Instead each point is advected by the real ECMWF wind vector at its
    location: a standard simplified nowcasting technique (storms roughly
    follow the steering flow), NOT a re-detected forecast — today's real
    detections, moved along today's real wind. The old per-region, all-4-
    hazard demo view (synthetic-backed downburst/cloudburst included) is
    still available at /hazards/region for whichever city is active."""
    with _lock:
        india_hazards = list(_india_hazards_cache["hazards"])
        error = _india_hazards_cache["error"]

    if lead_time > 0 and india_hazards:
        india_hazards = hazard_india.advect_hazards(india_hazards, lead_time)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
            "properties": {
                "hazards": [
                    {k: v for k, v in h.items() if k not in ("lat", "lon")}
                ],
                "lead_minutes": lead_time,
            },
        }
        for h in india_hazards
    ]
    note = "real hail (RainViewer) + lightning (Blitzortung) across all of India"
    if lead_time > 0:
        note += f"; positions advected {lead_time}min by real ECMWF wind, not a re-detected forecast"
    if error and not india_hazards:
        note = f"all-India hazard detection unavailable ({error}) — showing last known / empty"
    return {"type": "FeatureCollection", "features": features, "lead_time_minutes": lead_time, "note": note}


@app.get("/hazards/region")
def hazards_region(lead_time: int = Query(0, description="minutes; snaps to nearest pySTEPS lead step")):
    """The original per-region demo hazard view (all 4 hazard types,
    downburst/cloudburst synthetic-backed) for whichever city is active via
    /regions/{key} — superseded as the dashboard's default by /hazards
    (real, all-India, hail+lightning only) but kept available here."""
    _refresh()
    features = []
    for rec in _cache["records"]:
        if not rec["hazards"]:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
                "properties": {
                    "station_id": rec["station_id"],
                    "name": rec.get("name"),
                    "hazards": rec["hazards"],
                    "ts_severity": rec.get("ts_severity"),
                    "lightning_prob_cat": rec.get("lightning_prob_cat"),
                    "timestamp": rec.get("timestamp"),
                },
            }
        )

    # cloudburst hazard (4c): rule-based on pySTEPS extrapolated rain rate,
    # snapped to the lead step nearest the requested lead_time.
    try:
        fc = _refresh_forecast()
        steps = fc["timestamps_min"]
        nearest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - lead_time)) if lead_time > 0 else None
        hits = cloudburst_cells(fc, threshold_mm_hr=CLOUDBURST_RAIN_RATE_MM_HR)
        if nearest_idx is not None:
            target_lead = steps[nearest_idx]
            hits = [h for h in hits if h["lead_minutes"] == target_lead]
        else:
            hits = [h for h in hits if h["lead_minutes"] == steps[0]]
        for h in hits:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                "properties": {
                    "hazards": [{"type": "cloudburst", "severity": "high", "rainrate_mm_hr": h["rainrate_mm_hr"]}],
                    "lead_minutes": h["lead_minutes"],
                },
            })
    except Exception as exc:
        print(f"[api] cloudburst forecast unavailable: {exc}")

    # hail + downburst (4c): grid-based rules on the fused raster, current
    # timestep only — these don't have a pySTEPS-extrapolated future state.
    if lead_time == 0:
        try:
            frame = _refresh_fusion()
            if frame is not None:
                for h in hail_cells(frame):
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                        "properties": {
                            "hazards": [{"type": "hail", "severity": "high",
                                         "reflectivity_dbz": h["reflectivity_dbz"]}],
                        },
                    })
                for d in downburst_cells(frame):
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
                        "properties": {
                            "hazards": [{"type": "downburst", "severity": "high",
                                         "velocity_delta_ms": d["velocity_delta_ms"]}],
                        },
                    })
        except Exception as exc:
            print(f"[api] grid hazards unavailable: {exc}")

    return {"type": "FeatureCollection", "features": features, "lead_time_minutes": lead_time}


@app.get("/forecast")
def forecast(model: str = Query("pysteps", pattern="^(pysteps|dgmr)$")):
    """Forecast summary for the dashboard time slider / baseline-vs-AI
    comparison (section 4b). `model=pysteps` (default, 0-6h, calibrated
    mm/hr) or `model=dgmr` (0-90min, relative intensity 0-1 — see
    dgmr_nowcast module docstring for why it's not in mm/hr)."""
    if model == "dgmr":
        fc = _refresh_dgmr()
        if fc is None:
            return {"available": False, "reason": "DGMR failed to load in this process (see server log)"}
        return {
            "available": True,
            "timestamps_min": fc["timestamps_min"],
            "max_intensity": [round(float(f.max()), 2) for f in fc["intensity_forecast"]],
            "mean_intensity": [round(float(f.mean()), 3) for f in fc["intensity_forecast"]],
            "bbox": fc["bbox"],
            "source": "dgmr-synthetic-input",
            "note": fc["note"],
        }
    fc = _refresh_forecast()
    return {
        "available": True,
        "timestamps_min": fc["timestamps_min"],
        "max_rainrate_mm_hr": [round(float(f.max()), 1) for f in fc["rainrate_forecast"]],
        "mean_rainrate_mm_hr": [round(float(f.mean()), 2) for f in fc["rainrate_forecast"]],
        "bbox": fc["bbox"],
        "source": f"pysteps-{fc.get('source', 'synthetic')}",
    }


@app.get("/storm-eta")
def storm_eta():
    _refresh()
    return {"cells": storm_cells(_cache["records"])}


def _array_to_png_data_url(arr, cmap_name, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    from PIL import Image

    norm = mcolors.Normalize(vmin=vmin if vmin is not None else float(arr.min()),
                              vmax=vmax if vmax is not None else float(arr.max()))
    rgba = (cm.get_cmap(cmap_name)(norm(arr)) * 255).astype(np.uint8)
    # flip vertically: array row 0 is the southern edge of the grid, PNG row 0 is the top
    img = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/nowcast-frame")
def nowcast_frame(
    model: str = Query("pysteps", pattern="^(pysteps|dgmr)$"),
    lead_time: int = Query(10, description="minutes; snaps to nearest available lead step"),
):
    """Single forecast frame as a PNG overlay (section 4b's baseline-vs-AI
    comparison toggle) — pySTEPS rain rate (calibrated mm/hr, turbo
    colormap) or DGMR relative intensity (unitless 0-1, plasma colormap,
    distinct palette so it's visually obvious this is not the same unit)."""
    if model == "dgmr":
        fc = _refresh_dgmr()
        if fc is None:
            return {"available": False, "reason": "DGMR failed to load in this process (see server log)"}
        idx = min(range(len(fc["timestamps_min"])), key=lambda i: abs(fc["timestamps_min"][i] - lead_time))
        frame = fc["intensity_forecast"][idx]
        image = _array_to_png_data_url(frame, "plasma", vmin=0, vmax=1)
        return {"available": True, "image": image, "bbox": fc["bbox"],
                "lead_minutes": fc["timestamps_min"][idx], "source": "dgmr", "note": fc["note"]}

    fc = _refresh_forecast()
    idx = min(range(len(fc["timestamps_min"])), key=lambda i: abs(fc["timestamps_min"][i] - lead_time))
    frame = fc["rainrate_forecast"][idx]
    image = _array_to_png_data_url(frame, "turbo", vmin=0, vmax=65)
    return {"available": True, "image": image, "bbox": fc["bbox"],
            "lead_minutes": fc["timestamps_min"][idx], "source": f"pysteps-{fc.get('source', 'synthetic')}"}


@app.get("/raw-layers")
def raw_layers():
    """Satellite IR + radar reflectivity as image overlays (section 5a).

    Satellite is real (tir1 only) when `USE_LIVE_SATELLITE=true` — see
    satellite_insat.py / copernicus_satellite.py — sourced from Copernicus
    Sentinel-3 SLSTR, not MOSDAC/INSAT; still scoped to the active demo
    region's small bbox (Sentinel-3/EUMETSAT don't give an easy all-India
    single-request equivalent the way RainViewer does for radar). Radar is
    real when `USE_LIVE_RADAR=true` — see radar_puller.py /
    rainviewer_radar.py — sourced from RainViewer across all of India (the
    same cached fetch /hazards' hail detection uses, see hazard_india.py),
    not scoped to the active region at all: real weather doesn't confine
    itself to whichever demo city happens to be selected.
    """
    frame = _refresh_fusion()
    if frame is None:
        return {"layers": [], "note": "no fused frame yet — ingestion still warming up"}

    from nowcast.configs.settings import INDIA_BBOX
    from nowcast.ingestion.radar_puller import USE_LIVE_RADAR
    from nowcast.ingestion.satellite_insat import USE_LIVE_SATELLITE

    satellite_source = "copernicus-sentinel3" if USE_LIVE_SATELLITE else "synthetic"
    ch = frame["channels"]
    layers = [
        {
            "id": "satellite_tir1",
            "label": "Satellite IR (TIR-1, 10.8um)",
            "bbox": frame["bbox"],
            "image": _array_to_png_data_url(ch["tir1"], "gray_r", vmin=190, vmax=300),
            "source": satellite_source,
        },
    ]

    with _lock:
        india_reflectivity = _india_hazards_cache["reflectivity"]
    if USE_LIVE_RADAR and india_reflectivity is not None:
        radar_source = "rainviewer"
        layers.append(
            {
                "id": "radar_reflectivity",
                "label": "Radar reflectivity (dBZ) — all India",
                "bbox": INDIA_BBOX,
                "image": _array_to_png_data_url(india_reflectivity, "turbo", vmin=0, vmax=65),
                "source": radar_source,
            }
        )
    else:
        radar_source = "synthetic"
        layers.append(
            {
                "id": "radar_reflectivity",
                "label": "Radar reflectivity (dBZ)",
                "bbox": frame["bbox"],
                "image": _array_to_png_data_url(ch["reflectivity_dbz"], "turbo", vmin=0, vmax=65),
                "source": radar_source,
            }
        )
    notes = []
    notes.append("satellite real via Copernicus Sentinel-3 SLSTR" if USE_LIVE_SATELLITE else "satellite synthetic")
    notes.append("radar real via RainViewer (IMD-sourced, not direct MOSDAC)" if USE_LIVE_RADAR else "radar synthetic")
    return {"layers": layers, "note": "; ".join(notes)}


@app.get("/weather-layers")
def weather_layers(lead_time: int = Query(0, description="minutes ahead; ECMWF snaps to its nearest 3h step")):
    """Temperature/humidity/wind-speed as colored map overlays across the
    wide demo region (§WIDE_BBOX) — not just the narrow storm bbox used for
    radar/satellite/hazards. Real ECMWF Open Data when USE_LIVE_ECMWF=true,
    otherwise synthetic (see processing/weather_fields.py) — the response
    always reports which one actually happened, since a live fetch failure
    silently falls back to synthetic. `lead_time` lets the frontend animate
    this alongside the hazard/nowcast lead-time slider instead of only ever
    showing "now"."""
    g = weather_fields.generate_grid(lead_time)
    layers = [
        {
            "id": "temperature",
            "label": "Temperature",
            "unit": "°C",
            "bbox": g["bbox"],
            "vmin": 18, "vmax": 34,
            "image": _array_to_png_data_url(g["temperature_c"], "RdYlBu_r", vmin=18, vmax=34),
        },
        {
            "id": "humidity",
            "label": "Relative humidity",
            "unit": "%",
            "bbox": g["bbox"],
            "vmin": 0, "vmax": 100,
            "image": _array_to_png_data_url(g["humidity_pct"], "YlGnBu", vmin=0, vmax=100),
        },
        {
            "id": "wind_speed",
            "label": "Wind speed",
            "unit": "m/s",
            "bbox": g["bbox"],
            "vmin": 0, "vmax": 18,
            "image": _array_to_png_data_url(g["wind_speed_ms"], "plasma", vmin=0, vmax=18),
        },
        {
            "id": "pressure",
            "label": "Mean sea level pressure",
            "unit": "hPa",
            "bbox": g["bbox"],
            "vmin": 995, "vmax": 1015,
            "image": _array_to_png_data_url(g["pressure_hpa"], "coolwarm", vmin=995, vmax=1015),
        },
    ]

    # Rainfall: real, all-India, derived from the same cached RainViewer
    # reflectivity /hazards' hail detection uses (see hazard_india.py) via
    # the standard Marshall-Palmer Z-R relation (Z=200R^1.6) — a genuine
    # current rain-rate estimate, not a pySTEPS forecast, since there's no
    # real all-India forecast mechanism (no persisted real radar time
    # series for pySTEPS to extrapolate from). This replaced an earlier
    # version of this layer that silently used the per-region pySTEPS
    # frame, which meant "Rainfall" was the one weather variable still
    # secretly scoped to whichever demo city was active.
    import numpy as np
    from nowcast.configs.settings import INDIA_BBOX

    with _lock:
        india_reflectivity = _india_hazards_cache["reflectivity"]
    if india_reflectivity is not None:
        rainrate = np.power(np.power(10, india_reflectivity / 10) / 200, 1 / 1.6)
        layers.append(
            {
                "id": "rainfall",
                "label": "Rain rate (from real radar)",
                "unit": "mm/hr",
                "bbox": INDIA_BBOX,
                "vmin": 0, "vmax": 65,
                "image": _array_to_png_data_url(rainrate, "turbo", vmin=0, vmax=65),
            }
        )

    source = g.get("source", "synthetic")
    note = (
        "real ECMWF Open Data (HRES, CC-BY-4.0) — see nowcast/ingestion/ecmwf_weather.py"
        if source == "ecmwf-opendata"
        else "synthetic ambient fields — not an IMD/MOSDAC/ECMWF product"
    )
    return {"layers": layers, "note": note, "source": source}


@app.get("/wind-vectors")
def wind_vectors(lead_time: int = Query(0, description="minutes ahead, same semantics as /weather-layers")):
    """Sparse wind arrow points (speed + direction) for symbol rendering."""
    return {"points": weather_fields.wind_vector_points(t_min=lead_time)}


@app.get("/region-forecast")
def region_forecast(lat: float, lon: float, lead_time: int = Query(0, ge=0, le=360)):
    """Point-sampled future trend for a user-selected region (temperature/
    humidity/wind at a chosen lead time) — backs the dashboard's per-region
    time-scale panel. Real ECMWF Open Data when USE_LIVE_ECMWF=true (the
    lead-time rounds to ECMWF's nearest 3h forecast step), otherwise
    synthetic (see weather_fields.py). If the point falls inside the storm
    bbox, also includes the pySTEPS cloudburst rain-rate forecast (always
    synthetic input) at the nearest lead step for that location."""
    sample = weather_fields.sample_point(lat, lon, lead_time)

    cloudburst_rainrate = None
    fc = _refresh_forecast()
    lon_min, lat_min, lon_max, lat_max = fc["bbox"]
    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
        n = fc["grid_size"]
        xi = int(round((lon - lon_min) / (lon_max - lon_min) * (n - 1)))
        yi = int(round((lat - lat_min) / (lat_max - lat_min) * (n - 1)))
        idx = min(range(len(fc["timestamps_min"])), key=lambda i: abs(fc["timestamps_min"][i] - lead_time))
        cloudburst_rainrate = round(float(fc["rainrate_forecast"][idx][yi, xi]), 1)

    return {**sample, "lead_minutes": lead_time, "cloudburst_rainrate_mm_hr": cloudburst_rainrate}


@app.get("/area-forecast")
def area_forecast(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    lead_time: int = Query(0, ge=0, le=360),
):
    """Min/mean/max current-and-forecast stats over a user drag-selected
    area, not a single point (see /region-forecast for that) — backs the
    map's drag-to-select-area tool. Same real-vs-synthetic weather_fields.py
    backend as /region-forecast. Also includes the pySTEPS cloudburst
    rain-rate max/mean over whatever part of the area falls inside the
    storm-scale grid (REGION_BBOX), since that's a different, smaller bbox
    than the weather-variable grid — None if the area doesn't overlap it."""
    bbox = (lon_min, lat_min, lon_max, lat_max)
    stats = weather_fields.area_stats(bbox, lead_time)

    cloudburst_stats = None
    fc = _refresh_forecast()
    flon_min, flat_min, flon_max, flat_max = fc["bbox"]
    import numpy as np

    lons = np.linspace(flon_min, flon_max, fc["grid_size"])
    lats = np.linspace(flat_min, flat_max, fc["grid_size"])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = (lon_grid >= lon_min) & (lon_grid <= lon_max) & (lat_grid >= lat_min) & (lat_grid <= lat_max)
    if mask.any():
        idx = min(range(len(fc["timestamps_min"])), key=lambda i: abs(fc["timestamps_min"][i] - lead_time))
        vals = fc["rainrate_forecast"][idx][mask]
        cloudburst_stats = {"min": round(float(vals.min()), 1), "mean": round(float(vals.mean()), 1), "max": round(float(vals.max()), 1)}

    return {**stats, "cloudburst_rainrate_mm_hr": cloudburst_stats, "lead_minutes": lead_time, "bbox": bbox}


_BHUVAN_WMS = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/gwc/service/wms/"


@app.get("/wms-proxy/bhuvan")
def wms_proxy_bhuvan(request: Request):
    """Thin passthrough proxy for Bhuvan's WMS ("Bhuvan Maps" base layer).

    Bhuvan's server doesn't send CORS headers, so a browser can't fetch its
    tiles directly (confirmed via direct testing — curl gets 200, browser
    fetch gets blocked by CORS). The target host is hardcoded, not taken
    from the request, so this can't be used as an open SSRF proxy — it only
    ever forwards to Bhuvan's WMS with whatever WMS query params the caller
    sent (layers/bbox/etc.), which is exactly what a legitimate map tile
    request looks like.
    """
    try:
        upstream = requests.get(_BHUVAN_WMS, params=dict(request.query_params), timeout=10)
    except requests.RequestException as exc:
        return Response(content=str(exc), status_code=502, media_type="text/plain")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("Content-Type", "image/png"),
    )


_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")


@app.get("/history/timestamps")
def history_timestamps():
    """Every IMD ingestion snapshot currently on disk — the real (if
    short-lived, since it only covers this server process's uptime)
    historical archive Replay is built on. Each ingestion cycle writes a
    new timestamped file rather than overwriting the last one."""
    return {"timestamps": list_imd_timestamps()}


@app.get("/history/hazards")
def history_hazards(timestamp: str):
    """Reconstruct hazards for a specific historical IMD snapshot — real
    replay, not a re-run of "now". Station-level hazards (lightning, point
    hail flags) come directly from that snapshot's IMD records. Grid-based
    hail/downburst come from re-fusing the satellite/radar snapshots
    nearest that timestamp and re-running the same rule (hail_cells/
    downburst_cells only need one fused frame, so this is meaningful).

    Cloudburst is deliberately excluded: it comes from pySTEPS, which
    always regenerates its own synthetic present-moment history regardless
    of what timestamp is requested (see pysteps_baseline.py) — there's no
    persisted historical radar *sequence* to re-run it against, so a
    "replayed" cloudburst value would silently just be today's forecast
    mislabeled with a past timestamp. Better to omit it than fake it.
    """
    if not _TIMESTAMP_RE.match(timestamp):
        raise HTTPException(400, "timestamp must look like 20260918T083650Z")

    imd_path = os.path.join(IMD_DIR, f"{timestamp}.json")
    if not os.path.exists(imd_path):
        raise HTTPException(404, f"no IMD snapshot for timestamp {timestamp}")

    with open(imd_path) as f:
        records = json.load(f)["records"]

    features = []
    for rec in (classify_station(r) for r in records):
        if not rec["hazards"]:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
            "properties": {
                "station_id": rec["station_id"],
                "name": rec.get("name"),
                "hazards": rec["hazards"],
                "ts_severity": rec.get("ts_severity"),
                "lightning_prob_cat": rec.get("lightning_prob_cat"),
                "timestamp": rec.get("timestamp"),
            },
        })

    frame = build_fused_frame_for_timestamp(timestamp)
    if frame is not None:
        for h in hail_cells(frame):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                "properties": {"hazards": [{"type": "hail", "severity": "high", "reflectivity_dbz": h["reflectivity_dbz"]}]},
            })
        for d in downburst_cells(frame):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
                "properties": {"hazards": [{"type": "downburst", "severity": "high", "velocity_delta_ms": d["velocity_delta_ms"]}]},
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "timestamp": timestamp,
        "note": "cloudburst omitted — pySTEPS has no persisted historical sequence to replay against, see docstring",
    }


@app.get("/health")
def health():
    return {"status": "ok", "loaded_from": _cache["loaded_from"]}
