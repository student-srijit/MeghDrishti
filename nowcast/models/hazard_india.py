"""Real hail + lightning hazard detection across all of India.

Unlike the per-region demo (models/hazard.py + processing/synthetic_radar.py,
still used by the Forecast/Replay pages), this has no synthetic storm and no
fixed demo city — it looks at real RainViewer reflectivity and real
Blitzortung lightning strikes across the whole country and flags wherever
they actually indicate hail-favorable conditions or a strike, right now.

Downburst (needs Doppler radial velocity — no public source publishes raw
volumetric scans) and cloudburst (needs a persisted real radar time-series
for pySTEPS to extrapolate from, which a single "now" RainViewer frame per
cycle doesn't provide) have no real all-India equivalent. Rather than fake
either at country scale, both are simply absent from this module's output —
they remain available, synthetic-backed, in the per-region demo.

Hail rule here is simplified from hazard.py's grid rule: reflectivity +
collocated real lightning only, no cold-cloud-top requirement. Real
satellite coverage (Copernicus Sentinel-3's polar orbit, EUMETSAT pending
license) isn't available everywhere in India at once, so requiring it would
make hail flicker on/off based on incidental satellite coverage rather than
actual storm severity — reflectivity + lightning is still a real,
non-synthetic signal on its own.

Severity is 3-tier (low/moderate/high, colored green/yellow/red on the map)
based on reflectivity for hail, bumped up a tier if a real strike is
collocated; lightning strikes are always "high" (an actual strike is
inherently a live hazard, not a graded risk).

`lead_minutes` (used by /hazards' lead-time slider) does NOT re-run
detection at a future time — there's no real all-India forecast mechanism
for hail/lightning (same reason cloudburst/downburst were dropped
entirely). Instead each point's position is advected by the real ECMWF
wind vector at that location, a standard simplified nowcasting technique
(storms roughly follow the steering flow) — NOT a re-detected forecast,
just today's real detections moved along today's real wind. Documented
explicitly rather than left implicit, since it's a real/synthetic
distinction worth being honest about.

Output is real hail + lightning only — no synthetic filler points. The
map may legitimately show few or zero points when India has little active
convection; that's the true state, not something to paper over.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from nowcast.configs.settings import INDIA_BBOX, HAIL_REFLECTIVITY_MIN_DBZ
from nowcast.configs.districts_india import nearest_districts_vectorized

INDIA_GRID_SIZE = 150  # ~0.2deg/cell, ~22km — fine enough for a country overview
LIGHTNING_PROXIMITY_KM = 25.0  # collocated-with-lightning bumps hail severity up a tier
HAIL_MODERATE_DBZ = 65.0  # >= this (and below HIGH) -> "moderate"
HAIL_HIGH_DBZ = 78.0  # >= this -> "high"

_SEVERITY_ORDER = ["low", "moderate", "high"]


def _bump_severity(severity):
    idx = min(_SEVERITY_ORDER.index(severity) + 1, len(_SEVERITY_ORDER) - 1)
    return _SEVERITY_ORDER[idx]


def _km_per_deg(lat):
    return 111.0, 111.0 * np.cos(np.radians(lat))


def advect_point(lat, lon, lead_minutes):
    """Shift (lat, lon) by the real ECMWF wind vector at that point over
    `lead_minutes` — see module docstring for what this is and isn't."""
    if lead_minutes <= 0:
        return lat, lon
    from nowcast.processing import weather_fields

    sample = weather_fields.sample_point(lat, lon, lead_minutes)
    speed_ms = sample["wind_speed_ms"]
    if speed_ms <= 0:
        return lat, lon
    # wind_dir_deg is the direction wind blows FROM (met convention) —
    # movement is the opposite direction.
    to_rad = np.radians((sample["wind_dir_deg"] + 180) % 360)
    distance_km = speed_ms * (lead_minutes * 60) / 1000.0
    km_lat, km_lon = _km_per_deg(lat)
    dlat = (distance_km * np.cos(to_rad)) / km_lat
    dlon = (distance_km * np.sin(to_rad)) / km_lon
    return lat + dlat, lon + dlon


def advect_hazards(hazards, lead_minutes):
    """Apply advect_point to a list of hazard dicts (main.py calls this on
    the cached "now" detections per-request instead of re-running the full
    ~15s RainViewer+Blitzortung fetch for every lead_time)."""
    if lead_minutes <= 0:
        return hazards
    out = []
    for h in hazards:
        lat, lon = advect_point(h["lat"], h["lon"], lead_minutes)
        out.append({**h, "lat": lat, "lon": lon})
    return out


def detect(reflectivity=None, strikes=None):
    """Real hail + lightning hazard points across all of India.

    `reflectivity`/`strikes` can be pre-fetched and passed in (main.py does
    this, sharing one RainViewer/Blitzortung fetch between hazard detection
    and the /raw-layers all-India radar image instead of fetching twice) —
    left as None, this fetches them itself, so the module stays runnable
    standalone via `python -m nowcast.models.hazard_india`.

    Returns a list of {lat, lon, type, severity, ...} dicts. Raises if the
    radar fetch itself fails (no data at all to work with) — callers should
    treat that like any other live-source failure. A failed *lightning*
    fetch is non-fatal: hail detection still runs on reflectivity alone,
    just without the lightning-proximity severity bump, and simply
    contributes no lightning hazard points itself.
    """
    if reflectivity is None:
        from nowcast.ingestion.rainviewer_radar import fetch_india_reflectivity

        reflectivity = fetch_india_reflectivity(INDIA_GRID_SIZE)

    if strikes is None:
        try:
            from nowcast.ingestion.blitzortung_lightning import fetch_india_strikes

            strikes = fetch_india_strikes()
        except Exception as exc:
            print(f"[hazard_india] lightning fetch failed ({exc}), hail runs on reflectivity alone")
            strikes = []

    lon_min, lat_min, lon_max, lat_max = INDIA_BBOX
    lons = np.linspace(lon_min, lon_max, INDIA_GRID_SIZE)
    lats = np.linspace(lat_min, lat_max, INDIA_GRID_SIZE)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    hazards = []

    for s in strikes:
        # A real strike is inherently an immediate hazard, not a graded
        # risk — always "high" (red), unlike hail's threshold-based tiers.
        hazards.append({"lat": s["lat"], "lon": s["lon"], "type": "lightning", "severity": "high"})

    strike_lats = np.array([s["lat"] for s in strikes]) if strikes else None
    strike_lons = np.array([s["lon"] for s in strikes]) if strikes else None

    ys, xs = np.where(reflectivity >= HAIL_REFLECTIVITY_MIN_DBZ)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cell_lat, cell_lon = float(lat_grid[y, x]), float(lon_grid[y, x])
        dbz = float(reflectivity[y, x])
        if dbz >= HAIL_HIGH_DBZ:
            severity = "high"
        elif dbz >= HAIL_MODERATE_DBZ:
            severity = "moderate"
        else:
            severity = "low"
        if strikes:
            km_lat, km_lon = _km_per_deg(cell_lat)
            nearest_km = np.min(
                np.hypot((cell_lat - strike_lats) * km_lat, (cell_lon - strike_lons) * km_lon)
            )
            if nearest_km <= LIGHTNING_PROXIMITY_KM:
                severity = _bump_severity(severity)
        hazards.append(
            {
                "lat": cell_lat,
                "lon": cell_lon,
                "type": "hail",
                "severity": severity,
                "reflectivity_dbz": round(dbz, 1),
            }
        )

    for h in hazards:
        h["source"] = "real"

    # District/state labels (judges think in districts, not grid cells —
    # see districts_india.py for what "nearest centroid" actually means
    # here). Vectorized across every hazard point at once rather than a
    # per-point lookup loop, same reasoning as the lightning-proximity
    # vectorization above.
    if hazards:
        district_labels = nearest_districts_vectorized(
            [h["lat"] for h in hazards], [h["lon"] for h in hazards]
        )
        for h, label in zip(hazards, district_labels):
            h["district"] = label["district"]
            h["state"] = label["state"]

    return hazards


if __name__ == "__main__":
    result = detect()
    hail = [h for h in result if h["type"] == "hail"]
    lightning = [h for h in result if h["type"] == "lightning"]
    print(f"{len(hail)} hail point(s), {len(lightning)} lightning strike(s) across India right now")
    for h in hail[:5]:
        print(h)
