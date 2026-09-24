"""Central config: demo region, thresholds, paths.

Demo region default: Pune district, Maharashtra (good IMD AWS density,
inside MOSDAC radar footprint). Change BBOX to retarget the whole pipeline.
"""
import contextlib
import os
import threading
from dotenv import load_dotenv

load_dotenv()

# Selectable demo regions — the storm-scale grid (radar/satellite/pySTEPS/
# DGMR/hazards) is deliberately a small, fixed-size box (~0.5deg, matches
# GRID_SIZE=64 in synthetic_radar.py for ~800m/cell resolution): widening
# this box itself to cover all of India would collapse the demo storm to
# a sub-pixel blob and blow up hazard-rule filter windows tuned for this
# scale. Instead, the SAME size box can be repositioned to any major city,
# so real layers (RainViewer/Blitzortung/ECMWF, which already cover all of
# India) and the synthetic storm-scale grid both center on wherever the
# user picks.
REGIONS = {
    "pune": {"name": "Pune", "bbox": (73.6, 18.3, 74.1, 18.8)},
    "delhi": {"name": "Delhi NCR", "bbox": (76.85, 28.35, 77.35, 28.85)},
    "mumbai": {"name": "Mumbai", "bbox": (72.6, 18.85, 73.1, 19.35)},
    "chennai": {"name": "Chennai", "bbox": (80.0, 12.85, 80.5, 13.35)},
    "kolkata": {"name": "Kolkata", "bbox": (88.15, 22.35, 88.65, 22.85)},
    "bengaluru": {"name": "Bengaluru", "bbox": (77.35, 12.75, 77.85, 13.25)},
    "hyderabad": {"name": "Hyderabad", "bbox": (78.25, 17.15, 78.75, 17.65)},
    "ahmedabad": {"name": "Ahmedabad", "bbox": (72.35, 22.80, 72.85, 23.30)},
    "jaipur": {"name": "Jaipur", "bbox": (75.55, 26.65, 76.05, 27.15)},
    "guwahati": {"name": "Guwahati", "bbox": (91.5, 26.0, 92.0, 26.5)},
}

_active_region_key = "pune"  # persistent global, set only by set_active_region()
_region_override = threading.local()  # per-thread override, see override_active_region()


def get_active_region_key():
    """The active region — a thread-local override if one is in effect on
    *this* thread (see override_active_region), else the persistent global
    that every other thread/request sees."""
    return getattr(_region_override, "key", None) or _active_region_key


def set_active_region(key):
    """Persistently switch the active demo region for every future request
    on every thread — this is the real, user-facing switch (the
    /regions/{key} endpoint). Takes effect immediately: every consumer calls
    get_region_bbox()/get_region_name() fresh rather than importing a frozen
    constant. For a temporary, single-thread-only switch, use
    override_active_region() instead — see its docstring for why the
    distinction matters."""
    global _active_region_key
    if key not in REGIONS:
        raise ValueError(f"unknown region '{key}', choose from {list(REGIONS)}")
    _active_region_key = key


@contextlib.contextmanager
def override_active_region(key):
    """Temporarily switch the active region for the CURRENT THREAD ONLY,
    leaving the persistent global (and therefore every other in-flight
    request, which may run on a different threadpool thread) untouched.

    Exists for the background region pre-warm loop (api/main.py): warming
    region B for a ~10-20s ingest cycle must not make a concurrent request
    for region A transiently see region B's bbox — that happened in
    testing when this used a naive "save global, mutate it, restore it"
    approach instead, and a live request landed mid-warm and silently got
    the wrong region's data. threading.local() isolates it per-OS-thread,
    which is exactly the boundary FastAPI's sync-endpoint threadpool and
    this module's dedicated background thread both already respect.
    """
    if key not in REGIONS:
        raise ValueError(f"unknown region '{key}', choose from {list(REGIONS)}")
    prev = getattr(_region_override, "key", None)
    _region_override.key = key
    try:
        yield
    finally:
        if prev is None:
            del _region_override.key
        else:
            _region_override.key = prev


def get_region_bbox():
    return REGIONS[get_active_region_key()]["bbox"]


def get_region_name():
    return REGIONS[get_active_region_key()]["name"]


# Legacy static constants — frozen at import time, do NOT reflect region
# switches made after import. Kept only so nothing crashes if something still
# imports these directly; every ingestion/processing/model module in this
# project has been converted to call get_region_bbox()/get_region_name()
# instead. New code should always use the getters.
REGION_BBOX = REGIONS[_active_region_key]["bbox"]
REGION_NAME = REGIONS[_active_region_key]["name"]

# Wider weather-variable grid (temperature/humidity/wind/pressure) — always
# covers all of India, not just a box around the active region. Unlike the
# storm-scale REGION_BBOX (radar/satellite/hazards), this is a smooth
# ambient field with no per-pixel storm signature to collapse, and ECMWF
# Open Data genuinely covers the whole globe — cropping it to a small box
# was an artificial limit, not a resolution necessity. Decoupled from the
# region picker entirely: switching demo cities moves the storm-scale grid,
# this stays fixed on the whole country. 96 cells/side keeps real ECMWF
# fetch+regrid cheap (interpolation target size barely affects fetch time,
# which is dominated by GRIB download) while still looking reasonably
# smooth across a ~30x30deg extent.
WIDE_GRID_SIZE = 96
INDIA_BBOX = (68.0, 6.5, 97.5, 37.0)


def get_wide_bbox():
    return INDIA_BBOX

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMD_DIR = os.path.join(DATA_DIR, "imd")

# Set True once real IMD API key / MOSDAC creds are configured via .env.
# False = replay/mock mode (see fallback matrix, section 6 of project.md).
USE_LIVE_IMD = os.getenv("USE_LIVE_IMD", "false").lower() == "true"
IMD_API_KEY = os.getenv("IMD_API_KEY", "")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY", "")

# ECMWF Open Data (temperature/humidity/wind grid) — genuinely free, no API
# key needed (their older key-based public-datasets service was mostly
# decommissioned in 2023; see nowcast/ingestion/ecmwf_weather.py). Still
# opt-in like the other USE_LIVE_* flags: it makes real network calls on
# every distinct forecast step requested, so it's not on by default.
USE_LIVE_ECMWF = os.getenv("USE_LIVE_ECMWF", "false").lower() == "true"

# RainViewer radar reflectivity — real, quantitative dBZ, no API key needed.
# India coverage is IMD's public radar network, republished by RainViewer.
# Radial (Doppler) velocity has no public equivalent and stays synthetic
# even with this on — see nowcast/ingestion/rainviewer_radar.py.
USE_LIVE_RADAR = os.getenv("USE_LIVE_RADAR", "false").lower() == "true"

# Blitzortung.org real lightning strikes — free community VLF network, no
# API key needed, fills the gap neither the IMD feed nor Tomorrow.io cover
# (Tomorrow.io's realtime endpoint has no lightning field at all). Independent
# of USE_LIVE_IMD: applies on top of whichever station-data source is active.
# See nowcast/ingestion/blitzortung_lightning.py.
USE_LIVE_LIGHTNING = os.getenv("USE_LIVE_LIGHTNING", "false").lower() == "true"

# Copernicus Data Space Ecosystem (Sentinel-3 SLSTR F1 thermal band) — real
# satellite brightness temperature, the one hazard input with no other free
# live source. Needs a free CDSE account + OAuth2 client credentials (client
# ID/secret from your account's API credentials page, NOT your login
# password) — see nowcast/ingestion/copernicus_satellite.py for the caveats
# (polar-orbit revisit gap, F1 is a thermal/fire channel not literally
# INSAT's TIR1, wv/mwir stay synthetic even when this succeeds).
USE_LIVE_SATELLITE = os.getenv("USE_LIVE_SATELLITE", "false").lower() == "true"
COPERNICUS_CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID", "")
COPERNICUS_CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET", "")

# EUMETSAT Data Store + Data Tailor (MSG SEVIRI IR10.8) — continuous-coverage
# alternative to Copernicus above: geostationary, updates every 15min, and
# actually centered on India/Indian Ocean, vs Sentinel-3's ~1-2 passes/day.
# Also needs a free account + API credentials (consumer key/secret from
# api.eumetsat.int/api-key, NOT your login password). Tried first when both
# are configured — see satellite_insat.py — since continuous coverage beats
# occasional passes. See nowcast/ingestion/eumetsat_satellite.py for the
# "written but not live-tested" caveats.
EUMETSAT_CONSUMER_KEY = os.getenv("EUMETSAT_CONSUMER_KEY", "")
EUMETSAT_CONSUMER_SECRET = os.getenv("EUMETSAT_CONSUMER_SECRET", "")

# Hazard thresholds (section 4c of project.md) — documented here, not buried.
HAIL_LIGHTNING_CAT_MIN = "cat17"       # IMD hail flag category
CLOUDBURST_RAIN_RATE_MM_HR = 15.0      # IMD "very heavy rain" threshold
LIGHTNING_PROB_HIGH = 0.60             # Cat19 boundary

# Grid-based hail rule (4c): reflectivity core AND cold cloud top AND
# elevated lightning, all collocated on the fusion grid.
HAIL_REFLECTIVITY_MIN_DBZ = 55.0
HAIL_COLD_TOP_MAX_K = 210.0            # TIR-1 brightness temp, overshoot-top territory
HAIL_LIGHTNING_PROB_MIN = 0.30

# Downburst (4c): radial-velocity couplet magnitude — inbound/outbound
# delta across the storm core. Only computable with real radar velocity,
# never from a PNG overlay fallback.
DOWNBURST_VELOCITY_DELTA_MS = 25.0

INGEST_CYCLE_MINUTES = 15

# Twilio SMS alerts — last-mile notification for farmers/local administration
# (explicitly called out in the problem statement) when a real "high"
# severity hazard is detected. Off by default like every other USE_LIVE_*
# flag; needs a real Twilio account (free trial works) — see
# nowcast/alerts/sms_alerts.py.
USE_LIVE_ALERTS = os.getenv("USE_LIVE_ALERTS", "false").lower() == "true"
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
# Comma-separated E.164 numbers, e.g. "+919812345678,+919898765432".
ALERT_TO_NUMBERS = [n.strip() for n in os.getenv("ALERT_TO_NUMBERS", "").split(",") if n.strip()]
# Minimum severity that triggers an SMS — "high" only by default, so a
# fresh live radar cycle every ~3min doesn't spam the same ongoing storm.
ALERT_MIN_SEVERITY = os.getenv("ALERT_MIN_SEVERITY", "high")
# Once a district has been alerted, don't alert it again for this many
# minutes even if new high-severity hazards keep appearing there — an
# active storm easily spans several ingest cycles.
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
