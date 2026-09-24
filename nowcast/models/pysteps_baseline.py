"""pySTEPS optical-flow + extrapolation baseline (section 4a of project.md).

This is the guaranteed-working nowcast per the plan: given a short history
of reflectivity frames, estimate a motion field (Lucas-Kanade) and
extrapolate it forward (semi-Lagrangian) for 0-6h.

Real input when `USE_LIVE_RADAR=true`: RainViewer's own `radar.past` list
already holds ~2h of history frames, so `rainviewer_radar.
fetch_reflectivity_sequence()` stitches the last `history_frames` of them
into the (T, H, W) stack pySTEPS needs — genuine motion estimated from
genuine consecutive radar frames, not a single "now" mosaic. Falls back to
the synthetic sequence (processing/synthetic_radar.py) on any live-fetch
failure (RainViewer down, fewer than 2 past frames available, network
error) — same fallback pattern as every other USE_LIVE_* source in this
project. `run_forecast`'s return dict reports which one actually happened
via `source`.
"""
import numpy as np
from pysteps import motion, nowcasts
from pysteps.utils import transformation

from nowcast.processing.synthetic_radar import generate_sequence, GRID_SIZE
from nowcast.configs.settings import get_region_bbox, USE_LIVE_RADAR


def _dbz_to_rainrate(dbz):
    """Marshall-Palmer Z-R relation: Z = 200 R^1.6 (dBZ -> mm/hr)."""
    z = 10 ** (dbz / 10.0)
    r = (z / 200.0) ** (1 / 1.6)
    return r


def _live_sequence(history_frames):
    """Real RainViewer reflectivity time series, regridded to GRID_SIZE
    over the active region's bbox. Raises on any failure — the caller
    decides how to fall back."""
    from nowcast.ingestion.rainviewer_radar import fetch_reflectivity_sequence

    stack, dt_minutes_list = fetch_reflectivity_sequence(
        n_frames=history_frames, grid_size=GRID_SIZE, bbox=get_region_bbox()
    )
    # pySTEPS' LK/extrapolation methods assume uniform frame spacing;
    # RainViewer's cadence is usually but not guaranteed exactly 10min, so
    # use the median observed spacing as the single dt fed to pySTEPS
    # rather than pretending every gap was identical.
    dt_minutes = float(np.median(dt_minutes_list)) if dt_minutes_list else 10.0
    return stack, dt_minutes


def run_forecast(n_lead_steps=36, dt_minutes=10, history_frames=6):
    """Returns dict: {timestamps_min, dbz_forecast, rainrate_forecast, motion_field}.

    dbz_forecast / rainrate_forecast: list of (GRID_SIZE, GRID_SIZE) arrays,
    one per lead step, each `dt_minutes` after the last observed frame.

    `dt_minutes` is the requested synthetic-mode cadence; in live mode it's
    overridden by the real measured spacing between RainViewer frames (see
    _live_sequence) since that's what the motion field was actually
    estimated over.
    """
    source = "synthetic"
    if USE_LIVE_RADAR:
        try:
            stack, dt_minutes = _live_sequence(history_frames)
            source = "rainviewer-live"
        except Exception as exc:
            print(f"[pysteps_baseline] live RainViewer sequence failed ({exc}), falling back to synthetic")
            frames, _ = generate_sequence(n_frames=history_frames, dt_minutes=dt_minutes)
            stack = np.stack(frames, axis=0)  # (T, H, W), dBZ
    else:
        frames, _ = generate_sequence(n_frames=history_frames, dt_minutes=dt_minutes)
        stack = np.stack(frames, axis=0)  # (T, H, W), dBZ

    # pySTEPS optical flow expects reflectivity in dB-like units; convert to
    # rain rate then to dB-R domain, which is what its transform utilities assume.
    rainrate_stack = _dbz_to_rainrate(stack)
    rainrate_db, _ = transformation.dB_transform(rainrate_stack, threshold=0.1, zerovalue=-15.0)
    rainrate_db = np.nan_to_num(rainrate_db, nan=-15.0, neginf=-15.0)

    oflow = motion.get_method("LK")
    motion_field = oflow(rainrate_db)

    extrapolate = nowcasts.get_method("extrapolation")
    forecast_db = extrapolate(rainrate_db[-1], motion_field, n_lead_steps)

    forecast_rainrate = transformation.dB_transform(forecast_db, inverse=True, threshold=-15.0, zerovalue=0.0)[0]
    forecast_rainrate = np.clip(np.nan_to_num(forecast_rainrate, nan=0.0), 0, None)

    # back to dBZ for hazard thresholds that use reflectivity (4c hail rule)
    forecast_dbz = 10 * np.log10(np.clip(200 * forecast_rainrate ** 1.6, 1e-3, None))

    timestamps_min = [(i + 1) * dt_minutes for i in range(n_lead_steps)]
    return {
        "timestamps_min": timestamps_min,
        "dbz_forecast": [f.astype(np.float32) for f in forecast_dbz],
        "rainrate_forecast": [f.astype(np.float32) for f in forecast_rainrate],
        "motion_field": motion_field,
        "bbox": get_region_bbox(),
        "grid_size": GRID_SIZE,
        "source": source,
    }


def cloudburst_cells(forecast: dict, threshold_mm_hr=15.0):
    """Grid cells (lat, lon, lead_min, rainrate) exceeding the very-heavy-rain threshold."""
    lon_min, lat_min, lon_max, lat_max = forecast["bbox"]
    n = forecast["grid_size"]
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)

    hits = []
    for lead_min, rr in zip(forecast["timestamps_min"], forecast["rainrate_forecast"]):
        ys, xs = np.where(rr >= threshold_mm_hr)
        for y, x in zip(ys, xs):
            hits.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "lead_minutes": lead_min,
                "rainrate_mm_hr": round(float(rr[y, x]), 1),
            })
    return hits


if __name__ == "__main__":
    fc = run_forecast()
    print(f"forecast steps: {fc['timestamps_min']}")
    print(f"max rainrate per step (mm/hr): {[round(float(f.max()), 1) for f in fc['rainrate_forecast']]}")
    print(f"cloudburst hits: {len(cloudburst_cells(fc))}")
