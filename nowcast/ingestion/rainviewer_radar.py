"""Real radar reflectivity via RainViewer (stand-in for MOSDAC volumetric DWR).

Free, unauthenticated, no API key: `https://api.rainviewer.com/public/weather-maps.json`
lists recent radar frames as slippy-map tile paths; India's coverage there is itself
built from IMD's public radar network, republished by RainViewer rather than pulled
straight from MOSDAC. We fetch the "Black and White" color scheme (scheme id 0), which
RainViewer defines as a direct linear encoding of dBZ into greyscale — not a rendered
color ramp — so this is genuine quantitative reflectivity, not a PNG-inversion guess:
  grey 1..127   -> dBZ = grey - 32   (rain)
  grey 129..255 -> dBZ = grey - 160  (snow)
  alpha 0       -> no data

RainViewer has no Doppler radial-velocity product (that needs a raw volumetric scan,
which no public aggregator exposes) — `velocity_ms` therefore stays synthetic even in
live mode, so the downburst hazard rule (needs a real velocity couplet) never becomes
"real" this way. Document this clearly wherever radar_puller output feeds into hazard
rules.
"""
import concurrent.futures
import os
import sys

import numpy as np
import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from nowcast.configs.settings import get_region_bbox

TILE_SIZE = 256
ZOOM = 10  # ~0.35 deg/tile at the equator, well under REGION_BBOX's ~0.5 deg extent
INDIA_ZOOM = 6  # ~5.6 deg/tile — keeps an all-India fetch to ~30-40 tiles instead of thousands


def _latlon_to_tile(lat, lon, zoom):
    lat_rad = np.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n
    return x, y


def _tile_bounds(x_tile, y_tile, zoom):
    """Lon/lat bounds of tile (x_tile, y_tile) at `zoom` (top-left, bottom-right)."""
    n = 2.0**zoom
    lon_left = x_tile / n * 360.0 - 180.0
    lon_right = (x_tile + 1) / n * 360.0 - 180.0
    lat_top = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y_tile / n))))
    lat_bottom = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y_tile + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top


def _decode_dbz(png_bytes):
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)
    grey = arr[:, :, 0].astype(np.float32)
    alpha = arr[:, :, 3]

    dbz = np.where(grey <= 127, grey - 32, grey - 160)
    dbz = np.where(alpha == 0, 0.0, dbz)  # no-data -> 0 dBZ, matches synthetic baseline's floor
    return np.clip(dbz, 0, None)


def _weather_maps():
    resp = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    past_frames = data["radar"]["past"]
    if not past_frames:
        raise RuntimeError("RainViewer returned no past radar frames")
    return data["host"], past_frames


def _latest_frame_path():
    host, past_frames = _weather_maps()
    return host, past_frames[-1]["path"]  # most recent


def _fetch_frame_mosaic(host, frame_path, bbox, grid_size, zoom):
    """One RainViewer frame (`frame_path`, a specific past timestamp — not
    necessarily the latest), fetched as tiles, mosaicked, and regridded to
    (grid_size, grid_size) over `bbox`. Factored out of fetch_reflectivity
    so fetch_reflectivity_sequence can call it once per historical frame
    without duplicating the tile/mosaic/regrid logic."""
    lon_min, lat_min, lon_max, lat_max = bbox
    x_min, y_max = _latlon_to_tile(lat_min, lon_min, zoom)  # smaller lat -> larger y
    x_max, y_min = _latlon_to_tile(lat_max, lon_max, zoom)
    tx_range = range(int(np.floor(x_min)), int(np.floor(x_max)) + 1)
    ty_range = range(int(np.floor(y_min)), int(np.floor(y_max)) + 1)

    def _fetch_one(txy):
        tx, ty = txy
        url = f"{host}{frame_path}/{TILE_SIZE}/{zoom}/{tx}/{ty}/0/0_0.png"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return txy, _decode_dbz(r.content)

    tile_coords = [(tx, ty) for ty in ty_range for tx in tx_range]
    tiles = {}
    # Sequential is fine for a handful of tiles (single-region fetch); a
    # country-wide fetch is 30-40+ tiles, so fan them out concurrently to
    # keep total latency close to one round-trip instead of the sum of all.
    if len(tile_coords) <= 4:
        for txy in tile_coords:
            k, v = _fetch_one(txy)
            tiles[k] = v
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for k, v in pool.map(_fetch_one, tile_coords):
                tiles[k] = v

    tx0, ty0 = min(tx_range), min(ty_range)
    mosaic = np.zeros((len(ty_range) * TILE_SIZE, len(tx_range) * TILE_SIZE), dtype=np.float32)
    for (tx, ty), dbz in tiles.items():
        row0 = (ty - ty0) * TILE_SIZE
        col0 = (tx - tx0) * TILE_SIZE
        mosaic[row0 : row0 + TILE_SIZE, col0 : col0 + TILE_SIZE] = dbz

    # _tile_bounds returns (lon_left, lat_bottom, lon_right, lat_top) for ONE
    # tile. With multiple tile rows/columns, the overall mosaic bounds come
    # from different corners than a naive same-tile read: the NW tile
    # (tx0, ty0) gives the west edge AND the north edge (its OWN lat_bottom
    # is just the bottom of the top row, not the mosaic's south edge); the SE
    # tile gives the east edge AND the south edge. Mixing these up (taking
    # lat_bottom from the NW tile) silently produced a degenerate or reversed
    # latitude array whenever a region's bbox spanned >1 tile row — Kolkata's
    # bbox does, Pune's happened not to, which is why this only surfaced now.
    mosaic_lon_min, _, _, mosaic_lat_max = _tile_bounds(tx0, ty0, zoom)
    _, mosaic_lat_min, mosaic_lon_max, _ = _tile_bounds(max(tx_range), max(ty_range), zoom)

    src_lons = np.linspace(mosaic_lon_min, mosaic_lon_max, mosaic.shape[1])
    src_lats = np.linspace(mosaic_lat_max, mosaic_lat_min, mosaic.shape[0])  # row 0 = top = max lat

    from scipy.interpolate import RegularGridInterpolator

    lat_order = np.argsort(src_lats)
    interp = RegularGridInterpolator(
        (src_lats[lat_order], src_lons), mosaic[lat_order, :], bounds_error=False, fill_value=0.0
    )

    dst_lons = np.linspace(lon_min, lon_max, grid_size)
    dst_lats = np.linspace(lat_min, lat_max, grid_size)
    dst_lon_grid, dst_lat_grid = np.meshgrid(dst_lons, dst_lats)
    pts = np.stack([dst_lat_grid.ravel(), dst_lon_grid.ravel()], axis=-1)
    return interp(pts).reshape(dst_lat_grid.shape).astype(np.float32)


def fetch_reflectivity(grid_size=64, bbox=None, zoom=None):
    """Real reflectivity grid over `bbox` (defaults to the active region's
    storm-scale bbox), regridded to (grid_size, grid_size). Pass a coarser
    `zoom` for a large bbox (see INDIA_ZOOM / fetch_india_reflectivity) —
    tile count grows with (bbox extent / tile extent)^2, and ZOOM=10's
    ~0.35deg tiles would mean thousands of requests across all of India."""
    if bbox is None:
        bbox = get_region_bbox()
    if zoom is None:
        zoom = ZOOM
    host, frame_path = _latest_frame_path()
    return _fetch_frame_mosaic(host, frame_path, bbox, grid_size, zoom)


def fetch_india_reflectivity(grid_size):
    """Real reflectivity across all of India (settings.INDIA_BBOX), at a
    coarser zoom than the per-region fetch — used by hazard_india.py."""
    from nowcast.configs.settings import INDIA_BBOX

    return fetch_reflectivity(grid_size=grid_size, bbox=INDIA_BBOX, zoom=INDIA_ZOOM)


def fetch_reflectivity_sequence(n_frames=6, grid_size=64, bbox=None, zoom=None):
    """Real reflectivity TIME SERIES from RainViewer's own history buffer —
    the input pySTEPS actually needs (a single "now" frame has no motion to
    estimate). `weather-maps.json`'s `radar.past` list already holds the
    last ~2h of frames at ~10min cadence; this fetches the most recent
    `n_frames` of them (oldest first) and regrids each the same way
    fetch_reflectivity does for one frame.

    Returns (stack, dt_minutes_list) where `stack` is (T, grid_size,
    grid_size) float32 dBZ, oldest frame first, and `dt_minutes_list` is
    the real elapsed minutes between each consecutive pair of frames
    (RainViewer's cadence is usually but not guaranteed to be exactly
    10min, so this is measured from each frame's actual `time` field
    rather than assumed).
    """
    if bbox is None:
        bbox = get_region_bbox()
    if zoom is None:
        zoom = ZOOM
    host, past_frames = _weather_maps()
    if len(past_frames) < 2:
        raise RuntimeError(
            f"RainViewer only has {len(past_frames)} past frame(s) right now — need >=2 for a motion estimate"
        )
    chosen = past_frames[-n_frames:] if len(past_frames) >= n_frames else past_frames

    frames = [_fetch_frame_mosaic(host, f["path"], bbox, grid_size, zoom) for f in chosen]
    times = [f["time"] for f in chosen]  # unix seconds, ascending (oldest first)
    dt_minutes_list = [(times[i + 1] - times[i]) / 60.0 for i in range(len(times) - 1)]
    return np.stack(frames, axis=0), dt_minutes_list


def fetch_india_reflectivity_sequence(n_frames=6, grid_size=150):
    """All-India equivalent of fetch_reflectivity_sequence, for a real
    all-India pySTEPS cloudburst extrapolation (was previously impossible —
    see hazard_india.py's docstring — because only a single "now" mosaic
    was ever fetched; this fetches `n_frames` all-India mosaics instead).
    Several times the cost of a single all-India fetch (~15s each), so
    callers should cache aggressively."""
    from nowcast.configs.settings import INDIA_BBOX

    return fetch_reflectivity_sequence(n_frames=n_frames, grid_size=grid_size, bbox=INDIA_BBOX, zoom=INDIA_ZOOM)


if __name__ == "__main__":
    grid = fetch_reflectivity()
    print(f"reflectivity_dbz: min={grid.min():.1f} max={grid.max():.1f} mean={grid.mean():.1f}")
    stack, dts = fetch_reflectivity_sequence()
    print(f"sequence: {stack.shape[0]} frames, dt_minutes={[round(d, 1) for d in dts]}")
