from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import mercantile
from PIL import Image
import xarray as xr
import numpy as np
import pandas as pd

import io
from datetime import datetime, timezone, timedelta
import time
from typing import Tuple, Optional

import copernicusmarine as cm

from functools import lru_cache
from threading import Lock

DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
WAVE_VAR = "VHM0"

DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 12.0
BLEED_PIXELS = 1

TILE_SIZE = 256
ALPHA = 255
# Small spatial padding to avoid edge effects when interpolating
BBOX_PAD_DEG = 0.15

# Thread lock makes sure only one thread fetches dataset at a time
dataset_cache_lock = Lock() 

# Dataset spatial coverage (from describe)
DATA_LAT_MIN = -80.0
DATA_LAT_MAX = 90.0
DATA_LON_MIN = -180.0
DATA_LON_MAX = 180.0

# Simple in-process cache TTL (seconds) for tile PNGs
TILE_CACHE_TTL = 300
MAX_TILE_CACHE = 4096

app = Flask(__name__)
CORS(app)

def round_to_3h(dt: datetime) -> datetime:
    # Dataset is every 3 hours; round to nearest 3h
    hour = dt.hour
    nearest = int(round(hour / 3.0) * 3) % 24
    base = dt.replace(minute=0, second=0, microsecond=0)
    delta_hours = (nearest - hour) % 24
    if delta_hours <= 12:
        return base + timedelta(hours=delta_hours)
    # If rounding "backwards" gives smaller delta (tie-breaker), step back
    return base - timedelta(hours=(24 - delta_hours))

def parse_time() -> str:
    t_str = request.args.get("time")
    if t_str:
        try:
            t = pd.to_datetime(t_str, utc=True)
        except Exception:
            t = pd.Timestamp.utcnow()
    else:
        t = pd.Timestamp.utcnow()
    t_rounded = round_to_3h(t.to_pydatetime().replace(tzinfo=timezone.utc))
    return t_rounded.isoformat()

def clamp_bbox(w: float, s: float, e: float, n: float) -> Optional[Tuple[float, float, float, float]]:
    # Intersect with dataset coverage
    w = max(w, DATA_LON_MIN)
    e = min(e, DATA_LON_MAX)
    s = max(s, DATA_LAT_MIN)
    n = min(n, DATA_LAT_MAX)
    if e <= w or n <= s:
        return None
    return w, s, e, n

def build_palette() -> np.ndarray:
    # Interpolate a 256-color palette from 5 Viridis-like anchors
    anchors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 97],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    steps = 256
    segs = anchors.shape[0] - 1
    ramp = np.zeros((steps, 3), dtype=np.uint8)
    for i in range(steps):
        t = i / (steps - 1)
        pos = t * segs
        k = min(int(pos), segs - 1)
        f = pos - k
        col = anchors[k] * (1 - f) + anchors[k + 1] * f
        ramp[i] = np.clip(col, 0, 255).astype(np.uint8)
    return ramp

PALETTE = build_palette()

def colorize(values: np.ndarray, vmin: float, vmax: float, alpha: int = ALPHA) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    # Handle degenerate scales
    denom = vmax - vmin
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    t = (values - vmin) / denom
    t = np.nan_to_num(t, nan=-1.0, neginf=-1.0, posinf=1.0)
    t = np.clip(t, 0.0, 1.0)
    idx = np.floor(t * (len(PALETTE) - 1)).astype(np.int16)
    rgba[..., :3] = PALETTE[idx]
    rgba[..., 3] = (np.isfinite(values)).astype(np.uint8) * int(alpha)
    return rgba

def nearest_time_select(ds: xr.Dataset, iso: str) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    ts = pd.to_datetime(iso, utc=True)
    ts_naive = pd.Timestamp(ts.tz_convert("UTC").tz_localize(None))
    # datetime64 route
    try:
        tv = ds["time"].values
        if np.issubdtype(np.asarray(tv).dtype, np.datetime64):
            target = np.datetime64(ts_naive.to_datetime64())
            idx = int(np.nanargmin(np.abs(tv - target)))
            return ds.isel(time=idx)
    except Exception:
        pass
    # Index route (works for cftime as well)
    try:
        idx2 = ds.indexes["time"].get_indexer([ts_naive], method="nearest")[0]
        if idx2 >= 0:
            return ds.isel(time=int(idx2))
    except Exception:
        pass
    return ds

# WebMercator pixel center grids for an XYZ tile
def webmercator_lon_grid(z: int, x: int, size: int = TILE_SIZE, bleed: int = 0) -> np.ndarray:
    n = 2**z
    cols = (np.arange(-bleed, size + bleed) + 0.5) / size
    lon = (x + cols) / n * 360.0 - 180.0
    return lon.astype(np.float64)

def webmercator_lat_grid(z: int, y: int, size: int = TILE_SIZE, bleed: int = 0) -> np.ndarray:
    # Pixel rows top->bottom -> lat (north->south)
    n = 2**z
    rows = (np.arange(-bleed, size + bleed) + 0.5) / size
    y_norm = (y + rows) / n
    # Inverse WebMercator
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_norm)))
    lat_deg_top_to_bottom = np.degrees(lat_rad)
    # For xarray interpolation (expects monotonic coord), use south->north
    lat_deg_south_to_north = lat_deg_top_to_bottom[::-1].copy()
    return lat_deg_south_to_north.astype(np.float64)

def transparent_png(w: int, h: int) -> bytes:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ------------------------------------------------------------------------------
# Lightweight in-memory tile cache (TTL)
# ------------------------------------------------------------------------------

class TileCache:
    def __init__(self, ttl: int, maxsize: int):
        self.ttl = ttl
        self.maxsize = maxsize
        self._store: dict = {}  # key -> (expires_at, bytes)

    def get(self, key: tuple) -> Optional[bytes]:
        item = self._store.get(key)
        if not item:
            return None
        expires, data = item
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return data

    def set(self, key: tuple, data: bytes):
        # Evict oldest if over capacity
        if len(self._store) >= self.maxsize:
            # Simple eviction: remove earliest expiring
            oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest_key, None)
        self._store[key] = (time.time() + self.ttl, data)

tile_cache = TileCache(TILE_CACHE_TTL, MAX_TILE_CACHE)

# ------------------------------------------------------------------------------
# Core data access and rendering
# ------------------------------------------------------------------------------

# Cache global dataset to avoid refetching from Copernicus
@lru_cache(maxsize=2) 
def cached_global_dataset(time_iso: str):
    # Fetch the entire global dataset for the timestamp
    ds = cm.open_dataset(
        dataset_id=DATASET_ID,
        variables=[WAVE_VAR],
        minimum_longitude=-180.0, # full longitude range
        maximum_longitude=180.0,
        minimum_latitude=-80.0, # full latitude range
        maximum_latitude=90.0,
        start_datetime=time_iso,
        end_datetime=time_iso,
        coordinates_selection_method="nearest", 
    )
    return nearest_time_select(ds, time_iso)

def open_wave_dataset(time_iso: str, west: float, south: float, east: float, north: float) -> xr.Dataset: 
    # Use cached global dataset
    with dataset_cache_lock:
        ds_global = cached_global_dataset(time_iso)
    lon_name = "longitude" if "longitude" in ds_global.coords else "lon"
    lat_name = "latitude" if "latitude" in ds_global.coords else "lat"
    
    return ds_global.sel({ # Return only the requested slice
        lon_name: slice(west, east),
        lat_name: slice(south, north)
    })

def render_tile_png(
    time_iso: str,
    z: int,
    x: int,
    y: int,
    vmin: float,
    vmax: float,
    var: str = WAVE_VAR,
) -> bytes:
    # Tile geographic bounds (WGS84)
    bounds = mercantile.bounds(x, y, z)
    west, south, east, north = bounds.west, bounds.south, bounds.east, bounds.north

    bbox = clamp_bbox(west, south, east, north)
    if bbox is None:
        return transparent_png(TILE_SIZE, TILE_SIZE)
    west, south, east, north = bbox

    ds = open_wave_dataset(time_iso, west, south, east, north)
    if var not in ds.data_vars:
        lowmap = {k.lower(): k for k in ds.data_vars}
        var = lowmap.get(var.lower(), None)
        if not var:
            return transparent_png(TILE_SIZE, TILE_SIZE)

    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0).drop_vars("time", errors="ignore")

    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
    if not lon_name or not lat_name:
        return transparent_png(TILE_SIZE, TILE_SIZE)

    # Ensure slices match dataset coord order
    lon_inc = float(da[lon_name][0]) < float(da[lon_name][-1])
    lat_inc = float(da[lat_name][0]) < float(da[lat_name][-1])
    lon_slice = slice(west, east) if lon_inc else slice(east, west)
    lat_slice = slice(south, north) if lat_inc else slice(north, south)
    part = da.sel({lon_name: lon_slice, lat_name: lat_slice})

    if part.size == 0:
        return transparent_png(TILE_SIZE, TILE_SIZE)

    # Build WebMercator-aligned target grids
    lon_grid = webmercator_lon_grid(z, x, TILE_SIZE, BLEED_PIXELS)  # west->east
    lat_grid = webmercator_lat_grid(z, y, TILE_SIZE, BLEED_PIXELS)  # south->north (monotonic)

    # Interpolate to WebMercator pixel centers
    part_interp = part.interp(
    {
            lon_name: xr.DataArray(lon_grid, dims=("x_ext",)),
            lat_name: xr.DataArray(lat_grid, dims=("y_ext",)),
        },
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    arr_ext = np.asarray(part_interp.values, dtype=np.float32)  # (y_ext, x_ext), south->north
    # Crop the bleed border back to 256x256
    arr = arr_ext[BLEED_PIXELS:-BLEED_PIXELS, BLEED_PIXELS:-BLEED_PIXELS]
    if not np.isfinite(arr).any():
        return transparent_png(TILE_SIZE, TILE_SIZE)

    # Flip vertically so row 0 is north (MapLibre expects top=North)
    arr = np.flipud(arr)

    # Colorize with fixed or user-provided scale
    rgba = colorize(arr, vmin, vmax, alpha=ALPHA)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "dataset_id": DATASET_ID, "var": WAVE_VAR}

@app.get("/capabilities")
def capabilities():
    return {
        "tiles": "/tiles/waves/{z}/{x}/{y}.png?time={ISO8601}&vmin=0&vmax=12&var=VHM0",
        "dataset_id": DATASET_ID,
        "variable": WAVE_VAR,
        "default_vmin": DEFAULT_VMIN,
        "default_vmax": DEFAULT_VMAX,
        "tile_size": TILE_SIZE,
        "coverage": {
            "lon": [DATA_LON_MIN, DATA_LON_MAX],
            "lat": [DATA_LAT_MIN, DATA_LAT_MAX],
        },
    }

@app.get("/times")
def times():
    # Return min/max and step (3h) from catalogue; no big list.
    try:
        cat = cm.describe(dataset_id=DATASET_ID)
        # Find time coordinate (from any part's variable)
        tmin = tmax = None
        step_ms = None
        for p in cat.products:
            for d in p.datasets:
                if d.dataset_id == DATASET_ID:
                    for v in d.versions:
                        for part in v.parts:
                            coords = part.get_coordinates()
                            if "time" in coords:
                                coord = coords["time"][0]
                                tmin = coord.minimum_value
                                tmax = coord.maximum_value
                                step_ms = coord.step
                                raise StopIteration
    except StopIteration:
        pass
    except Exception as e:
        app.logger.exception("times endpoint failed: %s", e)

    return {
        "dataset_id": DATASET_ID,
        "time_min": tmin,
        "time_max": tmax,
        "step_hours": 3,
        "note": "Use ?time=YYYY-MM-DDTHH:00:00Z; server rounds to nearest 3h.",
    }

@app.get("/debug/dataset")
def debug_dataset():
    t = parse_time()
    # Use a small global bbox to inspect quickly (world coarsely)
    west, south, east, north = -20, -10, 20, 10
    try:
        ds = open_wave_dataset(t, west, south, east, north)
        var = WAVE_VAR if WAVE_VAR in ds.data_vars else list(ds.data_vars)[0]
        out = {
            "time_param": t,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "coords": list(ds.coords),
            "vars": list(ds.data_vars),
            "picked_var": var,
            "lon_minmax": [
                float(ds["longitude"].min().values),
                float(ds["longitude"].max().values),
            ]
            if "longitude" in ds.coords
            else None,
            "lat_minmax": [
                float(ds["latitude"].min().values),
                float(ds["latitude"].max().values),
            ]
            if "latitude" in ds.coords
            else None,
        }
        return jsonify(out)
    except Exception as e:
        app.logger.exception("debug/dataset failed")
        return jsonify({"error": str(e)}), 500

@app.get("/tiles/waves/<int:z>/<int:x>/<int:y>.png")
def tile(z: int, x: int, y: int):
    time_iso = parse_time()
    vmin = request.args.get("vmin", type=float) or DEFAULT_VMIN
    vmax = request.args.get("vmax", type=float) or DEFAULT_VMAX
    var = request.args.get("var", default=WAVE_VAR)

    key = (z, x, y, time_iso, float(vmin), float(vmax), var)
    cached = tile_cache.get(key)
    if cached is not None:
        return _png_response(cached)

    try:
        png = render_tile_png(time_iso, z, x, y, vmin, vmax, var=var)
    except Exception:
        app.logger.exception("Tile render failed z=%s x=%s y=%s time=%s", z, x, y, time_iso)
        png = transparent_png(TILE_SIZE, TILE_SIZE)

    tile_cache.set(key, png)
    return _png_response(png)

def _png_response(png_bytes: bytes) -> Response:
    resp = Response(png_bytes, mimetype="image/png")
    # Client caching hint
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp

if __name__ == "__main__":
    app.run(debug=True)
