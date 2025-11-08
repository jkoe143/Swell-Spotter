import sys
sys.dont_write_bytecode = True

from flask import Flask, jsonify, request, Response, make_response
from flask_cors import CORS
import mercantile
from PIL import Image
import xarray as xr
import numpy as np
import pandas as pd
import gzip
import hashlib
import json

import io
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Optional

from mapbox_vector_tile import encode as mvt_encode
from skimage import measure

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

import copernicusmarine as cm

import os
import torch
from torch import nn
from wave_predictor import WavePredictor

from functools import lru_cache
from threading import Lock

from pathfinder import find_path 
from scipy.interpolate import RegularGridInterpolator

DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
WAVE_VAR = "VHM0"

ANALYTIC_SCALE = 0.01
ANALYTIC_OFFSET = 0.0
ANALYTIC_UNITS = "m"

MIN_ZOOM = 0
MAX_ZOOM = 11

TILE_SIZE = 256
MVT_EXTENT = 4096  # Standard MVT extent

# Dataset spatial coverage (from describe)
DATA_LAT_MIN = -80.0
DATA_LAT_MAX = 90.0
DATA_LON_MIN = -180.0
DATA_LON_MAX = 180.0

# Simple in-process cache TTL (seconds) for tile PNGs
TILE_CACHE_TTL = 300
MAX_TILE_CACHE = 4096

ALPHA = 255
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 12.0

# Thread lock makes sure only one thread fetches dataset at a time
dataset_cache_lock = Lock()

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------------------
## Color Utils
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
## Time Utils
# ------------------------------------------------------------------------------
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

"""Generate list of ISO times between start and end with given step."""
def generate_time_range(start: str, end: str, step_hours: int = 3) -> List[str]:
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    times = []
    current = start_dt
    while current <= end_dt:
        rounded = round_to_3h(current.to_pydatetime().replace(tzinfo=timezone.utc))
        iso_time = rounded.isoformat()
        if not times or times[-1] != iso_time:  # Avoid duplicates
            times.append(iso_time)
        current += timedelta(hours=step_hours)

    return times

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

# ------------------------------------------------------------------------------
## Bound Utils
# ------------------------------------------------------------------------------
def clamp_bbox(w: float, s: float, e: float, n: float) -> Optional[Tuple[float, float, float, float]]:
    # Intersect with dataset coverage
    w = max(w, DATA_LON_MIN)
    e = min(e, DATA_LON_MAX)
    s = max(s, DATA_LAT_MIN)
    n = min(n, DATA_LAT_MAX)
    if e <= w or n <= s:
        return None
    return w, s, e, n

# WebMercator pixel center grids for an XYZ tile
def webmercator_lon_grid(z: int, x: int, size: int = TILE_SIZE) -> np.ndarray:
    n = 2**z
    cols = (np.arange(size) + 0.5) / size
    lon = (x + cols) / n * 360.0 - 180.0
    return lon.astype(np.float64)

def webmercator_lat_grid(z: int, y: int, size: int = TILE_SIZE) -> np.ndarray:
    # Pixel rows top->bottom -> lat (north->south)
    n = 2**z
    rows = (np.arange(size) + 0.5) / size
    y_norm = (y + rows) / n
    # Inverse WebMercator
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_norm)))
    lat_deg = np.degrees(lat_rad)

    return lat_deg[::-1].copy()

# ------------------------------------------------------------------------------
## Caching Utils
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

"""Generate ETag from key parts."""
def generate_etag(*parts) -> str:
    key = "-".join(str(p) for p in parts)
    return hashlib.md5(key.encode()).hexdigest()[:16]

"""Add caching headers to response."""
def cache_response(
        response: Response,
        max_age: int = 31536000,
        immutable: bool = True,
        etag: Optional[str] = None
) -> Response:
    cache_parts = [f"public", f"max-age={max_age}"]
    if immutable:
        cache_parts.append("immutable")

    response.headers["Cache-Control"] = ", ".join(cache_parts)
    response.headers["Access-Control-Allow-Origin"] = "*"

    if etag:
        response.headers["ETag"] = f'"{etag}"'

    return response

#-------------------------------------------------------------------------------
## ML Prediction Utils
#-------------------------------------------------------------------------------
MODEL_CKPT_PTH = "models/best_vhm0_model.pt"
PRED_CACHE_TTL = 300

class PredCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: dict = {}  # key -> (expires_at, ndarray)

    def get(self, key):
        item = self._store.get(key)
        if not item:
            return None
        expires, data = item
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return data
    
    def set(self, key, data):
        self._store[key] = (time.time() + self.ttl, data)

pred_cache = PredCache(PRED_CACHE_TTL)
_model_lock = Lock()
_model_bundle = None # tuple (model, ckpt_cfg, device, Tin, K, vmax)

def load_model():
    global _model_bundle
    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle
        
        ckpt_path = os.envioron.get(MODEL_CKPT_PTH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location = device)
        cfg = ckpt.get("cfg", {})
        Tin = int(cfg.get("Tin", 4))
        K = int(cfg.get("K", 2))
        vmax = float(cfg.get("vmax", 10.0))

        model = WavePredictor(T_in = Tin, K = K).to(device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.eval()

        _model_bundle = (model, cfg, device, Tin, K, vmax)
        return _model_bundle

# ------------------------------------------------------------------------------
## Core data access
# ------------------------------------------------------------------------------
# Cache global dataset to avoid refetching from Copernicus
@lru_cache(maxsize=4)
def cached_global_dataset(time_iso: str, var: str = WAVE_VAR):
    # Fetch the entire global dataset for the timestamp
    ds = cm.open_dataset(
        dataset_id=DATASET_ID,
        variables=[var],
        minimum_longitude=-180.0, # full longitude range
        maximum_longitude=180.0,
        minimum_latitude=-80.0, # full latitude range
        maximum_latitude=90.0,
        start_datetime=time_iso,
        end_datetime=time_iso,
        coordinates_selection_method="nearest", 
    )
    if "time" in ds.coords and len(ds.time) > 0:
        ts = pd.to_datetime(time_iso, utc=True)
        ts_naive = pd.Timestamp(ts.tz_convert("UTC").tz_localize(None))
        try:
            idx = ds.indexes["time"].get_indexer([ts_naive], method="nearest")[0]
            if idx >= 0:
                ds = ds.isel(time=int(idx))
        except Exception:
            ds = ds.isel(time=0)
    return ds

"""
    Fetch wave data for tile with buffer.
    Returns: (data_array, tile_bounds)
"""
def get_wave_data_for_tile(
    time_iso: str,
    z: int,
    x: int,
    y: int,
    var: str = WAVE_VAR,
    buffer_deg: float = 0.2
) -> Tuple[Optional[xr.DataArray], mercantile.LngLatBbox]:
    # Tile geographic bounds (WGS84)
    tile_bounds = mercantile.bounds(x, y, z)

    west = tile_bounds.west - buffer_deg
    south = tile_bounds.south - buffer_deg
    east = tile_bounds.east + buffer_deg
    north = tile_bounds.north + buffer_deg

    bbox = clamp_bbox(west, south, east, north)
    if bbox is None:
        return None, tile_bounds

    west, south, east, north = bbox
    with dataset_cache_lock:
        ds = cached_global_dataset(time_iso, var)

    if var not in ds.data_vars:
        return None, tile_bounds

    da = ds[var]
    if "time" in da.dims:
        da = da.drop_vars("time", errors="ignore")

    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_name = "latitude" if "latitude" in da.coords else "lat"

    # Ensure slices match dataset coord order
    lon_inc = float(da[lon_name][0]) < float(da[lon_name][-1])
    lat_inc = float(da[lat_name][0]) < float(da[lat_name][-1])

    lon_slice = slice(west, east) if lon_inc else slice(east, west)
    lat_slice = slice(south, north) if lat_inc else slice(north, south)

    part = da.sel({lon_name: lon_slice, lat_name: lat_slice})

    if part.size == 0:
        return None, tile_bounds

    return part, tile_bounds

# ------------------------------------------------------------------------------
## Single Band Raster Tiles
# ------------------------------------------------------------------------------
"""Encode physical values to uint16 with scale/offset."""
def encode_to_uint16(data: np.ndarray, scale: float, offset: float) -> np.ndarray:
    encoded = np.round((data - offset) / scale).astype(np.float32)
    encoded = np.clip(encoded, 0, 65535)
    encoded = np.where(np.isnan(data), 0, encoded)  # NaN -> 0
    return encoded.astype(np.uint16)

"""Render 16-bit grayscale analytic tile."""
def render_analytic_tile(
        time_iso: str,
        z: int,
        x: int,
        y: int,
        var: str = WAVE_VAR
) -> Tuple[bytes, Dict[str, str]]:

    part, tile_bound = get_wave_data_for_tile(time_iso, z, x, y, var)

    if part is None or part.size == 0:
        # Return transparent 16-bit tile
        arr = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
        img = Image.fromarray(arr, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=6)
        return buf.getvalue(), {}

    # Get coordinate names
    lon_name = "longitude" if "longitude" in part.coords else "lon"
    lat_name = "latitude" if "latitude" in part.coords else "lat"

    # Build target grids
    lon_grid = webmercator_lon_grid(z, x, TILE_SIZE)
    lat_grid = webmercator_lat_grid(z, y, TILE_SIZE)

    # Interpolate
    part_interp = part.interp(
        {
            lon_name: xr.DataArray(lon_grid, dims=("x_tile",)),
            lat_name: xr.DataArray(lat_grid, dims=("y_tile",)),
        },
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    arr = np.asarray(part_interp.values, dtype=np.float32)
    arr = np.flipud(arr)

    rgba = colorize(arr, DEFAULT_VMIN, DEFAULT_VMAX, alpha=ALPHA)

    #
    # arr_u16 = encode_to_uint16(arr, ANALYTIC_SCALE, ANALYTIC_OFFSET)
    #
    # # Create 16-bit grayscale PNG
    # img = Image.fromarray(arr_u16, "I;16")
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    headers = {
        "X-Scale": str(ANALYTIC_SCALE),
        "X-Offset": str(ANALYTIC_OFFSET),
        "X-Units": ANALYTIC_UNITS,
    }
    return buf.getvalue(), headers

# ------------------------------------------------------------------------------
## Vector Isobands
# ------------------------------------------------------------------------------
"""Generate isoband polygons from raster data."""
def generate_isobands(
        data: np.ndarray,
        west: float,
        south: float,
        east: float,
        north: float,
        levels: List[float],
) -> List[Dict]:

    if data.size == 0 or not np.isfinite(data).any():
        return []

    height, width = data.shape
    # Replace NaN with a value below minimum
    data_filled = np.where(np.isnan(data), -9999, data)
    features = []

    for i in range(len(levels) - 1):
        min_val = levels[i]
        max_val = levels[i + 1]

        # Create mask for this band
        mask = (data_filled >= min_val) & (data_filled < max_val)

        if not mask.any():
            continue

        # Find contours
        try:
            contours = measure.find_contours(mask.astype(float), 0.5)
        except Exception:
            continue

        polys_geo = []
        for contour in contours:
            if len(contour) < 4:
                continue

            # Convert pixel coords to GEOGRAPHIC coords
            geo_coords = []
            for row, col in contour:
                # row=0 is north, col=0 is west
                lon = west + (col / width) * (east - west)
                lat = north - (row / height) * (north - south)
                geo_coords.append((lon, lat))

                # Close polygon
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            if len(geo_coords) >= 4:
                try:
                    poly = Polygon(geo_coords)
                    if poly.is_valid and not poly.is_empty:
                        polys_geo.append(poly)
                except Exception:
                    continue
                    
        if polys_geo:
            # Merge overlapping polygons
            try:
                merged = unary_union(polys_geo)
                if merged.is_empty:
                    continue

                # Convert to features
                if isinstance(merged, Polygon):
                    geoms = [merged]
                elif isinstance(merged, MultiPolygon):
                    geoms = list(merged.geoms)
                else:
                    continue

                for geom in geoms:
                    features.append({
                        "geometry": mapping(geom),
                        "properties": {
                            "band_id": i,
                            "min": float(min_val),
                            "max": float(max_val),
                            "label": f"{min_val:.1f}–{max_val:.1f} m"
                        }
                    })
            except Exception:
                continue

    return features

"""Get appropriate isoband levels for zoom level."""
def get_isoband_levels(z: int) -> List[float]:
    if z <= 3:
        # Coarse: every 2m
        return [0, 2, 4, 6, 8, 10, 15, 20, 30]
    elif z <= 6:
        # Medium: every 1m
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    else:
        # Fine: every 0.5m
        return [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

"""Render isobands as Mapbox Vector Tile."""
def render_isobands_mvt(
        time_iso: str,
        z: int,
        x: int,
        y: int,
        var: str = WAVE_VAR
) -> bytes:

    part, tile_bounds = get_wave_data_for_tile(
        time_iso, z, x, y, var, buffer_deg=0.1
    )

    if part is None or part.size == 0:
        return create_empty_mvt()

    # Get raw data
    data = np.asarray(part.values, dtype=np.float32)

    lon_name = "longitude" if "longitude" in part.coords else "lon"
    lat_name = "latitude" if "latitude" in part.coords else "lat"

    data_west = float(part[lon_name].min())
    data_east = float(part[lon_name].max())
    data_south = float(part[lat_name].min())
    data_north = float(part[lat_name].max())

    # Generate isobands
    levels = get_isoband_levels(z)
    features = generate_isobands(
        data, 
        data_west, data_south, data_east, data_north,
        levels
    )

    if not features:
        return create_empty_mvt()

    # Encode as MVT
    layers = {
        "name": "isobands",
        "features": features
    }

    # MVT expects tile coordinates (0-4096)
    # We need to convert geographic coords to tile coords
    tile_data = mvt_encode(
        layers,
        default_options={
            "quantize_bounds": (
                tile_bounds.west,
                tile_bounds.south,
                tile_bounds.east,
                tile_bounds.north
            ),
            "y_coord_down": True,
            "extents": MVT_EXTENT,
        }
    )

    return gzip.compress(tile_data)

# ------------------------------------------------------------------------------
## Vector Gridded Coverage (MVT)
# ------------------------------------------------------------------------------
"""Get grid cell count for zoom level."""
def get_grid_resolution(z: int) -> int:
    if z <= 2:
        return 4  # 4x4 grid
    elif z <= 4:
        return 8
    elif z <= 6:
        return 16
    elif z <= 8:
        return 32
    else:
        return 64


"""Render gridded coverage as MVT with quantized values."""
def render_grid_mvt(
        time_iso: str,
        z: int,
        x: int,
        y: int,
        var: str = WAVE_VAR
) -> bytes:

    part, tile_bounds = get_wave_data_for_tile(time_iso, z, x, y, var)

    if part is None or part.size == 0:
        return create_empty_mvt()

    # Get coordinate names
    lon_name = "longitude" if "longitude" in part.coords else "lon"
    lat_name = "latitude" if "latitude" in part.coords else "lat"

    # Determine grid resolution
    grid_size = get_grid_resolution(z)

    west, south, east, north = tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north

    # Create regular grid
    lon_grid = np.linspace(west, east, grid_size + 1)
    lat_grid = np.linspace(south, north, grid_size + 1)

    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_w = lon_grid[j]
            cell_e = lon_grid[j + 1]
            cell_s = lat_grid[i]
            cell_n = lat_grid[i + 1]

            # Sample data in cell (use mean)
            try:
                cell_data = part.sel({
                    lon_name: slice(cell_w, cell_e),
                    lat_name: slice(cell_s, cell_n)
                })

                if cell_data.size == 0:
                    continue

                mean_val = float(cell_data.mean().values)
                if not np.isfinite(mean_val):
                    continue

                # Quantize to 0-255
                h_q = int(np.clip(mean_val / ANALYTIC_SCALE, 0, 255))

                poly = Polygon([
                    (cell_w, cell_s),
                    (cell_e, cell_s),
                    (cell_e, cell_n),
                    (cell_w, cell_n),
                    (cell_w, cell_s)
                ])

                features.append({
                    "geometry": mapping(poly),
                    "properties": {
                        "h_q": h_q,
                        "value": round(mean_val, 2),
                        "i": i,
                        "j": j
                    }
                })
            except Exception:
                continue

    if not features:
        return create_empty_mvt()

    layers = {
        "name": "wave_grid",
        "features": features
    }

    tile_data = mvt_encode(
        layers,
        default_options={
            "quantize_bounds": (
                tile_bounds.west,
                tile_bounds.south,
                tile_bounds.east,
                tile_bounds.north
            ),
            "y_coord_down": True,
            "extents": MVT_EXTENT,
        }
    )

    return gzip.compress(tile_data)

"""Create an empty MVT tile."""
def create_empty_mvt() -> bytes:
    layers = {
        "name": "empty",
        "features": []
    }
    tile_data = mvt_encode(
        layers,
        default_options={
            "y_coord_down": False,
            "extents": MVT_EXTENT,
        }
    )
    return gzip.compress(tile_data)

# using haversine formula
def calculate_distance(route):
    total = 0.0
    
    for i in range(len(route) - 1):
        lat1, lon1 = route[i]
        lat2, lon2 = route[i + 1]
        
        earth_radius = 6371  
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lat_diff = np.radians(lat2 - lat1)
        lon_diff = np.radians(lon2 - lon1)
        
        a = (np.sin(lat_diff / 2) **2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(lon_diff / 2) **2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = earth_radius * c
        total += distance
    
    return total

def wave_data_to_grid(wave_array, wave_lons, wave_lats):
    # use interpolation to get wave heights at specific coordinates
    interp  = RegularGridInterpolator(
        (wave_lats, wave_lons), # coordinates of interest
        wave_array, # wave height values
        bounds_error=False,
        fill_value=np.nan
    )
    # define routing grid coordinates
    new_lats = np.linspace(90, -90, 360) 
    new_lons = np.linspace(-180, 180, 720) 
        
    # create 2d coordinate meshgrid 
    mesh_lon, mesh_lat = np.meshgrid(new_lons, new_lats)
    pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
    result = interp(pts).reshape(360, 720) # reshape to 2d grid
    return result   

def get_wave_data_for_route(from_port, to_port, time):
    try:
        # define bounds based on port locations
        min_lat = max(min(from_port['LATITUDE'], to_port['LATITUDE']) - 10, DATA_LAT_MIN)
        max_lat = min(max(from_port['LATITUDE'], to_port['LATITUDE']) + 10, DATA_LAT_MAX)
        min_lon = max(min(from_port['LONGITUDE'], to_port['LONGITUDE']) - 10, DATA_LON_MIN)
        max_lon = min(max(from_port['LONGITUDE'], to_port['LONGITUDE']) + 10, DATA_LON_MAX)
        
        # get data from Copernicus based on time
        data = cm.open_dataset(
            dataset_id=DATASET_ID,
            variables=[WAVE_VAR],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=time,
            end_datetime=time,
            coordinates_selection_method="nearest",
        )
        
        # makes sure the data returned is 2d
        if "time" in data.coords and len(data.time) > 0:
            data = data.isel(time=0)
        
        wave_values = data[WAVE_VAR].values.astype(np.float32)
        lon_name = "longitude" if "longitude" in data.coords else "lon"
        lat_name = "latitude" if "latitude" in data.coords else "lat"
        
        src_lons = data[lon_name].values
        src_lats = data[lat_name].values
        
        # transforms source wave grid to routing grid
        route_grid = wave_data_to_grid(wave_values, src_lons, src_lats)
        return route_grid
        
    except Exception as e:
        app.logger.exception(f"Error in fetching wave data: {e}")
        # return zeros if process fails, to allow routing without wave height consideration
        return np.zeros((360, 720), dtype=np.float32)

#-------------------------------------------------------------------------------
## Build input frames over a tile bbox and run model
#-------------------------------------------------------------------------------
def stack_last_Tin(time_iso: str, Tin: int, west: float, south: float, east: float, north: float, var: str):
    frames = []
    lat = None
    lon = None

    base = pd.to_datetime(time_iso, utc = True)
    times = [(base - pd.Timedelta(hours = 3 * (Tin - k - 1))).isoformat() for k in range(Tin)]

    for time in times:
        ds = cached_global_dataset(time, var)
        da = ds[var]
        lat = "latitude" if "latitude" in da.coords else "lat"
        lon = "longitude" if "longitude" in da.coords else "lon"

        lat_inc = float(da[lat][0]) < float(da[lat][-1])
        lon_inc = float(da[lon][0]) < float(da[lon][-1])
        lat_slice = slice(south, north) if lat_inc else slice(north, south)
        lon_slice = slice(west, east) if lon_inc else slice(east, west)

        part = da.sel({lat: lat_slice, lon: lon_slice})
        frames.append(np.asarray(part.values, dtype = np.float32))
        arr = np.stack(frames, axis = 0)  # (Tin, H, W)
        return arr, lat, lon

def predict_for_tile(time_iso: str, z: int, x: int, y: int, lead: int, var: str = WAVE_VAR) -> np.ndarray:
    model, ckpt_cfg, device, Tin, K, vmax = load_model()

    tile_bounds = mercantile.bounds(x, y, z)
    west = tile_bounds.west
    south = tile_bounds.south
    east = tile_bounds.east
    north = tile_bounds.north

    stack, lat, lon = stack_last_Tin(time_iso, Tin, west, south, east, north, var)
    x_np = np.clip(stack / max(vmax, 1e-6), 0.0, 1.0).astype(np.float32)  # normalize to [0, 1]
    x_tensor = torch.from_numpy(x_np[None, :, None, ...]).to(device)  # (1, Tin, 1, H, W)

    with torch.no_grad():
        y_tensor = model(x_tensor)  # (1, K, 1, H, W)
        y_lead = y_tensor[:, lead, 0] # (1, H, W)
        first_pred = y_lead.detach().cpu().numpy()[0]  # (H, W)

    pred_physical = first_pred * vmax  # denormalize
    
    lat_grid = webmercator_lat_grid(z, y, TILE_SIZE)
    lon_grid = webmercator_lon_grid(z, x, TILE_SIZE)

    da = xr.DataArray(
        pred_physical,
        dims = ("lat_src", "lon_src"),
        coords = {
            "lon_src": np.linspace(west, east, pred_physical.shape[1]),
            "lat_src": np.linspace(south, north, pred_physical.shape[0])
        },
    )

    da_interp = da.interp(
        lon_src = xr.DataArray(lon_grid, dims = ("x_tile",)),
        lat_src = xr.DataArray(lat_grid, dims = ("y_tile",)),
        method = "nearest",
        kwargs = {"fill_value": np.nan},
    )

    arr = np.asarray(da_interp.values, dtype = np.float32)
    arr = np.flipud(arr)
    return arr

def render_predicted_analytic_tile(time_iso: str, z: int, x: int, y: int, lead: int, var: str = WAVE_VAR):
    cache_key = ("pred-analytic", time_iso, z, x, y, var, int(lead))
    cached = pred_cache.get(cache_key)
    if cached is not None:
        return cached, {
            "X-Scale": str(ANALYTIC_SCALE),
            "X-Offset": str(ANALYTIC_OFFSET),
            "X-Units": ANALYTIC_UNITS,
            "X-Lead": str(lead),
        }
    
    try:
        arr_pred = predict_for_tile(time_iso, z, x, y, lead, var)
    except Exception as e:
        app.logger.exception(f"Prediction error: {e}")
        rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
        pred_cache.set(cache_key, png)

        headers = {
            "X-Scale": str(ANALYTIC_SCALE),
            "X-Offset": str(ANALYTIC_OFFSET),
            "X-Units": ANALYTIC_UNITS,
            "X-Lead": str(lead),
        }

    return arr_pred, headers

# ------------------------------------------------------------------------------
## Routes
# ------------------------------------------------------------------------------

@app.route('/api/ports', methods=['GET'])
def get_ports():
    try:
        with open('data/ports.json', 'r', encoding='utf-8') as f: 
            ports = json.load(f)
        return jsonify(ports)
    except FileNotFoundError:
        return jsonify({"error": "Ports file not found"}), 404

@app.route('/api/route', methods=['POST'])
def generate_route():
    try:
        request_data = request.get_json() 
        from_city = request_data.get('from')
        to_city = request_data.get('to')
        
        if not from_city or not to_city:
            return jsonify({"error": "Need both 'from' and 'to' port cities"}), 400
        
        # load ports from file
        with open('data/ports.json', 'r', encoding='utf-8') as f:
            ports = json.load(f)
        
        # find the ports by city name
        from_port = None
        to_port = None
        
        for port in ports:
            if port.get('CITY') == from_city:
                from_port = port
            if port.get('CITY') == to_city:
                to_port = port
        
        if not from_port:
            return jsonify({"error": f"Can't find port '{from_city}'"}), 404
        if not to_port:
            return jsonify({"error": f"Can't find port '{to_city}'"}), 404
        
        # Get current time for wave data
        time = request_data.get('time')
        if not time:
            return jsonify({"error": "Require time to get wave data"}), 400
        
        # Get wave data for this route
        wave_data = get_wave_data_for_route(from_port, to_port, time)
        
        # Find path with wave avoidance
        route = find_path(
            from_port['LATITUDE'], from_port['LONGITUDE'],
            to_port['LATITUDE'], to_port['LONGITUDE'],
            wave_data
        )
        
        if route is None:
            return jsonify({"error": "No ocean route found between these ports"}), 404
        
        total_distance = calculate_distance(route)
        
        return jsonify({
            "route": route,
            "distance_km": round(total_distance, 2),
            "from": from_port,
            "to": to_port,
            "waypoints": len(route)
        })
    
    except Exception as e:
        app.logger.exception(f"Error generating route: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return {"status": "ok", "dataset_id": DATASET_ID}

@app.get("/capabilities")
def capabilities():
    base_url = request.host_url.rstrip("/")
    return {
        "products": {
            "analytic": f"{base_url}/tiles/waves-analytic/{{time}}/{{z}}/{{x}}/{{y}}.png",
            "isobands": f"{base_url}/tiles/waves-isobands/{{time}}/{{z}}/{{x}}/{{y}}.pbf",
            "grid": f"{base_url}/tiles/waves-grid/{{time}}/{{z}}/{{x}}/{{y}}.pbf"
        },
        "tilejson": {
            "analytic": f"{base_url}/tilejson/waves-analytic.json",
            "isobands": f"{base_url}/tilejson/waves-isobands.json",
            "grid": f"{base_url}/tilejson/waves-grid.json"
        },
        "dataset_id": DATASET_ID,
        "variables": [WAVE_VAR],
        "step_hours": 3,
        "coverage": {
            "lon": [DATA_LON_MIN, DATA_LON_MAX],
            "lat": [DATA_LAT_MIN, DATA_LAT_MAX]
        }
    }

#
## Tile endpoints
#
"""Serve 16-bit analytic raster tile."""
@app.get("/tiles/waves-analytic/<time>/<int:z>/<int:x>/<int:y>.png")
def waves_analytic_tile(time: str, z: int, x: int, y: int):
    var = request.args.get("var", WAVE_VAR)
    cache_key = ("analytic", time, z, x, y, var)
    cached = tile_cache.get(cache_key)

    if cached:
        resp = make_response(cached)
        resp.headers["Content-Type"] = "image/png"
        resp.headers["X-Scale"] = str(ANALYTIC_SCALE)
        resp.headers["X-Offset"] = str(ANALYTIC_OFFSET)
        resp.headers["X-Units"] = ANALYTIC_UNITS
        etag = generate_etag("analytic", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    try:
        png_bytes, headers = render_analytic_tile(time, z, x, y, var)
        tile_cache.set(cache_key, png_bytes)

        resp = make_response(png_bytes)
        resp.headers["Content-Type"] = "image/png"

        for key, value in headers.items():
            resp.headers[key] = value

        etag = generate_etag("analytic", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    except Exception as e:
        app.logger.exception(f"Analytic tile error: {e}")
        # Return empty tile
        arr = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
        img = Image.fromarray(arr, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = make_response(buf.getvalue())
        resp.headers["Content-Type"] = "image/png"
        return cache_response(resp, max_age=60, immutable=False)


"""Serve isobands vector tile."""
@app.get("/tiles/waves-isobands/<time>/<int:z>/<int:x>/<int:y>.pbf")
def waves_isobands_tile(time: str, z: int, x: int, y: int):
    var = request.args.get("var", WAVE_VAR)
    cache_key = ("isobands", time, z, x, y, var)
    cached = tile_cache.get(cache_key)

    if cached:
        resp = make_response(cached)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        etag = generate_etag("isobands", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    try:
        mvt_bytes = render_isobands_mvt(time, z, x, y, var)
        tile_cache.set(cache_key, mvt_bytes)

        resp = make_response(mvt_bytes)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        etag = generate_etag("isobands", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    except Exception as e:
        app.logger.exception(f"Isobands tile error: {e}")
        empty = create_empty_mvt()
        resp = make_response(empty)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        return cache_response(resp, max_age=60, immutable=False)


"""Serve gridded coverage vector tile."""
@app.get("/tiles/waves-grid/<time>/<int:z>/<int:x>/<int:y>.pbf")
def waves_grid_tile(time: str, z: int, x: int, y: int):
    var = request.args.get("var", WAVE_VAR)
    cache_key = ("grid", time, z, x, y, var)
    cached = tile_cache.get(cache_key)

    if cached:
        resp = make_response(cached)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        etag = generate_etag("grid", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    try:
        mvt_bytes = render_grid_mvt(time, z, x, y, var)
        tile_cache.set(cache_key, mvt_bytes)

        resp = make_response(mvt_bytes)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        etag = generate_etag("grid", time, z, x, y, var, "v1")
        return cache_response(resp, etag=etag)

    except Exception as e:
        app.logger.exception(f"Grid tile error: {e}")
        empty = create_empty_mvt()
        resp = make_response(empty)
        resp.headers["Content-Type"] = "application/x-protobuf"
        resp.headers["Content-Encoding"] = "gzip"
        return cache_response(resp, max_age=60, immutable=False)

"""Serve predicted analytic raster tile."""
@app.get("/tiles/pred-waves-analytic/<time>/<int:z>/<int:x>/<int:y>.png")
def pred_waves_analytic_tile(time: str, z: int, x: int, y: int):
    var = request.args.get("var", WAVE_VAR)
    try:
        lead = int(request.args.get("lead", "0"))
    except Exception:
        return jsonify({"error": "Invalid lead parameter"}), 400
    
    try:
        png_bytes, headers = render_predicted_analytic_tile(time, z, x, y, lead, var)
        resp = make_response(png_bytes)
        resp.headers["Content-Type"] = "image/png"
        for key, value in headers.items():
            resp.headers[key] = value
        etag = generate_etag("pred-analytic", time, z, x, y, var, str(lead), "v1")
        return cache_response(resp, etag=etag)
    except Exception as e:
        app.logger.exception(f"Pred analytic tile error: {e}")
        # Return empty tile
        rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
        resp = make_response(buf.getvalue())
        resp.headers["Content-Type"] = "image/png"
        return cache_response(resp, max_age=60, immutable=False)

#
## TileJson endpoints
#
"""TileJSON for analytic single-band tiles."""
@app.get("/tilejson/waves-analytic.json")
def waves_analytic_tilejson():
    base_url = request.host_url.rstrip("/")

    tj = {
        "tilejson": "3.0.0",
        "name": "Wave Height Analytic (16-bit)",
        "description": "Single-band 16-bit raster tiles for client-side colorization",
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [
            f"{base_url}/tiles/waves-analytic/{{time}}/{{z}}/{{x}}/{{y}}.png?var={WAVE_VAR}"
        ],
        "minzoom": MIN_ZOOM,
        "maxzoom": MAX_ZOOM,
        "bounds": [DATA_LON_MIN, DATA_LAT_MIN, DATA_LON_MAX, DATA_LAT_MAX],
        "center": [0, 0, 2],
        "format": "png",
        "encoding": {
            "type": "uint16",
            "scale": ANALYTIC_SCALE,
            "offset": ANALYTIC_OFFSET,
            "units": ANALYTIC_UNITS,
            "formula": "value = offset + scale × pixel_value"
        },
        "attribution": "Copernicus Marine Service",
        "dataset_id": DATASET_ID,
        "variable": WAVE_VAR,
        "step_hours": 3
    }

    resp = make_response(jsonify(tj))
    return cache_response(resp, max_age=3600, immutable=False)


"""TileJSON for isoband vector tiles."""
@app.get("/tilejson/waves-isobands.json")
def waves_isobands_tilejson():
    base_url = request.host_url.rstrip("/")

    tj = {
        "tilejson": "3.0.0",
        "name": "Wave Height Isobands",
        "description": "Contour polygons for wave height visualization",
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [
            f"{base_url}/tiles/waves-isobands/{{time}}/{{z}}/{{x}}/{{y}}.pbf?var={WAVE_VAR}"
        ],
        "minzoom": MIN_ZOOM,
        "maxzoom": MAX_ZOOM,
        "bounds": [DATA_LON_MIN, DATA_LAT_MIN, DATA_LON_MAX, DATA_LAT_MAX],
        "center": [0, 0, 2],
        "format": "pbf",
        "vector_layers": [
            {
                "id": "isobands",
                "description": "Wave height contour bands",
                "fields": {
                    "band_id": "Number",
                    "min": "Number",
                    "max": "Number",
                    "label": "String"
                },
                "minzoom": MIN_ZOOM,
                "maxzoom": MAX_ZOOM
            }
        ],
        "attribution": "Copernicus Marine Service",
        "dataset_id": DATASET_ID,
        "variable": WAVE_VAR,
        "step_hours": 3
    }

    resp = make_response(jsonify(tj))
    return cache_response(resp, max_age=3600, immutable=False)


"""TileJSON for gridded coverage vector tiles."""
@app.get("/tilejson/waves-grid.json")
def waves_grid_tilejson():
    base_url = request.host_url.rstrip("/")

    tj = {
        "tilejson": "3.0.0",
        "name": "Wave Height Grid",
        "description": "Gridded coverage for interactive analytics",
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [
            f"{base_url}/tiles/waves-grid/{{time}}/{{z}}/{{x}}/{{y}}.pbf?var={WAVE_VAR}"
        ],
        "minzoom": MIN_ZOOM,
        "maxzoom": MAX_ZOOM,
        "bounds": [DATA_LON_MIN, DATA_LAT_MIN, DATA_LON_MAX, DATA_LAT_MAX],
        "center": [0, 0, 2],
        "format": "pbf",
        "vector_layers": [
            {
                "id": "wave_grid",
                "description": "Grid cells with quantized wave height",
                "fields": {
                    "h_q": "Number (0-255 quantized)",
                    "value": "Number (meters)",
                    "i": "Number (row index)",
                    "j": "Number (column index)"
                },
                "minzoom": MIN_ZOOM,
                "maxzoom": MAX_ZOOM
            }
        ],
        "quantization": {
            "scale": ANALYTIC_SCALE,
            "offset": ANALYTIC_OFFSET,
            "units": ANALYTIC_UNITS
        },
        "attribution": "Copernicus Marine Service",
        "dataset_id": DATASET_ID,
        "variable": WAVE_VAR,
        "step_hours": 3
    }

    resp = make_response(jsonify(tj))
    return cache_response(resp, max_age=3600, immutable=False)

"""TileJSON for predicted analytic raster tiles."""
def pred_waves_analytic_tilejson():
    base_url = request.host_url.rstrip("/")
    tj = {
        "tilejson": "3.0.0",
        "name": "Predicted Wave Height Analytic (16-bit)",
        "description": "Single-band 16-bit raster tiles for client-side colorization from ML predictions",
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [
            f"{base_url}/tiles/pred-waves-analytic/{{time}}/{{z}}/{{x}}/{{y}}.png?var={WAVE_VAR}&lead={{lead}}"
        ],
        "minzoom": MIN_ZOOM,
        "maxzoom": MAX_ZOOM,
        "bounds": [DATA_LON_MIN, DATA_LAT_MIN, DATA_LON_MAX, DATA_LAT_MAX],
        "center": [0, 0, 2],
        "format": "png",
        "encoding": {
            "type": "uint16",
            "scale": ANALYTIC_SCALE,
            "offset": ANALYTIC_OFFSET,
            "units": ANALYTIC_UNITS,
            "formula": "value = offset + scale × pixel_value"
        },
        "attribution": "Copernicus Marine Service",
        "dataset_id": DATASET_ID,
        "variable": WAVE_VAR,
        "step_hours": 3,
    }
    resp = make_response(jsonify(tj))
    return cache_response(resp, max_age=3600, immutable=False)

#
## Time endpoints
#
"""Get available time range."""
@app.get("/times")
def times():
    try:
        cat = cm.describe(dataset_id=DATASET_ID)
        tmin = tmax = None

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
                                raise StopIteration
    except StopIteration:
        pass
    except Exception as e:
        app.logger.exception("times endpoint failed: %s", e)
    return jsonify({
        "dataset_id": DATASET_ID,
        "time_min": tmin,
        "time_max": tmax,
        "step_hours": 3,
        "note": "Times are at 3-hour intervals"
    })

@app.get("/times/search")
def search_times():
    # Single time search - find nearest
    time_param = request.args.get("time")
    if time_param:
        try:
            t = pd.to_datetime(time_param, utc=True)
            rounded = round_to_3h(t.to_pydatetime().replace(tzinfo=timezone.utc))
            return jsonify({
                "requested": time_param,
                "nearest_available": rounded.isoformat(),
                "step_hours": 3,
                "note": "Times are rounded to nearest 3-hour interval"
            })
        except Exception as e:
            return jsonify({"error": f"Invalid time format: {str(e)}"}), 400

    # Range search
    start_param = request.args.get("start")
    end_param = request.args.get("end")

    if start_param and end_param:
        try:
            limit = min(int(request.args.get("limit", 100)), 1000)
            _times = generate_time_range(start_param, end_param)

            # Apply limit
            if len(_times) > limit:
                _times = _times[:limit]
                truncated = True
            else:
                truncated = False

            return jsonify({
                "start": start_param,
                "end": end_param,
                "count": len(_times),
                "truncated": truncated,
                "limit": limit,
                "times": _times,
                "step_hours": 3
            })
        except Exception as e:
            return jsonify({"error": f"Invalid range: {str(e)}"}), 400

    # No valid params
    return jsonify({
        "error": "Provide either 'time' for nearest search or 'start' and 'end' for range",
        "examples": {
            "nearest": "/times/search?time=2025-10-29T12:00:00Z",
            "range": "/times/search?start=2025-10-29T00:00:00Z&end=2025-10-30T00:00:00Z&limit=50"
        }
    }), 400

if __name__ == "__main__":
    app.run(debug=True)
