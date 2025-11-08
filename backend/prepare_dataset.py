import os
import re
import inspect
from typing import Tuple, Optional, Union

import numpy as np
import xarray as xr

DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
VAR_NAME = "VHM0" # variable to predict for

_CF_UNITS_RE = re.compile(
    r"^\s*(seconds|second|minutes|minute|hours|hour|days|day)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?\s*$",
    flags=re.IGNORECASE,
)

def _parse_units_origin(units: str) -> Tuple[str, np.datetime64]:
    if not isinstance(units, str):
        return "s", np.datetime64("1970-01-01T00:00:00Z", "ns")
    m = _CF_UNITS_RE.match(units)
    if not m:
        return "s", np.datetime64("1970-01-01T00:00:00Z", "ns")
    unit_word = m.group(1).lower()
    date_part = m.group(2)
    time_part = m.group(3) or "00:00:00"
    td_unit = {"second": "s", "seconds": "s", "minute": "m", "minutes": "m",
               "hour": "h", "hours": "h", "day": "D", "days": "D"}[unit_word]
    origin = np.datetime64(f"{date_part}T{time_part}Z", "ns")
    return td_unit, origin

def _decode_cf_time(numeric, units: Optional[str]) -> np.ndarray:
    td_unit, origin = _parse_units_origin(units or "")
    num = np.asarray(numeric).astype("int64", copy=False)
    td = num.astype(f"timedelta64[{td_unit}]")
    time64 = origin + td
    return time64.astype("datetime64[ns]")

def save_to_zarr(da: xr.DataArray, zarr_path: str, mode: str = "w") -> None:
    ds = da.to_dataset(name=da.name or VAR_NAME)
    ds.to_zarr(zarr_path, mode=mode)

def open_from_zarr(zarr_path: str) -> xr.DataArray:
    ds = xr.open_zarr(zarr_path)
    name = VAR_NAME if VAR_NAME in ds.data_vars else list(ds.data_vars)[0]
    return ds[name]

def save_to_npy(da: xr.DataArray, npy_path: str) -> None:
    arr = np.asarray(da.astype("float32").fillna(0).data)
    arr = np.ascontiguousarray(arr)
    np.save(npy_path, arr)

def open_memmap(npy_path: str) -> np.memmap:
    return np.load(npy_path, mmap_mode="r")

def open_series(
    start: str,
    end: str,
    *,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    coarsen: int = 8,
    dtype: str = "float32",
    chunks: Optional[dict] = None,
    use_zarr: Optional[str] = None,
) -> xr.DataArray:
    if use_zarr and os.path.exists(use_zarr):
        da_z = open_from_zarr(use_zarr)
        da_z = _clip_time_bbox(da_z, start, end, lat_min, lat_max, lon_min, lon_max)
        if coarsen and coarsen > 1:
            da_z = da_z.coarsen(latitude=coarsen, longitude=coarsen, boundary="trim").mean()
        return da_z.astype(dtype)

    import copernicusmarine as cm

    try:
        sig = inspect.signature(cm.open_dataset)
        if "stream" in sig.parameters:
            ds = cm.open_dataset(DATASET_ID, stream="GRID")
        else:
            try:
                ds = cm.open_dataset(DATASET_ID)
            except TypeError:
                ds = cm.open_dataset(dataset_id=DATASET_ID)
    except Exception:
        try:
            ds = cm.open_dataset(DATASET_ID)
        except Exception as e:
            raise RuntimeError(f"copernicusmarine.open_dataset failed for '{DATASET_ID}': {e}")

    if isinstance(ds, (list, tuple)):
        if not ds:
            raise RuntimeError("copernicusmarine.open_dataset returned an empty list.")
        ds = ds[0]

    var_name = VAR_NAME if VAR_NAME in ds.data_vars else list(ds.data_vars)[0]

    ds = ds.sel(time=slice(np.datetime64(start), np.datetime64(end)))

    time_coord = ds["time"]
    if not np.issubdtype(time_coord.dtype, np.datetime64):
        units = time_coord.attrs.get("units", "")
        ds = ds.assign_coords(time=("time", _decode_cf_time(time_coord.values, units)))

    da = ds[var_name]
    da = _clip_time_bbox(da, start, end, lat_min, lat_max, lon_min, lon_max)

    if coarsen and coarsen > 1:
        da = da.coarsen(latitude=coarsen, longitude=coarsen, boundary="trim").mean()

    if chunks is None:
        chunks = {"time": 32, "latitude": 128, "longitude": 128}
    da = da.chunk(chunks)

    da = da.astype(dtype).where(np.isfinite(da), 0)

    order = [d for d in ["time", "latitude", "longitude"] if d in da.dims]
    return da.transpose(*order)


def _clip_time_bbox(
    da: xr.DataArray,
    start: str,
    end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    lat_min = max(float(da.latitude.min()), lat_min) if "latitude" in da.coords else lat_min
    lat_max = min(float(da.latitude.max()), lat_max) if "latitude" in da.coords else lat_max
    lon_min = max(float(da.longitude.min()), lon_min) if "longitude" in da.coords else lon_min
    lon_max = min(float(da.longitude.max()), lon_max) if "longitude" in da.coords else lon_max

    sel_kw = {}
    if "time" in da.coords:
        sel_kw["time"] = slice(np.datetime64(start), np.datetime64(end))
    if "latitude" in da.coords:
        sel_kw["latitude"] = slice(lat_min, lat_max)
    if "longitude" in da.coords:
        sel_kw["longitude"] = slice(lon_min, lon_max)
    return da.sel(**sel_kw)

import torch
from torch.utils.data import Dataset
import random

class WavesSeqDataset(Dataset):
    def __init__(
        self,
        da: Optional[xr.DataArray] = None,
        *,
        np_backend: Optional[Union[np.ndarray, np.memmap]] = None,
        T_in: int,
        K: int,
        vmax: float = 10.0,
        patch_hw: Optional[Tuple[int, int]] = (256, 256),
        stride: int = 1,
    ):
        if (da is None) == (np_backend is None):
            raise ValueError("Provide exactly one of (da) or (np_backend).")

        self.backend = "numpy" if np_backend is not None else "xarray"
        self.vmax = float(vmax)
        self.patch_hw = patch_hw if (patch_hw is None or len(patch_hw) == 2) else (256, 256)
        self.T_in = int(T_in)
        self.K = int(K)
        self.stride = max(1, int(stride))

        if self.backend == "xarray":
            if list(da.dims)[:3] != ["time", "latitude", "longitude"]:
                da = da.transpose("time", "latitude", "longitude")
            da = da.fillna(0)
            self.src = da
            self.T = int(da.sizes["time"])
            self.H = int(da.sizes["latitude"])
            self.W = int(da.sizes["longitude"])
        else:
            arr = np_backend  # shape [T,H,W], float32 memmap/ndarray
            if arr.ndim != 3:
                raise ValueError("np_backend must have shape [time, lat, lon]")
            self.src = arr
            self.T, self.H, self.W = map(int, arr.shape)

        total_needed = self.T_in + self.K
        self.n = 0 if self.T < total_needed else (self.T - total_needed) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def _sample_patch(self) -> Tuple[slice, slice]:
        if self.patch_hw is None:
            return slice(0, self.H), slice(0, self.W)
        ph, pw = min(self.patch_hw[0], self.H), min(self.patch_hw[1], self.W)
        top = random.randint(0, max(0, self.H - ph))
        left = random.randint(0, max(0, self.W - pw))
        return slice(top, top + ph), slice(left, left + pw)

    def __getitem__(self, idx: int):
        t0 = int(idx) * self.stride
        t1 = t0 + self.T_in
        t2 = t1 + self.K
        hs, ws = self._sample_patch()

        vmax = self.vmax if self.vmax > 0 else 1.0

        if self.backend == "xarray":
            x_da = self.src.isel(time=slice(t0, t1), latitude=hs, longitude=ws)
            y_da = self.src.isel(time=slice(t1, t2), latitude=hs, longitude=ws)
            x = np.asarray(x_da.data, dtype=np.float32, order="C")
            y = np.asarray(y_da.data, dtype=np.float32, order="C")
        else:
            x = np.asarray(self.src[t0:t1, hs, ws], dtype=np.float32, order="C")
            y = np.asarray(self.src[t1:t2, hs, ws], dtype=np.float32, order="C")

        x = np.clip(x / vmax, 0.0, 1.0)[:, None, :, :]
        y = np.clip(y / vmax, 0.0, 1.0)[:, None, :, :]
        return torch.from_numpy(x), torch.from_numpy(y)
