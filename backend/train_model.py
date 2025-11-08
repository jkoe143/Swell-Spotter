import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prepare_dataset import (
    open_series, save_to_zarr, open_from_zarr,
    save_to_npy, open_memmap, WavesSeqDataset
)
from wave_predictor import WavePredictor


CONFIG = {
    # Data slice
    "start": "2024-01-01T00:00:00Z",
    "end":   "2024-01-05T00:00:00Z",
    "lat_min": -80.0,
    "lat_max":  90.0,
    "lon_min": -180.0,
    "lon_max":  180.0,

    # Spatial / temporal pipeline
    "coarsen": 12,
    "patch_hw": (128, 128),
    "vmax": 10.0,
    "Tin": 4,
    "K": 2,
    "stride": 2,

    # Dataset backend caching
    "zarr_path": None,
    "npy_cache_path": "waves_trimmed.npy",

    # Dataloaders
    "batch_size": 8,
    "num_workers": max(1, (os.cpu_count() or 2) // 2),
    "pin_memory": False,

    # Training
    "epochs": 10,
    "lr": 1e-3,
    "validate_every": 2,
    "early_stop_patience": 2,

    # CPU threading
    "torch_threads": max(1, (os.cpu_count() or 4) // 2),

    # Checkpoint
    "ckpt_path": "wave_ckpt.pt",
}


def _set_cpu_threads(n_threads: int):
    n = max(1, int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    print(f"CPU threading: OMP/MKL/torch threads = {n}")


def run_training(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_cpu_threads(cfg.get("torch_threads", 0))
    print(f"Device: {device}")

    npy_path = cfg.get("npy_cache_path")
    use_memmap = bool(npy_path)

    if use_memmap and os.path.exists(npy_path):
        print(f"Opening memmap cache: {npy_path}")
        arr = open_memmap(npy_path)   # [T,H,W], float32
        using_numpy_backend = True
    else:
        print("Opening dataset...")
        da = open_series(
            cfg["start"], cfg["end"],
            lat_min=cfg["lat_min"], lat_max=cfg["lat_max"],
            lon_min=cfg["lon_min"], lon_max=cfg["lon_max"],
            coarsen=cfg["coarsen"],
            dtype="float32",
            chunks=None,
            use_zarr=cfg.get("zarr_path"),
        )

        zarr_path = cfg.get("zarr_path")
        if zarr_path and not os.path.exists(zarr_path):
            try:
                print(f"Saving prepared data to Zarr: {zarr_path}")
                save_to_zarr(da, zarr_path, mode="w")
            except Exception as e:
                print(f"Warning: failed to save Zarr ({e}). Continuing...")

        if use_memmap:
            try:
                print(f"Materializing to .npy cache: {npy_path}")
                save_to_npy(da, npy_path)
                arr = open_memmap(npy_path)
                using_numpy_backend = True
            except Exception as e:
                print(f"Warning: failed to save .npy cache ({e}). Using xarray backend.")
                using_numpy_backend = False
        else:
            using_numpy_backend = False

    if using_numpy_backend:
        T, H, W = map(int, arr.shape)
        split_t = int(0.8 * T)
        train_arr = arr[0:split_t]
        val_arr   = arr[split_t:T]

        train_ds = WavesSeqDataset(
            np_backend=train_arr,
            T_in=cdf(cfg, "Tin"), K=cdf(cfg, "K"),
            vmax=cdf(cfg, "vmax"),
            patch_hw=cdf(cfg, "patch_hw"),
            stride=cdf(cfg, "stride"),
        )
        val_ds = WavesSeqDataset(
            np_backend=val_arr,
            T_in=cdf(cfg, "Tin"), K=cdf(cfg, "K"),
            vmax=cdf(cfg, "vmax"),
            patch_hw=cdf(cfg, "patch_hw"),
            stride=cdf(cfg, "stride"),
        )
    else:
        full_len = da.sizes["time"]
        split_t = int(0.8 * full_len)
        train_da = da.isel(time=slice(0, split_t))
        val_da   = da.isel(time=slice(split_t, None))

        train_ds = WavesSeqDataset(
            da=train_da,
            T_in=cdf(cfg, "Tin"), K=cdf(cfg, "K"),
            vmax=cdf(cfg, "vmax"),
            patch_hw=cdf(cfg, "patch_hw"),
            stride=cdf(cfg, "stride"),
        )
        val_ds = WavesSeqDataset(
            da=val_da,
            T_in=cdf(cfg, "Tin"), K=cdf(cfg, "K"),
            vmax=cdf(cfg, "vmax"),
            patch_hw=cdf(cfg, "patch_hw"),
            stride=cdf(cfg, "stride"),
        )

    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=(2 if cfg["num_workers"] > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=(2 if cfg["num_workers"] > 0 else None),
    )

    model = WavePredictor(T_in=cfg["Tin"], K=cfg["K"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val = float("inf")
    patience = int(cfg.get("early_stop_patience", 0))
    since_best = 0
    validate_every = max(1, int(cfg.get("validate_every", 1)))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)

            pred = model(x)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            n_batches += 1

        epoch_train = train_loss / max(1, n_batches)
        t1 = time.time()

        do_val = (epoch % validate_every == 0) or (epoch == cfg["epochs"])
        if do_val:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=False)
                    y = y.to(device, non_blocking=False)
                    pred = model(x)
                    val_loss += F.mse_loss(pred, y).item()
                    val_batches += 1
            epoch_val = val_loss / max(1, val_batches)

            improved = epoch_val < best_val - 1e-8
            if improved:
                best_val = epoch_val
                since_best = 0
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg, "val_loss": best_val},
                    cfg["ckpt_path"],
                )
                print(f"Epoch {epoch:03d} | train {epoch_train:.5f} | val {epoch_val:.5f} | {t1 - t0:.1f}s → saved {cfg['ckpt_path']}")
            else:
                since_best += 1
                print(f"Epoch {epoch:03d} | train {epoch_train:.5f} | val {epoch_val:.5f} | {t1 - t0:.1f}s (no improve {since_best}/{patience})")
                if patience and since_best >= patience:
                    print("Early stopping.")
                    break
        else:
            print(f"Epoch {epoch:03d} | train {epoch_train:.5f} | {t1 - t0:.1f}s")

    print("Done.")


def cdf(cfg: dict, key: str):
    return cfg[key]


if __name__ == "__main__":
    run_training(CONFIG)
