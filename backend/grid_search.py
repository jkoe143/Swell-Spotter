import itertools
import json
import os
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import torch

from train_model import run_training, CONFIG as BASE_CONFIG

RESULTS_PATH: Optional[str] = "grid_results.json"
TRIALS_SLICE: Optional[Tuple[int, int]] = (605, 2592)

FIXED_OVERRIDES: Dict = {
    "epochs": 6,
    "validate_every": 2,
    "early_stop_patience": 2,
}

PARAM_GRID: Dict[str, List] = {
    "coarsen": [8, 12, 16],
    "patch_hw": [(256, 256), (192, 192), (128, 128)],
    "Tin": [3, 4, 6],
    "stride": [1, 2, 3],
    "vmax": [8.0, 10.0],

    "lr": [1e-2, 3e-3, 1e-3, 3e-4],
    "batch_size": [4, 8, 12, 16],

    "num_workers": [0, 2, 4],
    "torch_threads": [2, 4, 8],

    "env_WAVE_USE_DEPTHWISE": ["1", "0"],
}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_trials(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = list(itertools.product(*vals))
    return [dict(zip(keys, combo)) for combo in combos]


def split_params_and_env(trial_params: dict) -> tuple[dict, dict]:
    cfg_params, env_overrides = {}, {}
    for k, v in trial_params.items():
        if k.startswith("env_"):
            env_overrides[k[4:]] = str(v)
        else:
            cfg_params[k] = v
    return cfg_params, env_overrides


def cfg_for_trial(base_cfg: dict, trial_params: dict, trial_idx: int) -> tuple[dict, dict]:
    cfg_params, env_overrides = split_params_and_env(trial_params)

    cfg = deepcopy(base_cfg)

    for k, v in FIXED_OVERRIDES.items():
        cfg[k] = v

    for k, v in cfg_params.items():
        cfg[k] = v

    ensure_dir("ckpts")
    cfg["ckpt_path"] = os.path.join("ckpts", f"trial_{trial_idx:04d}.pt")

    coarsen = int(cfg.get("coarsen", 8))
    ensure_dir("caches")
    cfg["npy_cache_path"] = os.path.join("caches", f"waves_coarsen{coarsen}.npy")

    return cfg, env_overrides


def read_val_loss_from_ckpt(ckpt_path: str) -> float | None:
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return float(ckpt.get("val_loss", None))
    except Exception:
        return None


def set_env_bulk(env_overrides: dict) -> dict:
    prev = {}
    for k, v in env_overrides.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def restore_env(prev: dict):
    for k, old in prev.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def main():
    all_trials = make_trials(PARAM_GRID)
    total = len(all_trials)

    if TRIALS_SLICE is None:
        run_indices = list(range(total))
    else:
        s, e = TRIALS_SLICE
        run_indices = list(range(max(0, s), min(total - 1, e) + 1))

    print(f"Total combos: {total} | Running trials: {run_indices[0]}..{run_indices[-1]}")

    results = {
        "meta": {
            "total_combos": total,
            "run_indices": run_indices,
            "base_config_used": True,
            "fixed_overrides": FIXED_OVERRIDES,
            "param_grid": PARAM_GRID,
        },
        "trials": []
    }

    for i in run_indices:
        trial_params = all_trials[i]
        cfg, env_over = cfg_for_trial(BASE_CONFIG, trial_params, i)

        print("\n" + "=" * 80)
        print(f"Trial {i}:")
        print("  params:", {k: v for k, v in trial_params.items() if not k.startswith('env_')})
        if env_over:
            print("  env   :", env_over)
        print("=" * 80)

        prev_env = set_env_bulk(env_over)

        t0 = time.time()
        status = "ok"
        try:
            run_training(cfg)
        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
        t1 = time.time()

        restore_env(prev_env)

        ckpt_path = cfg["ckpt_path"]
        val_loss = read_val_loss_from_ckpt(ckpt_path) if status == "ok" else None

        rec = {
            "trial_index": i,
            "status": status,
            "duration_sec": round(t1 - t0, 3),
            "params": {k: v for k, v in trial_params.items() if not k.startswith("env_")},
            "env": env_over,
            "derived_paths": {
                "ckpt_path": ckpt_path,
                "npy_cache_path": cfg.get("npy_cache_path", None),
                "zarr_path": cfg.get("zarr_path", None),
            },
            "metrics": {"val_loss": val_loss},
        }
        results["trials"].append(rec)

        if RESULTS_PATH:
            try:
                with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

    if RESULTS_PATH:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
