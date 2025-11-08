import json
import math
from typing import Any, Dict, Optional

RESULTS_PATH: str = "grid_results.json"
EXPORT_BEST_TO: Optional[str] = "best_params.json"


def find_best(results: Dict[str, Any]):
    best = None
    for rec in results.get("trials", []):
        status = rec.get("status", "unknown")
        metrics = rec.get("metrics", {})
        val = metrics.get("val_loss", None)
        if status != "ok" or val is None or not math.isfinite(val):
            continue
        if (best is None) or (val < best["metrics"]["val_loss"]):
            best = rec
    return best


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = find_best(data)
    if not best:
        print("No successful trials with a finite val_loss were found.")
        return

    print("\n=== Best Trial ===")
    print(f"trial_index: {best['trial_index']}")
    print(f"status     : {best['status']}")
    print(f"val_loss   : {best['metrics']['val_loss']:.6f}")
    print(f"duration   : {best['duration_sec']} sec")

    print("\nParameters:")
    for k, v in best["params"].items():
        print(f"  {k}: {v}")

    print("\nArtifacts:")
    for k, v in best.get("derived_paths", {}).items():
        print(f"  {k}: {v}")

    if EXPORT_BEST_TO:
        export_cfg = dict(best["params"])
        with open(EXPORT_BEST_TO, "w", encoding="utf-8") as f:
            json.dump(export_cfg, f, indent=2)
        print(f"\nExported best params to {EXPORT_BEST_TO}")


if __name__ == "__main__":
    main()
