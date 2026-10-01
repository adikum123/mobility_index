"""Expert-split sensitivity experiment.

Runs the hybrid ANFIS N times, each with a different random shuffle of the expert
survey sheets across train/val/test, for both the 3- and 4-indicator models.
Measures train/val/test RMSE, MSE and R^2 per run and aggregates mean/std.

Requires: ANFIS.__init__ accepts `split_seed`, and _set_data_expert_split shuffles
sheet order when it is set (see anfis.py edit).

Design notes (so the numbers are interpretable):
  * Each run reshuffles which experts land in train/val/test. Whether the MF weight
    init also varies depends on how you wired random_state in anfis.py:
      - if random_state stays random.randint(...): std = COMBINED split + init variance
      - if random_state = self.split_seed: std = split variance only (fully reproducible)
    For hybrid on this data premise params are ~frozen, so init barely matters either way.
  * Shuffling is seeded per run (RUN_SEED_BASE + run_idx), so the experiment is reproducible.
  * MF is fixed to gauss2mf (gaussian2), the best-test performer from the prior sweep;
    all 8 MFs were near-identical, so one is representative.
"""

import json
import statistics
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src.engine import ANFIS

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_RUNS = 2000
MF = "gaussian2"          # library name for Vidovic's gauss2mf
RUN_SEED_BASE = 10_000
OUT = Path(__file__).parent / "split_experiment_results.json"


def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate(per_run, split, metric):
    vals = [r[split][metric] for r in per_run]
    return {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals),   # population std over the N runs
        "min": min(vals),
        "max": max(vals),
    }


results = {"config": {"n_runs": N_RUNS, "mf": MF, "seed_base": RUN_SEED_BASE}, "by_num_indices": {}}

for num_indices in (3, 4):
    print(f"\n{'#' * 70}\n# {num_indices} indicators, MF={MF}, {N_RUNS} runs\n{'#' * 70}")
    per_run = []
    for run_idx in range(N_RUNS):
        seed = RUN_SEED_BASE + run_idx
        print(f"\n--- run {run_idx + 1}/{N_RUNS} (n={num_indices}) ---")
        try:
            model = ANFIS(
                num_indices=num_indices,
                num_epochs=1,
                learning_rate=0.1,
                membership_functions=MF,
                loss_function="rmse",
                optimizer="hybrid",
                shuffle=True,
                n_mfs=3,
                sheet_dir="real",
                split_seed=seed,
                verbose=False
            )
            model.train()
            run_row = {
                "run": run_idx,
                "seed": seed,
                "train_experts": model.train_sheet_names,
                "val_experts": model.val_sheet_names,
                "test_experts": model.test_sheet_names,
                "train": metrics(model.Y_train, model.model.predict(model.X_train)),
                "val": metrics(model.Y_val, model.model.predict(model.X_val)),
                "test": metrics(model.Y_test, model.model.predict(model.X_test)),
            }
            per_run.append(run_row)
            # track best-by-test-rmse across all runs (both indicator counts)
            key = run_row["test"]["rmse"]
            if "best_by_test_rmse" not in results or key < results["best_by_test_rmse"]["test"]["rmse"]:
                results["best_by_test_rmse"] = {**run_row, "num_indices": num_indices}
        except Exception as e:
            print(f"  run {run_idx} FAILED: {type(e).__name__}: {e}")
            per_run.append({"run": run_idx, "seed": seed, "error": str(e)})

    ok = [r for r in per_run if "error" not in r]
    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = {m: aggregate(ok, split, m) for m in ("rmse", "mse", "r2")}

    results["by_num_indices"][str(num_indices)] = {
        "n_successful_runs": len(ok),
        "summary": summary,
        "runs": per_run,
    }

    print(f"\n=== {num_indices}-indicator summary over {len(ok)} runs ===")
    for split in ("train", "val", "test"):
        s = summary[split]
        print(f"  {split:5s}  RMSE {s['rmse']['mean']:.6f} ± {s['rmse']['std']:.6f}   "
              f"MSE {s['mse']['mean']:.6f} ± {s['mse']['std']:.6f}   "
              f"R2 {s['r2']['mean']:.4f} ± {s['r2']['std']:.4f}")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved {OUT}")