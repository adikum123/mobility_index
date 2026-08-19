import json
import random
from pathlib import Path

from src.engine import ANFIS

# "gaussian" and "triangular" are confirmed to work in this codebase already.
# The rest are common ANFIS naming conventions but UNVERIFIED against this
# specific anfis_toolbox version -- unsupported ones are caught and skipped
# below rather than crashing the sweep. Trim/extend this list once you've
# checked ANFISRegressor's accepted mf_type values locally.
MEMBERSHIP_FUNCTION_CANDIDATES = [
    "triangular",
    "gaussian",
    "trapezoidal",
    "bell",
    "sigmoid",
]

BASE_ANFIS_ARGS = {
    "num_epochs": 4,
    "learning_rate": 0.1,
    "loss_function": "mse",
    "optimizer": "hybrid",
    "shuffle": True,
    "n_mfs": 3,
}

results = []

for sheet_dir in ["real"]:
    print(f"Using: {sheet_dir} sheets")
    for num_indices in (3, 4):
        for mf in MEMBERSHIP_FUNCTION_CANDIDATES:
            # Pin the seed per num_indices (not per mf) so every membership
            # function is trained/tested on the SAME train/val/test sheet
            # split -- isolates the effect of mf type, same idea as the
            # thesis holding the dataset fixed across its 32 FIS configs.
            random.seed(1000 + num_indices)

            label = f"num_indices={num_indices}, mf={mf}"
            print(f"\n{'='*70}\n{label}\n{'='*70}")
            try:
                model = ANFIS(
                    **BASE_ANFIS_ARGS,
                    membership_functions=mf,
                    num_indices=num_indices,
                    sheet_dir=sheet_dir,
                )
                model.train()
                metrics = model.test()
            except Exception as e:
                print(f"SKIPPED ({mf}): {type(e).__name__}: {e}")
                results.append(
                    {"num_indices": num_indices, "membership_function": mf, "error": str(e)}
                )
                continue

            print(json.dumps(metrics, indent=4))
            results.append({"num_indices": num_indices, "membership_function": mf, **metrics})

# --- Summary, sorted best-first by RMSE (mirrors thesis Table 40 comparison) ---
print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
ok = sorted((r for r in results if "error" not in r), key=lambda r: r["rmse"])
for r in ok:
    print(
        f"num_indices={r['num_indices']:<2} mf={r['membership_function']:<12} "
        f"RMSE={r['rmse']:.6f}  R²={r['r2']:.6f}"
    )

skipped = [r["membership_function"] for r in results if "error" in r]
if skipped:
    print(f"\nSkipped (unsupported mf_type or other error): {sorted(set(skipped))}")

out_path = Path(__file__).parent / "mf_sweep_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nFull results saved to {out_path}")