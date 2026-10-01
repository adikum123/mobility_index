import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from src.engine import ANFIS

VIDOVIC_MF_MAP = {
    "trimf": "triangular", "tramf": "trapezoidal", "gbellmf": "gbell",
    "gaussmf": "gaussian", "gauss2mf": "gaussian2", "pimf": "pimf",
    "dsigmf": "diffsigmoidal", "psigmf": "prodsigmoidal",
}
EPOCHS_GRID = (3, 10, 100)   # his actual grid (Table 40), NOT 3/40/100
LR = 0.1                    # Matlab initial step size; irrelevant for hybrid, real for sgd

results = []
for num_indices in (3, 4):
    for vidovic_name, lib_name in VIDOVIC_MF_MAP.items():
        for opt in ("hybrid", "sgd"):
            for n_epochs in EPOCHS_GRID:
                label = f"n={num_indices} {vidovic_name:<9} {opt:<6} ep={n_epochs}"
                print(f"\n{'='*70}\n{label}\n{'='*70}")
                rec = {
                    "num_indices": num_indices, "vidovic_mf": vidovic_name,
                    "library_mf": lib_name, "optimizer": opt, "epochs": n_epochs,
                }
                try:
                    model = ANFIS(
                        num_indices=num_indices, num_epochs=n_epochs, learning_rate=LR,
                        membership_functions=lib_name, loss_function="mse",
                        optimizer=opt, shuffle=True, n_mfs=3, sheet_dir="real", verbose=False
                    )
                    model.train()
                    y_true = model.Y_test
                    y_pred = model.model.predict(model.X_test)
                    rec["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    print(f"  RMSE={rec['rmse']:.6f}")
                except Exception as e:
                    print(f"  SKIPPED: {type(e).__name__}: {e}")
                    rec["rmse"] = None
                results.append(rec)

# --- Table: MF rows, columns = optimizer × epoch ---
print(f"\n\n{'='*96}")
print("test RMSE — hybrid vs sgd across his epoch grid (3/10/100)")
print(f"{'='*96}")
def fmt(v): return f"{v:.5f}" if isinstance(v, float) else "SKIP"
for num_indices in (3, 4):
    print(f"\n--- {num_indices} indicators ---")
    print(f"{'MF':<11}{'hyb-3':>10}{'hyb-10':>10}{'hyb-100':>10}{'sgd-3':>10}{'sgd-10':>10}{'sgd-100':>10}")
    for vidovic_name in VIDOVIC_MF_MAP:
        cells = []
        for opt in ("hybrid", "sgd"):
            for ep in EPOCHS_GRID:
                match = next((r["rmse"] for r in results
                             if r["num_indices"] == num_indices and r["vidovic_mf"] == vidovic_name
                             and r["optimizer"] == opt and r["epochs"] == ep), None)
                cells.append(fmt(match))
        print(f"{vidovic_name:<11}" + "".join(f"{c:>10}" for c in cells))

Path(__file__).parent.joinpath("optimizer_epoch_comparison.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved to {Path(__file__).parent / 'optimizer_epoch_comparison.json'}")