"""Train the Vidović-style hybrid ANFIS (3 epochs) across all 8 membership-function
types, for both 3- and 4-indicator models.

Reports train / val / test RMSE on the same [0, 1] output scale used by Vidović,
so the numbers are directly comparable to his Table 40 figures. Writes
`hybrid_results.json`, which `generate_report.py` turns into a markdown report.

Notes on faithfulness to Vidović (2018):
  * His optimal model: hybrid learning, linear (first-order Sugeno) output, 3 epochs.
  * His Table 40 sweeps 8 input MF types; hybrid error is essentially identical
    across all of them (systems are insensitive to MF shape on this data).
  * Internal training objective is squared error (least-squares consequent solve);
    RMSE is the *reported* metric. This mirrors his method exactly.
  * The 3-indicator (27-rule) model replicates his thesis. The 4-indicator
    (81-rule) model has NO counterpart in his thesis -- it is our own extension.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from src.engine import ANFIS


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Vidović's 8 Table-40 input MF types -> this library's accepted names.
# (All 8 confirmed to resolve; none were rejected in earlier sweeps.)
VIDOVIC_MF_MAP = {
    "trimf": "triangular",
    "tramf": "trapezoidal",
    "gbellmf": "gbell",
    "gaussmf": "gaussian",
    "gauss2mf": "gaussian2",
    "pimf": "pimf",
    "dsigmf": "diffsigmoidal",
    "psigmf": "prodsigmoidal",
}

BASE_ARGS = {
    "num_epochs": 3,
    "learning_rate": 0.1,   # irrelevant for hybrid (LSE solves consequents); pinned for reproducibility
    "loss_function": "rmse",  # reported metric; hybrid's LSE consequent solve minimizes squared error regardless
    "optimizer": "hybrid",
    "shuffle": True,
    "n_mfs": 3,
}

results = []
for num_indices in (3, 4):
    for vidovic_name, lib_name in VIDOVIC_MF_MAP.items():
        print(f"\n{'=' * 70}\nn={num_indices}  {vidovic_name} -> {lib_name}  (hybrid, 3 epochs)\n{'=' * 70}")
        try:
            model = ANFIS(
                **BASE_ARGS,
                membership_functions=lib_name,
                num_indices=num_indices,
                sheet_dir="real",
            )
            model.train()
            row = {
                "num_indices": num_indices,
                "n_rules": int(model.model.model_.n_rules),
                "vidovic_mf": vidovic_name,
                "library_mf": lib_name,
                "train_rmse": rmse(model.Y_train, model.model.predict(model.X_train)),
                "val_rmse": rmse(model.Y_val, model.model.predict(model.X_val)),
                "test_rmse": rmse(model.Y_test, model.model.predict(model.X_test)),
                "n_train": int(len(model.Y_train)),
                "n_val": int(len(model.Y_val)),
                "n_test": int(len(model.Y_test)),
            }
            print(json.dumps(row, indent=2))
            results.append(row)
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}")
            results.append({
                "num_indices": num_indices,
                "vidovic_mf": vidovic_name,
                "library_mf": lib_name,
                "error": str(e),
            })

out_path = Path(__file__).parent / "hybrid_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out_path}")