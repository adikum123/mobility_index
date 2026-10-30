"""Train and compare hybrid ANFIS models for all 8 membership-function types,
for both the 3-indicator (27 rules) and 4-indicator (81 rules) models.

Every model uses the same expert split (8 train / 3 val / 3 test experts) and the
same random seed, so differences between rows come only from the MF type and the
number of indicators.

Outputs (in scripts/results/):
  * mf_comparison.json  -- full per-model metrics
  * mf_comparison.csv   -- flat table, one row per model
  * mf_comparison.tex   -- LaTeX table (test RMSE / R^2, 3 vs 4 indicators)

Run from the project root:
    python -m scripts.main
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src.engine import ANFIS

# MATLAB MF name -> anfis_toolbox name
MF_TYPES: dict[str, str] = {
    "trimf": "triangular",
    "tramf": "trapezoidal",
    "gbellmf": "gbell",
    "gaussmf": "gaussian",
    "gauss2mf": "gaussian2",
    "pimf": "pimf",
    "dsigmf": "diffsigmoidal",
    "psigmf": "prodsigmoidal",
}

NUM_INDICES = (3, 4)
SEED = 1034  # used for both the expert split and model initialisation
SHEET_DIR = "real"

BASE_ARGS = {
    "num_epochs": 3,
    "learning_rate": 0.1,  # no effect on the LSE consequent solve; pinned for reproducibility
    "loss_function": "rmse",
    "optimizer": "hybrid",
    "shuffle": True,
    "n_mfs": 3,
    "sheet_dir": SHEET_DIR,
    "split_seed": None,
    "verbose": False,
}

OUT_DIR = Path(__file__).parent / "results"
SPLITS = ("train", "val", "test")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {"rmse": float(np.sqrt(mse)), "mse": mse, "r2": float(r2_score(y_true, y_pred))}


def train_one(num_indices: int, matlab_name: str, lib_name: str) -> dict:
    """Train one model and return its metrics (or the error that stopped it)."""
    row = {"num_indices": num_indices, "mf": matlab_name, "library_mf": lib_name}
    try:
        # ANFIS draws its random_state from `random`, so seed it for reproducibility
        random.seed(SEED)
        np.random.seed(SEED)

        model = ANFIS(**BASE_ARGS, membership_functions=lib_name, num_indices=num_indices)
        model.train()

        data = {
            "train": (model.X_train, model.Y_train),
            "val": (model.X_val, model.Y_val),
            "test": (model.X_test, model.Y_test),
        }
        for split, (X, y) in data.items():
            row[split] = compute_metrics(y, model.model.predict(X))
            row[f"n_{split}"] = int(len(y))

        row["n_rules"] = int(model.model.model_.n_rules)
        row["model_id"] = model.model_id
        row["experts"] = {
            "train": model.train_sheet_names,
            "val": model.val_sheet_names,
            "test": model.test_sheet_names,
        }
    except Exception as e:  # keep going so one bad MF doesn't kill the sweep
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def print_comparison(results: list[dict]) -> None:
    ok = {(r["num_indices"], r["mf"]): r for r in results if "error" not in r}

    print(f"\n{'=' * 86}\nTest-set comparison (RMSE on the normalised [0, 1] rating scale)\n{'=' * 86}")
    print(f"{'MF':<10}{'RMSE (3)':>11}{'R2 (3)':>9}{'RMSE (4)':>11}{'R2 (4)':>9}{'ΔRMSE 4−3':>12}  better")
    for mf in MF_TYPES:
        r3, r4 = ok.get((3, mf)), ok.get((4, mf))
        c3 = f"{r3['test']['rmse']:>11.5f}{r3['test']['r2']:>9.4f}" if r3 else f"{'FAILED':>20}"
        c4 = f"{r4['test']['rmse']:>11.5f}{r4['test']['r2']:>9.4f}" if r4 else f"{'FAILED':>20}"
        if r3 and r4:
            d = r4["test"]["rmse"] - r3["test"]["rmse"]
            tail = f"{d:>+12.5f}  {'4' if d < 0 else '3' if d > 0 else '='}"
        else:
            tail = ""
        print(f"{mf:<10}{c3}{c4}{tail}")

    for n in NUM_INDICES:
        rows = [r for (k, _), r in ok.items() if k == n]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["test"]["rmse"])
        rmses = [r["test"]["rmse"] for r in rows]
        print(
            f"\n{n} indicators: best MF = {best['mf']} (test RMSE {best['test']['rmse']:.5f}); "
            f"spread across MFs = {max(rmses) - min(rmses):.5f}"
        )

    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"FAILED n={r['num_indices']} {r['mf']}: {r['error']}")


def save_csv(results: list[dict], path: Path) -> None:
    fields = ["num_indices", "mf", "library_mf", "n_rules"]
    fields += [f"{s}_{m}" for s in SPLITS for m in ("rmse", "mse", "r2")]
    fields += ["n_train", "n_val", "n_test", "model_id", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            flat = {k: r.get(k) for k in fields if not any(k.startswith(f"{s}_") for s in SPLITS)}
            for s in SPLITS:
                for m in ("rmse", "mse", "r2"):
                    flat[f"{s}_{m}"] = r[s][m] if s in r else None
            w.writerow(flat)


def save_latex(results: list[dict], path: Path) -> None:
    ok = {(r["num_indices"], r["mf"]): r for r in results if "error" not in r}
    best = {
        n: min((r["test"]["rmse"] for (k, _), r in ok.items() if k == n), default=None)
        for n in NUM_INDICES
    }

    def cell(n: int, mf: str) -> str:
        r = ok.get((n, mf))
        if r is None:
            return "-- & --"
        rmse = f"{r['test']['rmse']:.4f}".replace(".", "{,}")
        if r["test"]["rmse"] == best[n]:
            rmse = rf"\textbf{{{rmse}}}"
        r2 = f"{r['test']['r2']:.3f}".replace(".", "{,}").replace("-", "$-$")
        return f"{rmse} & {r2}"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{3 indikatora} & \multicolumn{2}{c}{4 indikatora} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"  MF & RMSE & $R^2$ & RMSE & $R^2$ \\",
        r"  \midrule",
    ]
    for mf in MF_TYPES:
        lines.append(rf"  \code{{{mf}}} & {cell(3, mf)} & {cell(4, mf)} \\")
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\caption{Test RMSE i $R^2$ hibridnog ANFIS modela (3 epohe) po tipu membership "
        r"funkcije. Podebljana je najmanja greška za svaki broj indikatora.}",
        r"\label{tab:mf_rezultati}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> list[dict]:
    results = []
    for num_indices in NUM_INDICES:
        for matlab_name, lib_name in MF_TYPES.items():
            print(f"\n{'=' * 70}\n{num_indices} indicators | {matlab_name} -> {lib_name}\n{'=' * 70}")
            row = train_one(num_indices, matlab_name, lib_name)
            if "error" in row:
                print(f"  FAILED: {row['error']}")
            else:
                print(
                    "  " + "  ".join(
                        f"{s}: RMSE {row[s]['rmse']:.5f}, R2 {row[s]['r2']:.4f}" for s in SPLITS
                    )
                )
            results.append(row)

    print_comparison(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mf_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_csv(results, OUT_DIR / "mf_comparison.csv")
    save_latex(results, OUT_DIR / "mf_comparison.tex")
    print(f"\nSaved results to {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()