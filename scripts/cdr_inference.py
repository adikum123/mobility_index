"""Partial mobility index (pIM) for all station pairs and IM per interval.

pIM is computed for every ordered pair (i != j). Pairs with at least one journey
are split 80/20 in matrix order: IM is the mean pIM over the first 80%, IM_val over
the last 20% (validation set).

Run from the project root:
    python -m scripts.cdr_inference
"""

from pathlib import Path

import pandas as pd
from anfis_toolbox import ANFISRegressor

from src.engine.mobility_index import (
    INTERVAL_LABELS,
    NUM_INTERVALS,
    CDRIndicators,
    MobilityIndexCalculator,
)

PROJECT_ROOT = Path(__file__).parents[1]
OUT_DIR = Path(__file__).parent / "results"


def main(model_3_path: str, model_4_path: str, mf_type: str) -> None:
    model3 = ANFISRegressor.load(PROJECT_ROOT / "models" / model_3_path)
    model4 = ANFISRegressor.load(PROJECT_ROOT / "models" / model_4_path)
    calculators = {
        "3": MobilityIndexCalculator(model3, 3),
        "4_dest": MobilityIndexCalculator(model4, 4, index4_mode="dest"),
        "4_mean": MobilityIndexCalculator(model4, 4, index4_mode="mean"),
    }

    indicators = CDRIndicators()
    rows = []
    for t in range(NUM_INTERVALS):
        pairs = indicators.pairs(t)  # all pairs, with an `active` flag
        for name, calc in calculators.items():
            pairs[f"pIM{name}"] = calc.partial_indices(pairs)

        active = pairs[pairs["active"]]  # pairs with >= 1 journey, in matrix order
        n_main = round(0.8 * len(active))  # first 80% for IM, last 20% for validation
        main_set, val_set = active.iloc[:n_main], active.iloc[n_main:]
        row = {"interval": INTERVAL_LABELS[t], "n_pairs": len(main_set), "n_val": len(val_set)}
        for name in calculators:
            row[f"IM{name}"] = 100 * main_set[f"pIM{name}"].mean()
            row[f"IM{name}_val"] = 100 * val_set[f"pIM{name}"].mean()
            row[f"IM{name}_difference"] = f"{100 * (1 - (row[f"IM{name}"] / row[f"IM{name}_val"]))}%"
        rows.append(row)

    im = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.to_csv(OUT_DIR / f"{mf_type}_cdr_im_by_interval.csv", index=False)
    print(im.round(2).to_string(index=False))


if __name__ == "__main__":
    for mf_type in [
        "gaussian",
        "gaussian2",
        "bell",
        "triangular",
        "trapezoidal",
        "pi",
        "diffsigmoidal",
        "prodsigmoidal",
        "sigmoidal",
        "sshape",
        "zshape",
        "linsshape",
        "linzshape",
    ]:
        main(
            model_3_path=f"3_{mf_type}.pkl", 
            model_4_path=f"4_{mf_type}.pkl", 
            mf_type=mf_type
        )        
