"""Partial mobility index (pIM) for all station pairs and IM per interval.

pIM is computed for every ordered pair (i != j); IM is the mean pIM over the
pairs with at least one journey in the interval.

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
MODEL_3_PATH = PROJECT_ROOT / "models" / "3_gaussian.pkl"
MODEL_4_PATH = PROJECT_ROOT / "models" / "4_gaussian.pkl"
OUT_DIR = Path(__file__).parent / "results"


def main() -> None:
    model3 = ANFISRegressor.load(MODEL_3_PATH)
    model4 = ANFISRegressor.load(MODEL_4_PATH)
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

        active = pairs[pairs["active"]]  # pairs with >= 1 journey
        row = {"interval": INTERVAL_LABELS[t], "n_pairs": len(active)}
        for name in calculators:
            row[f"IM{name}"] = 100 * active[f"pIM{name}"].mean()
        rows.append(row)

    im = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.to_csv(OUT_DIR / "cdr_im_by_interval.csv", index=False)
    print(im.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
