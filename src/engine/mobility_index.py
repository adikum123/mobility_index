"""Partial (pIM) and urban (IM) mobility index from precomputed matrices (PZ steps 7-9).

Everything is precomputed and only loaded here:
  * A  journey counts per interval   data/output/journey_counts_matrices/
  * B  mean travel time per interval data/output/time_matrices/
  * C  distance                      data/output/distance_matrices/
  * D  Index-4 accessibility         data/output/index4_matrices/
         index4_dest_matrix.npz  D(i, j) = accessibility of destination j
         index4_mean_matrix.npz  D(i, j) = mean of origin i and destination j

For every interval t:
    pIM(i, j)   = ANFIS(A, B, C[, D])                 for ALL ordered pairs i != j
    IM_t        = mean of pIM over active pairs [%]   (active = >= 1 journey, A > 0)
    alpha(i, j) = A*C / sum(A*C)                      (0 for pairs without journeys)
    Uk_mob_t    = sum(A*C)

For pairs without journeys the time matrix holds 0 (no trip), so their pIM is the
model output for "0 trips, 0 travel time". It is kept for completeness but must
not enter IM, which is why IM filters on the ``active`` flag.

A and C are min-max normalised with minimum 0, so alpha equals the raw-value
alpha and Uk_mob is in normalised units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .matrices_processor import (
    DISTANCE_MATRIX_FILENAME,
    OUTPUT_BASE,
    MatricesProcessor,
    interval_journey_counts_matrix_filename,
    interval_time_matrix_filename,
)

NUM_INTERVALS = 8
INTERVAL_LABELS = [f"{3 * t:02d}-{3 * (t + 1):02d}h" for t in range(NUM_INTERVALS)]
INDEX4_MATRIX_FILENAMES = {
    "dest": "index4_dest_matrix.npz",
    "mean": "index4_mean_matrix.npz",
}


class CDRIndicators:
    """Loads all precomputed indicator matrices (same station order everywhere)."""

    def __init__(self, output_base: Path = OUTPUT_BASE):
        self.C, self.switch_ids = self._load(output_base / "distance_matrices" / DISTANCE_MATRIX_FILENAME)
        self.A = [
            self._load(output_base / "journey_counts_matrices" / interval_journey_counts_matrix_filename(t))[0]
            for t in range(NUM_INTERVALS)
        ]
        self.B = [
            self._load(output_base / "time_matrices" / interval_time_matrix_filename(t))[0]
            for t in range(NUM_INTERVALS)
        ]
        self.D = {
            mode: self._load(output_base / "index4_matrices" / filename)[0]
            for mode, filename in INDEX4_MATRIX_FILENAMES.items()
        }

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        values, switch_ids = MatricesProcessor.load_matrix_npz(path)
        if hasattr(self, "switch_ids") and not np.array_equal(switch_ids, self.switch_ids):
            raise ValueError(f"{path.name} uses a different station order")
        return values, switch_ids

    def pairs(self, t: int) -> pd.DataFrame:
        """All ordered pairs (i, j), i != j, in interval t with all indicator values.

        ``active`` marks pairs with at least one journey in t (A > 0).
        """
        n = len(self.switch_ids)
        i, j = np.nonzero(~np.eye(n, dtype=bool))
        a = self.A[t][i, j]
        df = pd.DataFrame(
            {
                "interval": t,
                "origin": self.switch_ids[i],
                "dest": self.switch_ids[j],
                "active": a > 0,
                "A": a,
                "B": self.B[t][i, j],
                "C": self.C[i, j],
            }
        )
        for mode, d in self.D.items():
            df[f"D_{mode}"] = d[i, j]
        ac = df["A"] * df["C"]
        df["alpha"] = ac / ac.sum() if ac.sum() > 0 else 0.0
        return df


class MobilityIndexCalculator:
    """Partial mobility index for one trained model."""

    def __init__(self, model, num_indices: int, index4_mode: str = "dest"):
        assert num_indices in (3, 4)
        assert index4_mode in INDEX4_MATRIX_FILENAMES
        self.model = model  # fitted anfis_toolbox ANFISRegressor
        self.num_indices = num_indices
        self.index4_mode = index4_mode

    def partial_indices(self, pairs: pd.DataFrame, batch_size: int = 200_000) -> np.ndarray:
        """pIM for every row of ``pairs``, predicted in batches to bound memory."""
        cols = ["A", "B", "C"] + ([f"D_{self.index4_mode}"] if self.num_indices == 4 else [])
        X = pairs[cols].to_numpy(dtype=np.float64)
        out = np.empty(len(X))
        for start in range(0, len(X), batch_size):
            out[start : start + batch_size] = self.model.predict(X[start : start + batch_size])
        return out
