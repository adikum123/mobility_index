"""Diagnostics: journey distances, or per-expert target (Y) statistics for ANFIS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error, r2_score

from src.engine import ANFIS
from src.engine.data_processor import DataProcessor
from src.engine.plotter import plot_journey_inter_station_distances

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"

# Keep in sync with scripts/main.py for meaningful analysis.
ANFIS_ARGS = dict(
    num_indices=3,
    num_epochs=5,
    learning_rate=0.001,
    membership_functions="triangular",
    time_interval=3,
    loss_function="mse",
    batch_size=256,
    optimizer="adam",
    shuffle=True,
    n_mfs=3,
    num_train_experts=3,
    num_val_experts=2,
    num_test_experts=2,
)


def _hist_line(y: np.ndarray, n_bins: int = 5) -> str:
    """Y is in [0,1], bins align with (Ocjena-1)/5 steps of 0.2."""
    edges = np.linspace(0, 1, n_bins + 1)
    h, _ = np.histogram(y, bins=edges)
    parts = [f"[{edges[i]:.1f},{edges[i + 1]:.1f}):{h[i]}" for i in range(n_bins)]
    return "  " + "  ".join(parts)


def _print_block(name: str, y: np.ndarray) -> None:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        print(f"\n{name}: (empty)")
        return
    std = float(y.std())
    print(f"\n{name}  n={n:,}")
    print(f"  min={y.min():.6f}  max={y.max():.6f}  mean={y.mean():.6f}  std={std:.6f}")
    if std < 1e-5:
        print(
            "  NOTE: std ~ 0  →  R² = 1 - SSE/SST is unstable: SST in the denominator is ~0, "
            "so even small SSE gives huge negative R²."
        )
    print("  bin counts (width 0.2, [0,1)):")
    print(_hist_line(y))
    z = (y == 0.0).mean()
    print(f"  share Y==0 (Ocjena 1): {z:.1%}")


def analyze_expert_y_distribution() -> None:
    """Load matrices + mapper, build Y the same way as ANFIS, per-expert slices."""
    m = ANFIS(**ANFIS_ARGS)
    m._set_data_expert_split()

    nt, nv, nte = m.num_train_experts, m.num_val_experts, m.num_test_experts
    n_base = m.Y_train.size // nt
    if (
        n_base * nt != m.Y_train.size
        or n_base * nv != m.Y_val.size
        or n_base * nte != m.Y_test.size
    ):
        raise RuntimeError("unexpected Y lengths vs expert counts")

    mp, _ = m._mapper_path_and_key_cols()
    sheets = pd.read_excel(mp, sheet_name=None)
    n_total = nt + nv + nte
    names = list(sheets.keys())[:n_total]
    tr_nm = names[:nt]
    va_nm = names[nt : nt + nv]
    te_nm = names[nt + nv : n_total]

    print(
        "=== Per-expert Y (same X tiled; Y differs only by expert lookup / sheet) ===\n"
    )
    print(f"OD rows per copy (X_base): {n_base:,}")
    for i, en in enumerate(tr_nm):
        sl = slice(i * n_base, (i + 1) * n_base)
        _print_block(f"TRAIN  [{i}] {en}", m.Y_train[sl])
    for i, en in enumerate(va_nm):
        sl = slice(i * n_base, (i + 1) * n_base)
        _print_block(f"VAL    [{i}] {en}", m.Y_val[sl])
    for i, en in enumerate(te_nm):
        sl = slice(i * n_base, (i + 1) * n_base)
        _print_block(f"TEST   [{i}] {en}", m.Y_test[sl])

    yt, yv, yte = m.Y_train, m.Y_val, m.Y_test
    print("\n--- Overall split (concatenated per split) ---")
    _print_block("Y_train (all train experts)", yt)
    _print_block("Y_val   (all val experts)", yv)
    _print_block("Y_test  (all test experts)", yte)

    # R² / shift diagnostics (no model needed)
    sst = float(np.sum((yte - yte.mean()) ** 2))
    print("\n--- Why test R² can look catastrophic even when RMSE is modest ---")
    print(
        f"SST on Y_test  = sum((y - mean(y))²) = {sst:.4f}  "
        f"(if this is tiny, 1 - SSE/SST explodes in magnitude for moderate SSE)\n"
    )
    for label, ypred in [
        ("predict train-split mean for every test row", np.full_like(yte, yt.mean())),
        ("predict val-split mean for every test row", np.full_like(yte, yv.mean())),
    ]:
        r2 = r2_score(yte, ypred)
        rmse = float(np.sqrt(mean_squared_error(yte, ypred)))
        print(f"  {label:45s}  R²={r2:12.4f}  RMSE={rmse:.6f}")
    mean_vec = np.full_like(yte, yte.mean())
    r2_mean_pred = r2_score(yte, mean_vec)
    rmse_mean_pred = float(np.sqrt(mean_squared_error(yte, mean_vec)))
    print(
        f"  {'predict y_test.mean() for every row (MSE baseline)':45s}  R²={r2_mean_pred:12.4f}  RMSE={rmse_mean_pred:.6f}"
    )
    print(
        "  (sklearn R² is 0 by definition for that constant; it is not 1. Compare other rows to see shift.)\n"
    )
    print(
        "If train/val mean >> test mean, a model fit on train will be biased high on test; "
        "if Y_test also has tiny variance, SST is small and R² = 1 - SSE/SST can be hugely negative "
        "even when RMSE looks small on an absolute [0,1] scale."
    )


def main_journey() -> None:
    dp = DataProcessor()
    coords = {s.switch_id: s.adjusted_coordinates() for s in dp.stations}

    journeys = dp.journeys_all
    d = np.array(
        [
            geodesic(coords[j.start.switch_id], coords[j.end.switch_id]).kilometers
            for j in journeys
        ],
        dtype=np.float64,
    )

    n = len(d)
    print(f"Journeys (before remove_journey filter): {n}")
    if n == 0:
        return

    print(f"  min (km):   {d.min():.4f}")
    print(f"  max (km):   {d.max():.4f}")
    print(f"  mean (km):  {d.mean():.4f}")
    print(f"  std (km):   {d.std():.4f}")
    print(f"  median:     {float(np.median(d)):.4f}")
    for p in (5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p}:        {float(np.percentile(d, p)):.4f}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "journey_inter_station_distances_km.png"
    plot_journey_inter_station_distances(d, out, bin_km=0.5)
    print(f"\nSaved {out}")


def main() -> None:
    analyze_expert_y_distribution()


if __name__ == "__main__":
    main()
