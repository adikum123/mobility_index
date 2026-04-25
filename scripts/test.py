"""Per-journey geodesic distance (start station → end station) for all raw journeys."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic

from src.engine.data_processor import DataProcessor

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"


def main() -> None:
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
    bin_km = 0.5  # 500 m per bin
    d_max = float(d.max())
    right = np.ceil(d_max / bin_km) * bin_km + bin_km
    hist_edges = np.arange(0.0, right + 1e-9, bin_km)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="tight")
    axes[0].hist(
        d,
        bins=hist_edges,
        color="steelblue",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.3,
    )
    axes[0].set_xlabel(
        "Geodesic distance origin → destination station (km) (500 m bins)"
    )
    axes[0].set_ylabel("Journey count")
    axes[0].set_title(
        f"Journey inter-station distances (n={n}, bin width = {int(bin_km * 1000)} m)"
    )
    axes[0].grid(True, alpha=0.3)

    s = np.sort(d)
    axes[1].plot(
        s, (np.arange(1, n + 1) - 0.5) / n, color="darkgreen", drawstyle="steps-post"
    )
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    out = PLOTS / "journey_inter_station_distances_km.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
