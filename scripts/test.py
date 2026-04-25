"""Per-journey geodesic distance (start station → end station) for all raw journeys."""

from pathlib import Path

import numpy as np
from geopy.distance import geodesic

from src.engine.data_processor import DataProcessor
from src.engine.plotter import plot_journey_inter_station_distances

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

    out = PLOTS / "journey_inter_station_distances_km.png"
    plot_journey_inter_station_distances(d, out, bin_km=0.5)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
