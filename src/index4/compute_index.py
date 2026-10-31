import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..interfaces import Station

BASE_PATH = Path(__file__).parents[2] / "data" / "index4"
OUTPUT_BASE = Path(__file__).parents[2] / "data" / "output"
INDEX4_PATH = OUTPUT_BASE / "index4" / "index4_array.npz"
INDEX4_MATRICES_DIR = OUTPUT_BASE / "index4_matrices"
INDEX4_DEST_MATRIX_PATH = INDEX4_MATRICES_DIR / "index4_dest_matrix.npz"
INDEX4_MEAN_MATRIX_PATH = INDEX4_MATRICES_DIR / "index4_mean_matrix.npz"

RADIUS_KM = 1.5
_EARTH_RADIUS_KM = 6371.0


def _load_config() -> dict:
    with open(BASE_PATH / "config.json", "r") as f:
        return json.load(f)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km); broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def station_accessibility_scores(
    stations: list[Station], radius_km: float = RADIUS_KM
) -> np.ndarray:
    """Index-4 (amenity accessibility) score per station, normalised to [0, 1].

    For every amenity subgroup, the station gets
        wG * n + beta * wH * min(n, c_cap)
    where n is the number of amenities within ``radius_km`` of the station's
    azimuth-adjusted position. Scores are summed over subgroups and min-max
    normalised. Distances are vectorised haversine (the 1.5 km cut-off makes
    the sphere/ellipsoid difference irrelevant).
    """
    config = _load_config()
    coords = np.array([s.adjusted_coordinates() for s in stations])  # (n, 2)
    scores = np.zeros(len(stations))

    for group, subgroups in config.items():
        for subgroup, subgroup_config in subgroups.items():
            w = subgroup_config["weights"]
            path = BASE_PATH / f"{group}__{subgroup}.csv"
            if not path.exists():
                print(f"Missing amenity file {path.name}; skipping")
                continue
            amen = pd.read_csv(path)[["lat", "lon"]].dropna().to_numpy()
            if len(amen) == 0:
                continue
            d = _haversine_km(
                coords[:, None, 0], coords[:, None, 1], amen[None, :, 0], amen[None, :, 1]
            )
            num = (d <= radius_km).sum(axis=1)
            scores += w["wG"] * num + w["beta"] * w["wH"] * np.minimum(num, w["c_cap"])

    lo, hi = scores.min(), scores.max()
    if hi != lo:
        scores = (scores - lo) / (hi - lo)
    return scores


def save_index4(scores: np.ndarray, switch_ids: list[int] | np.ndarray) -> None:
    """Save the per-station scores and the two station-pair matrices.

    Pair matrices use the same .npz format as the other indicator matrices
    (``values``, ``switch_ids``), with entry (i, j) for origin i, destination j:
      * index4_dest_matrix.npz: D[j]               (accessibility of the destination)
      * index4_mean_matrix.npz: (D[i] + D[j]) / 2  (mean of origin and destination)
    """
    scores = np.asarray(scores, dtype=np.float64)
    switch_ids = np.asarray(switch_ids, dtype=np.int64)

    os.makedirs(INDEX4_PATH.parent, exist_ok=True)
    np.savez_compressed(INDEX4_PATH, scores=scores, switch_ids=switch_ids)
    print(f"Saved index4 array to {INDEX4_PATH}")

    os.makedirs(INDEX4_MATRICES_DIR, exist_ok=True)
    n = len(scores)
    matrices = {
        INDEX4_DEST_MATRIX_PATH: np.broadcast_to(scores[None, :], (n, n)),
        INDEX4_MEAN_MATRIX_PATH: (scores[:, None] + scores[None, :]) / 2,
    }
    for path, values in matrices.items():
        values = np.array(values, dtype=np.float64)
        np.fill_diagonal(values, 0.0)  # no journeys within a station, as in the other matrices
        np.savez_compressed(path, values=values, switch_ids=switch_ids)
        print(f"Saved index4 matrix to {path}")


def compute_and_save_index4(azimuth_offset_km: float = 0.0):
    """Compute Index-4 for all CDR stations (ordered by switch_id) and save the
    per-station array plus the destination and mean pair matrices."""
    from ..engine.data_processor import DataProcessor

    print("Computing index...")
    data_processor = DataProcessor()
    scores = station_accessibility_scores(data_processor.stations)
    save_index4(scores, data_processor.switch_ids)


if __name__ == "__main__":
    compute_and_save_index4()
