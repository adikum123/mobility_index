import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import distance

from ..engine.matrices_processor import MatricesProcessor
from .fetch_data import AMENITIES_CONFIG, BASE_PATH

OUTPUT_BASE = Path(__file__).parents[2] / "data" / "output"


def compute_and_save_index4(azimuth_offset_km: float = 0.0):
    """Compute Index-4 (amenity accessibility) per-station array.

    Saves a 1-D array where position *i* holds the accessibility score
    for station *i* (ordered by switch_id).

    Parameters
    ----------
    azimuth_offset_km : float
        If > 0, each station's position is shifted along its azimuth
        bearing by this distance (km) before proximity filtering.
    """
    print("Computing index...")
    matrices_processor = MatricesProcessor()
    data_processor = matrices_processor.data_processor
    station_scores = defaultdict(float)

    for idx, switch_id in enumerate(data_processor.switch_ids):
        if idx % 100 == 0:
            print(f"Processed {idx} / {len(data_processor.switch_ids)} stations...")
        station = data_processor.get_station_by_switch_id(switch_id)

        for group, subgroups in AMENITIES_CONFIG.items():
            for subgroup, subgroup_config in subgroups.items():
                print(f"Processing group {group} and subgroup {subgroup}...")
                weights = subgroup_config["weights"]

                filename = f"{group}__{subgroup}.csv"
                filepath = BASE_PATH / filename
                subgroup_amenities = pd.read_csv(filepath).to_dict(orient="records")

                station_point = station.adjusted_coordinates()
                relevant_amenities = [
                    x
                    for x in subgroup_amenities
                    if distance(station_point, (x["lat"], x["lon"])).km <= 1.5
                ]
                num = len(relevant_amenities)
                station_scores[switch_id] += weights["wG"] * num + weights[
                    "beta"
                ] * weights["wH"] * min(num, weights["c_cap"])

    scores = np.array(
        [station_scores.get(sid, 0.0) for sid in data_processor.switch_ids]
    )

    # normalize to [0, 1]
    min_val, max_val = scores.min(), scores.max()
    if max_val != min_val:
        scores = (scores - min_val) / (max_val - min_val)

    save_dir = OUTPUT_BASE / "index4"
    os.makedirs(save_dir, exist_ok=True)
    filepath = save_dir / "index4_array.npz"
    switch_ids = np.asarray(data_processor.switch_ids, dtype=np.int64)
    np.savez_compressed(
        filepath,
        scores=np.asarray(scores, dtype=np.float64),
        switch_ids=switch_ids,
    )
    print(f"Saved index4 array to {filepath}")


if __name__ == "__main__":
    compute_and_save_index4()
