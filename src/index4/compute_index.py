from collections import defaultdict

import numpy as np
import pandas as pd
from geopy.distance import distance

from ..engine.matrices_processor import MatricesProcessor
from .fetch_data import AMENITIES_CONFIG, BASE_PATH


def compute_and_save_index4(azimuth_offset_km: float = 0.0):
    """Compute Index-4 (amenity accessibility) diagonal matrix.

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

                station_point = station.adjusted_coordinates(azimuth_offset_km)
                relevant_amenities = [
                    x
                    for x in subgroup_amenities
                    if distance(station_point, (x["lat"], x["lon"])).km <= 1.5
                ]
                num = len(relevant_amenities)
                station_scores[switch_id] += weights["wG"] * num + weights[
                    "beta"
                ] * weights["wH"] * min(num, weights["c_cap"])

    # Create diagonal matrix from scores
    n = len(data_processor.switch_ids)
    matrix = np.zeros((n, n))
    for i, switch_id in enumerate(data_processor.switch_ids):
        matrix[i, i] = station_scores.get(switch_id, 0.0)

    # Save using MatricesProcessor
    matrices_processor.save_matrix(
        matrix=matrix,
        filename="index4_matrix.xlsx",
        sheet_name="Index4",
        output_subdir="index4",
    )


if __name__ == "__main__":
    compute_and_save_index4()
