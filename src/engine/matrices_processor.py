import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from ..interfaces import Journey
from .data_processor import DataProcessor

OUTPUT_BASE = Path(__file__).parents[2] / "data" / "output"


class MatricesProcessor:
    def __init__(self) -> None:
        self.data_processor = DataProcessor()

    def save_matrix(
        self,
        matrix: np.ndarray,
        filename: str,
        sheet_name: str,
        output_subdir: str,
        azimuth_offset_km: float = 0.0,
    ) -> None:
        # save matrix
        matrix = MatricesProcessor.normalize_matrix(matrix=matrix)

        # create dir if not exists
        save_dir = OUTPUT_BASE / output_subdir
        os.makedirs(save_dir, exist_ok=True)

        # convert to dataframe and save to excel
        df = pd.DataFrame(
            matrix,
            index=self.data_processor.switch_ids,
            columns=self.data_processor.switch_ids,
        )
        df.index.name = ""
        filepath = save_dir / filename
        engine = "openpyxl" if str(filepath).endswith(".xlsx") else "xlsxwriter"
        with pd.ExcelWriter(filepath, engine=engine) as writer:
            df.to_excel(writer, sheet_name=sheet_name)
        print(f"Saved {sheet_name.lower()} to {filepath}")

        # set azimuth offset km
        self.azimuth_offset_km = azimuth_offset_km

    def compute_single_time_matrix_from_journeys(
        self, journeys: list[Journey]
    ) -> np.ndarray:
        n = len(self.data_processor.switch_ids)
        matrix = np.zeros((n, n))
        time_map = defaultdict(list)

        # get time map
        for j in journeys:
            start = j.start.switch_id
            end = j.end.switch_id
            time_map[(start, end)].append(j.datetime_diff_seconds)

        # fill the matrix
        for i, sid_i in enumerate(self.data_processor.switch_ids):
            for j, sid_j in enumerate(self.data_processor.switch_ids):
                if sid_i == sid_j:
                    continue
                durations = time_map.get((sid_i, sid_j), [])
                matrix[i][j] = np.mean(durations) if durations else 0

        return matrix

    def compute_time_matrices(self) -> None:
        for interval in range(8):
            journeys = self.data_processor.by_interval[interval]
            matrix = self.compute_single_time_matrix_from_journeys(journeys=journeys)
            filename = f"interval_{interval}_time_matrix.xlsx"
            self.save_matrix(
                matrix=matrix,
                filename=filename,
                sheet_name="IntervalMatrix",
                output_subdir="time_matrices",
            )

    def compute_distance_matrix(self) -> None:
        """Compute pairwise geodesic distance matrix.

        Parameters
        ----------
        azimuth_offset_km : float
            If > 0, each station's position is shifted along its azimuth
            bearing by this distance (km) before distances are computed.
            This yields a more realistic model when multiple sector
            antennas share the same tower coordinates.
        """
        n = len(self.data_processor.switch_ids)
        coords = {
            s.switch_id: s.adjusted_coordinates(offset_km=self.azimuth_offset_km)
            for s in self.data_processor.stations
        }

        matrix = np.zeros((n, n))
        for i, sid_i in enumerate(self.data_processor.switch_ids):
            for j, sid_j in enumerate(self.data_processor.switch_ids):
                if i == j:
                    continue
                matrix[i][j] = geodesic(coords[sid_i], coords[sid_j]).kilometers

        self.save_matrix(
            matrix=matrix,
            filename="distance_matrix.xlsx",
            sheet_name="DistanceMatrix",
            output_subdir="distance_matrices",
        )

    def compute_single_journey_count_matrix_from_journeys(
        self, journeys: list[Journey]
    ) -> np.ndarray:
        n = len(self.data_processor.switch_ids)
        matrix = np.zeros((n, n), dtype=int)
        count_map = defaultdict(int)

        for j in journeys:
            start = j.start.switch_id
            end = j.end.switch_id
            count_map[(start, end)] += 1

        for i, sid_i in enumerate(self.data_processor.switch_ids):
            for j, sid_j in enumerate(self.data_processor.switch_ids):
                if sid_i == sid_j:
                    continue
                matrix[i][j] = count_map.get((sid_i, sid_j), 0)

        return matrix

    def compute_journey_count_matrices(self) -> None:
        for interval in range(8):
            journeys = self.data_processor.by_interval[interval]
            matrix = self.compute_single_journey_count_matrix_from_journeys(
                journeys=journeys
            )
            filename = f"interval_{interval}_journey_counts_matrix.xlsx"
            self.save_matrix(
                matrix=matrix,
                filename=filename,
                sheet_name="JourneyCounts",
                output_subdir="journey_counts_matrices",
            )

    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize a matrix using (x - min) / (max - min)."""
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val == min_val:
            # Avoid division by zero; all values are equal
            return matrix
        return (matrix - min_val) / (max_val - min_val)
