import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from geopy.distance import geodesic

from ..interfaces import Journey
from .data_processor import DataProcessor

OUTPUT_BASE = Path(__file__).parents[2] / "data" / "output"

DISTANCE_MATRIX_FILENAME = "distance_matrix.npz"


def interval_time_matrix_filename(interval: int) -> str:
    return f"interval_{interval}_time_matrix.npz"


def interval_journey_counts_matrix_filename(interval: int) -> str:
    return f"interval_{interval}_journey_counts_matrix.npz"


class MatricesProcessor:
    def __init__(self) -> None:
        self.data_processor = DataProcessor()

    @staticmethod
    def load_matrix_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a square matrix and its switch_id axis order from ``.npz``."""
        with np.load(path) as data:
            values = np.asarray(data["values"], dtype=np.float64)
            switch_ids = np.asarray(data["switch_ids"], dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(
                f"Expected square matrix at {path}, got shape {values.shape}"
            )
        if values.shape[0] != len(switch_ids):
            raise ValueError(
                f"values shape {values.shape} does not match len(switch_ids)={len(switch_ids)}"
            )
        return values, switch_ids

    def _save_matrix_to_npz(
        self,
        matrix: np.ndarray,
        filename: str,
        output_subdir: str,
    ) -> None:
        save_dir = OUTPUT_BASE / output_subdir
        os.makedirs(save_dir, exist_ok=True)
        filepath = save_dir / filename
        switch_ids = np.asarray(self.data_processor.switch_ids, dtype=np.int64)
        if matrix.shape != (len(switch_ids), len(switch_ids)):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match n={len(switch_ids)}"
            )
        np.savez_compressed(
            filepath,
            values=np.asarray(matrix, dtype=np.float64),
            switch_ids=switch_ids,
        )
        print(f"Saved matrix to {filepath}")

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
        matrices = []
        for interval in range(8):
            journeys = self.data_processor.by_interval[interval]
            matrices.append(self.compute_single_time_matrix_from_journeys(journeys))

        global_min, global_max = self._global_min_max(matrices)
        for interval, matrix in enumerate(matrices):
            matrix = MatricesProcessor.normalize_matrix(matrix, global_min, global_max)
            self._save_matrix_to_npz(
                matrix=matrix,
                filename=interval_time_matrix_filename(interval),
                output_subdir="time_matrices",
            )

    def compute_distance_matrix(self, azimuth_offset_km: float = 0.0) -> None:
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
            s.switch_id: s.adjusted_coordinates(azimuth_offset_km)
            for s in self.data_processor.stations
        }

        matrix = np.zeros((n, n))
        for i, sid_i in enumerate(self.data_processor.switch_ids):
            for j, sid_j in enumerate(self.data_processor.switch_ids):
                if i == j:
                    continue
                matrix[i][j] = geodesic(coords[sid_i], coords[sid_j]).kilometers

        matrix = MatricesProcessor.normalize_matrix(matrix)

        self._save_matrix_to_npz(
            matrix=matrix,
            filename=DISTANCE_MATRIX_FILENAME,
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
        matrices = []
        for interval in range(8):
            journeys = self.data_processor.by_interval[interval]
            matrices.append(
                self.compute_single_journey_count_matrix_from_journeys(journeys)
            )

        global_min, global_max = self._global_min_max(matrices)
        for interval, matrix in enumerate(matrices):
            matrix = MatricesProcessor.normalize_matrix(matrix, global_min, global_max)
            self._save_matrix_to_npz(
                matrix=matrix,
                filename=interval_journey_counts_matrix_filename(interval),
                output_subdir="journey_counts_matrices",
            )

    @staticmethod
    def _global_min_max(matrices: list[np.ndarray]) -> tuple[float, float]:
        """Compute the global min and max across a list of matrices."""
        global_min = min(m.min() for m in matrices)
        global_max = max(m.max() for m in matrices)
        return global_min, global_max

    @staticmethod
    def normalize_matrix(
        matrix: np.ndarray,
        min_val: float = None,
        max_val: float = None,
    ) -> np.ndarray:
        """Normalize a matrix using (x - min) / (max - min).

        If min_val/max_val are not provided, uses the matrix's own bounds.
        """
        if min_val is None:
            min_val = matrix.min()
        if max_val is None:
            max_val = matrix.max()
        if max_val == min_val:
            return matrix
        return (matrix - min_val) / (max_val - min_val)
