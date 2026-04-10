from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict

from geopy.distance import geodesic

from .csv_data_row import CSVDataRow


@dataclass
class Journey:
    start: CSVDataRow
    end: CSVDataRow
    datetime_diff_seconds: float
    vincents_distance: float
    interval_num: int
    average_speed: float

    @staticmethod
    def from_records(departure: CSVDataRow, arrival: CSVDataRow) -> Journey | None:
        """Build a journey from a departure and arrival record.

        Returns None when the time delta is zero (no real transition).
        """
        dt_seconds = (arrival.datetime - departure.datetime).total_seconds()
        if dt_seconds == 0:
            return None

        start_point = (departure.latitude, departure.longitude)
        end_point = (arrival.latitude, arrival.longitude)
        dist_km = geodesic(start_point, end_point).kilometers

        return Journey(
            start=departure,
            end=arrival,
            datetime_diff_seconds=dt_seconds,
            vincents_distance=dist_km,
            interval_num=arrival.interval_num,
            average_speed=dist_km / (dt_seconds / 3600),
        )

    def remove_journey(self, filter_statistics: DefaultDict[str, int]) -> bool:
        """
        Filtering method which returns true if journey should be removed
        """
        if self.datetime_diff_seconds < 60 * 10 or self.datetime_diff_seconds > 60 * 60:
            filter_statistics["time"] += 1
            return True
        if self.vincents_distance < 1.0:
            filter_statistics["distance"] += 1
            return True
        if self.average_speed > 100:
            filter_statistics["speed"] += 1
            return True
        return False
