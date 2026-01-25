from dataclasses import dataclass
from typing import DefaultDict

from .csv_data_row import CSVDataRow


@dataclass
class Journey:
    start: CSVDataRow
    end: CSVDataRow
    datetime_diff_seconds: int
    vincents_distance: float
    interval_num: int
    average_speed: float

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
