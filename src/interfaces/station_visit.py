from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .csv_data_row import CSVDataRow


@dataclass
class StationVisit:
    """A contiguous stay by one user at one base station.

    Built by collapsing consecutive raw CDR records that share the same
    user_id *and* switch_id into a single visit.
    """

    user_id: int
    switch_id: int
    latitude: float
    longitude: float
    azimuth: int | None
    altitude: int
    arrival: datetime
    departure: datetime
    dwell_seconds: float
    num_records: int
    records: list[CSVDataRow]

    @property
    def interval_num(self) -> int:
        return self.arrival.hour // 3

    @property
    def is_overnight(self) -> bool:
        """True when the visit spans at least part of 22:00-06:00."""
        return self.arrival.hour >= 22 or self.departure.hour < 6

    @staticmethod
    def from_records(records: list[CSVDataRow]) -> StationVisit:
        """Collapse a contiguous run of same-user, same-station records."""
        first, last = records[0], records[-1]
        return StationVisit(
            user_id=first.user_id,
            switch_id=first.switch_id,
            latitude=first.latitude,
            longitude=first.longitude,
            azimuth=first.azimuth,
            altitude=first.altitude,
            arrival=first.datetime,
            departure=last.datetime,
            dwell_seconds=(last.datetime - first.datetime).total_seconds(),
            num_records=len(records),
            records=records,
        )
