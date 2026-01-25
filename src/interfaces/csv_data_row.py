from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(eq=True, frozen=True)
class CSVDataRow:
    user_id: int
    datetime: datetime
    latitude: float
    longitude: float
    switch_id: int
    azimuth: int | None
    altitude: int
    interval_num: int
