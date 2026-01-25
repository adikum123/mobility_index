from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Station:
    switch_id: int
    latitude: float
    longitude: float
    azimuth: int | None
    altitude: int
