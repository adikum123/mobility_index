from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Station:
    switch_id: int
    latitude: float
    longitude: float
    azimuth: int | None
    altitude: int

    _EARTH_RADIUS_KM: float = field(default=6371.0, init=False, repr=False)

    def adjusted_coordinates(self) -> tuple[float, float]:
        """Shift position along azimuth bearing by *offset_km*.

        For OMNI (360) or unknown azimuth, the original position is returned
        because there is no preferred direction.
        """
        if self.azimuth is None or self.azimuth == 360:
            return self.latitude, self.longitude
        offset_km = 0.5  # 500 m
        bearing_rad = math.radians(self.azimuth)
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)
        angular_dist = offset_km / self._EARTH_RADIUS_KM

        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist)
            + math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
        )
        new_lon = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat),
        )

        return math.degrees(new_lat), math.degrees(new_lon)
