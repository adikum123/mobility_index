from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict

from geopy.distance import geodesic

from ..interfaces import CSVDataRow, Journey, Station
from ..utils import load_data

# Global path to the data file
DATA_FILE_PATH = Path(__file__).parents[2] / "data" / "input" / "data.csv"


class DataProcessor:

    def __init__(self, output_base: str = ".") -> None:
        # save output base to save computed matrices
        self.output_base = output_base

        # process list of dicts raw data into list of objects
        raw_data = load_data(file_path=DATA_FILE_PATH)
        self.processed_raw_data = self.process_raw_data(raw_data=raw_data)

        # from list of processed objects get list of journeys
        self.by_user = self.group_by_user()
        self.journeys = self.compute_journeys()
        self.by_interval = self.group_journeys_by_interval()

        # get stations and station ids
        self.stations, self.switch_ids = self.get_station_data()

    def process_raw_data(self, raw_data: list[dict[str, Any]]) -> list[CSVDataRow]:
        processed_raw_data = []
        for item in raw_data:
            item = {k.lower(): v for k, v in item.items()}
            try:
                datetime_obj = datetime.strptime(item["datetime"], "%d.%m.%Y %H:%M:%S")
                processed_raw_data.append(
                    CSVDataRow(
                        user_id=int(item["id_korisnika"]),
                        datetime=datetime_obj,
                        latitude=float(item["latitude"]),
                        longitude=float(item["longitude"]),
                        switch_id=int(item["switch_id"]),
                        azimuth=self.get_azimuth(item["azimuth"]),
                        altitude=int(item["altitude"]),
                        interval_num=datetime_obj.hour // 3,
                    )
                )
            except Exception as e:
                print(
                    f"Could not process row:\n{json.dumps(item, indent=4)}\nDue to: {str(e)}"
                )
                continue
        return processed_raw_data

    def get_azimuth(self, azimuth: str) -> int | None:
        try:
            if azimuth == "OMNI":
                return 360
            return int(azimuth)
        except Exception:
            print(f"Could not process azimuth: {azimuth}")
            return None

    def group_by_user(self) -> DefaultDict[int, list[CSVDataRow]]:
        by_user = defaultdict(list)
        for item in self.processed_raw_data:
            by_user[item.user_id].append(item)
        return by_user

    def compute_journeys(self) -> list[Journey]:
        journeys = []
        for _, values in self.by_user.items():
            sorted_values = list(set(sorted(values, key=lambda x: x.datetime)))
            for idx in range(0, len(sorted_values) - 1):
                # get start and end object
                start = sorted_values[idx]
                end = sorted_values[idx + 1]

                # get start and end point
                start_point = (start.latitude, start.longitude)
                end_point = (end.latitude, end.longitude)

                # get distance and duration
                datetime_diff_seconds = (end.datetime - start.datetime).total_seconds()

                # exclude joruneys with 0 seconds
                if datetime_diff_seconds == 0:
                    continue

                vincents_distance = geodesic(start_point, end_point).kilometers
                try:
                    journeys.append(
                        Journey(
                            start=start,
                            end=end,
                            datetime_diff_seconds=datetime_diff_seconds,
                            vincents_distance=vincents_distance,
                            interval_num=end.interval_num,
                            average_speed=vincents_distance
                            / (datetime_diff_seconds / 3600),
                        )
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to construct journey due to: {str(e)} for:\n{start}\n{end}"
                    )

        # filter journeys before returning
        filter_statistics = defaultdict(int)
        filtered_joruneys = [
            x
            for x in journeys
            if not x.remove_journey(filter_statistics=filter_statistics)
        ]
        num_filtered, num_total = len(filtered_joruneys), len(journeys)
        print(
            f"Filtered: {num_filtered}/{num_total} ({100 * num_filtered / num_total  :.2f} %) journeys with following filter statistics:\n{json.dumps(filter_statistics, indent=4)}"
        )
        return filtered_joruneys

    def get_station_data(self) -> tuple[list[Station], list[int]]:
        stations, switch_ids = [], set()
        for item in self.processed_raw_data:
            if item.switch_id not in switch_ids:
                switch_ids.add(item.switch_id)
                stations.append(
                    Station(
                        switch_id=item.switch_id,
                        latitude=item.latitude,
                        longitude=item.longitude,
                        azimuth=item.azimuth,
                        altitude=item.altitude,
                    )
                )

        # return sorted switch ids so that order is always the same
        return stations, sorted(switch_ids)

    def get_station_by_switch_id(self, switch_id: int) -> Station | None:
        for station in self.stations:
            if station.switch_id == switch_id:
                return station
        return None

    def group_journeys_by_interval(self) -> DefaultDict[int, list[Journey]]:
        by_interval = defaultdict(list)
        for journey in self.journeys:
            by_interval[journey.interval_num].append(journey)
        return by_interval
