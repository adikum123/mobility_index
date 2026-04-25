from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict

from ..interfaces import CSVDataRow, Journey, Station, StationVisit
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

        # group by user, then derive visits and journeys in one pass
        self.by_user = self.group_by_user()
        self.visits, self.journeys = self.compute_visits_and_journeys()
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
            if azimuth == "INDOOR":
                return None
            return int(azimuth)
        except Exception:
            print(f"Could not process azimuth: {azimuth}")
            return None

    def group_by_user(self) -> DefaultDict[int, list[CSVDataRow]]:
        by_user = defaultdict(list)
        for item in self.processed_raw_data:
            by_user[item.user_id].append(item)
        return by_user

    def compute_visits_and_journeys(
        self,
    ) -> tuple[list[StationVisit], list[Journey]]:
        """Single pass per user: collapse into station visits, then derive
        journeys as transitions between consecutive visits.

        A journey's *start* is the last record of the departing visit and
        its *end* is the first record of the arriving visit.
        """
        all_visits: list[StationVisit] = []
        journeys: list[Journey] = []

        for _, records in self.by_user.items():
            sorted_records = sorted(set(records), key=lambda r: r.datetime)
            if not sorted_records:
                continue

            # --- build visits for this user ---
            user_visits: list[StationVisit] = []
            run: list[CSVDataRow] = [sorted_records[0]]
            for record in sorted_records[1:]:
                if record.switch_id == run[-1].switch_id:
                    run.append(record)
                else:
                    user_visits.append(StationVisit.from_records(run))
                    run = [record]
            user_visits.append(StationVisit.from_records(run))
            all_visits.extend(user_visits)

            # --- derive journeys between consecutive visits ---
            for i in range(len(user_visits) - 1):
                journey = Journey.from_records(
                    departure=user_visits[i].records[-1],
                    arrival=user_visits[i + 1].records[0],
                )
                if journey is not None:
                    journeys.append(journey)

        # filter journeys
        filter_statistics = defaultdict(int)
        filtered_journeys = [
            j
            for j in journeys
            if not j.remove_journey(filter_statistics=filter_statistics)
        ]
        num_filtered, num_total = len(filtered_journeys), len(journeys)
        print(
            f"Filtered: {num_filtered}/{num_total} ({100 * num_filtered / num_total:.2f}%) journeys with following filter statistics:\n{json.dumps(filter_statistics, indent=4)}"
        )
        return all_visits, filtered_journeys

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
        sorted_switch_ids = sorted(switch_ids)
        sorted_stations = sorted(stations, key=lambda s: s.switch_id)
        return sorted_stations, sorted_switch_ids

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
