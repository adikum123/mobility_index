import csv
from typing import Any


def load_data(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
        raw_data = list(csv.DictReader(csvfile))
    return raw_data
