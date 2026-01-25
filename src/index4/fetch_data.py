"""
Fetch amenities for Podgorica (Montenegro) from OpenStreetMap
Grouped by Indicator D groups and subgroups.

Output:
    One CSV file per subgroup with coordinates and metadata.

Requirements:
    pip install osmnx geopandas pandas shapely
"""

from __future__ import annotations

import json
from pathlib import Path

import osmnx as ox
import pandas as pd

ox.settings.use_cache = True
ox.settings.log_console = False

PLACE_NAME = "Podgorica, Montenegro"
BASE_PATH = Path(__file__).parents[2] / "data" / "index4"


with open(BASE_PATH / "config.json", "r") as f:
    AMENITIES_CONFIG = json.load(f)


def filter_by_tags(
    data: list[dict[str, str | float | None]], tags: dict[str, list[str]]
) -> list[dict[str, str | float | None]]:
    """
    Filtrira listu dictova prema vrijednostima tagova iz konfiguracije.

    Args:
        data (list of dict): lista objekata sa OSM podacima
        tags (dict[str, list[str]]): dozvoljeni tagovi iz configa

    Returns:
        list of dict: filtrirana lista objekata
    """
    if not data:
        return []

    filtered = []
    for row in data:
        keep = False
        for key, allowed_values in tags.items():
            if key in row and row[key] in allowed_values:
                keep = True
                break  # dovoljno da jedan tag odgovara
        if keep:
            filtered.append(row)

    return filtered


def apply_filters(
    data: list[dict[str, str | float | None]],
    tags: dict[str, list[str]],
    filters: dict[str, list[str]] = None,
) -> list[dict[str, str | float | None]]:
    """
    Primjenjuje sve filtere na listu dictova:
      - filter po tags
      - filter po imenu (exclude_names)

    Args:
        data (list of dict): lista objekata sa OSM podacima
        tags (dict[str, list[str]]): dozvoljeni tagovi iz configa
        filters (dict, optional): dodatni filteri, npr. {"exclude_names": ["Dom zdravlja"]}

    Returns:
        list of dict: filtrirana lista objekata
    """
    # 1. filter po tags
    data = filter_by_tags(data, tags)

    # 2. filter po imenu
    exclude_names = filters.get("exclude_names", []) if filters else []
    result = []
    for row in data:
        name = row.get("name", "")
        if not name:
            continue
        if not any(substr.lower() in name.lower() for substr in exclude_names):
            result.append(row)

    return result


def fetch_and_save_data():
    for group, subgroups in AMENITIES_CONFIG.items():
        for subgroup, tags in subgroups.items():
            request_tags = {
                k: v for k, v in tags.items() if k not in {"weights", "filters"}
            }
            print(f"Fetching: {group} → {subgroup}")
            gdf = ox.features_from_place(PLACE_NAME, tags=request_tags)

            if gdf.empty:
                print("  No data found.")
                continue

            # Convert geometry to centroid coordinates
            # First ensure we have a projected CRS for accurate centroid calculation
            if gdf.crs != "EPSG:32634":  # UTM zone 34N for Montenegro
                gdf_projected = gdf.to_crs(epsg=32634)  # UTM zone 34N for Montenegro
            else:
                gdf_projected = gdf

            # Calculate centroids in projected CRS
            centroids = gdf_projected.geometry.centroid

            # Convert centroids back to geographic coordinates
            centroids_geo = centroids.to_crs(epsg=4326)
            gdf["lat"] = centroids_geo.y
            gdf["lon"] = centroids_geo.x

            # Keep non-geometry attributes
            df = gdf.drop(columns="geometry", errors="ignore")

            # Add metadata
            df["group"] = group
            df["subgroup"] = subgroup

            # desired columns
            cols = [
                "group",
                "subgroup",
                "amenity",
                "highway",
                "school",
                "building",
                "healthcare",
                "social_facility",
                "leisure",
                "shop",
                "name",
                "name:cnr",
                "name:cnr-Latn",
                "addr:city",
                "addr:housenumber",
                "addr:street",
                "name:sr",
                "name:sr-Latn",
                "addr:neighbourhood",
                "lat",
                "lon",
            ]

            # keep only the desired columns and in the desired order
            df = df[[c for c in cols if c in df.columns]]

            # replace NaN with None
            df = df.where(pd.notna(df), None)

            # filter response and cast to list of dicts
            data = df.to_dict(orient="records")
            data = apply_filters(data=data, tags=tags, filters=tags.get("filters", {}))

            # Save as CSV and cast back to pandas DataFrame
            filename = f"{group}__{subgroup}.csv"
            filepath = BASE_PATH / filename
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            print(f"  Saved {len(df)} records → {filepath}")


if __name__ == "__main__":
    fetch_and_save_data()
