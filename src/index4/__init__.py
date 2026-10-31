from .compute_index import (
    INDEX4_DEST_MATRIX_PATH,
    INDEX4_MEAN_MATRIX_PATH,
    INDEX4_PATH,
    compute_and_save_index4,
    save_index4,
    station_accessibility_scores,
)


def fetch_and_save_data(*args, **kwargs):
    """Lazy wrapper so importing src.index4 does not require osmnx."""
    from .fetch_data import fetch_and_save_data as _fetch

    return _fetch(*args, **kwargs)


__all__ = [
    "INDEX4_DEST_MATRIX_PATH",
    "INDEX4_MEAN_MATRIX_PATH",
    "INDEX4_PATH",
    "compute_and_save_index4",
    "fetch_and_save_data",
    "save_index4",
    "station_accessibility_scores",
]
