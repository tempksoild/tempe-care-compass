# rag/geo.py — Location resolution for proximity scoring (task 9).
#
# NPPES (Affine DIM_PROVIDER_ADDRESS) has no latitude/longitude, so exact provider
# coordinates are only used if a record carries `latitude`/`longitude` (e.g. a future
# geocoded export). Otherwise ZIP centroids from the Census ZCTA Gazetteer are used.
# PO-box ZIPs without a ZCTA (85285, 85287, 85288) have no centroid -> same_city/unknown.

import csv
import math
from functools import lru_cache
from pathlib import Path

from rag.config import ZIP_CENTROIDS_CSV

ROOT = Path(__file__).parents[1]


@lru_cache(maxsize=1)
def zip_centroids() -> dict[str, tuple[float, float]]:
    path = ROOT / ZIP_CENTROIDS_CSV
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["zip"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)}


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def record_coordinates(record: dict) -> tuple[float, float] | None:
    try:
        lat, lon = float(record["latitude"]), float(record["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon) if not (math.isnan(lat) or math.isnan(lon)) else None


def zip_centroid(zip_code: str | None) -> tuple[float, float] | None:
    return zip_centroids().get(str(zip_code or "")[:5]) if zip_code else None


def user_location(zip_code: str | None, user_coords: tuple[float, float] | None):
    """-> (coords | None, method | None). method: 'user_coordinates' | 'zip_centroid' | 'zip_only'."""
    if user_coords:
        return user_coords, "user_coordinates"
    if zip_code:
        c = zip_centroid(zip_code)
        return (c, "zip_centroid") if c else (None, "zip_only")
    return None, None
