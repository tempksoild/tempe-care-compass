# geocoding.py — Geocoding service for Tempe Care Compass
# Converts address strings from Snowflake / Demo dataset into pinpoint coordinates.
# Supports Mapbox Geocoding API if MAPBOX_API_KEY / MAPBOX_TOKEN is provided.
# Falls back to OpenStreetMap Nominatim and cached Tempe ZIP centroids.

import os
import re
import json
import hashlib
import urllib.request
import urllib.parse
import streamlit as st

# Pre-indexed centroids for Tempe, AZ postal codes for instant fallback
TEMPE_ZIP_COORDINATES = {
    "85281": (33.4223, -111.9333),  # North Tempe / ASU / Town Lake
    "85282": (33.3932, -111.9287),  # Central Tempe / Broadway
    "85283": (33.3639, -111.9348),  # South-Central Tempe / Baseline
    "85284": (33.3444, -111.9219),  # South Tempe / Warner / Elliot
    "85285": (33.3850, -111.9400),  # Tempe West
    "85287": (33.4255, -111.9312),  # ASU Main Campus
}
DEFAULT_TEMPE_COORDINATES = (33.4255, -111.9400)


def _clean_street(addr: str) -> str:
    """Removes suite, unit, building, and apartment identifiers to optimize geocoding precision."""
    s = re.sub(r"(?i)\b(ste|suite|bldg|building|apt|apartment|fl|floor|unit|#)\s*[\w\d-]+", "", addr)
    s = re.sub(r",\s*\d+\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


@st.cache_data(show_spinner=False, ttl=86400)
def geocode_address(address: str, city: str = "Tempe", state: str = "AZ", zip_code: str = "") -> dict:
    """
    Converts a provider's street address into pinpoint latitude and longitude coordinates.
    Checks Mapbox Geocoding API first if credentials are configured, then OpenStreetMap Nominatim,
    and finally resolves via Tempe ZIP centroids so that coordinates are always available.
    """
    clean_st = _clean_street(address or "")
    city_val = (city or "Tempe").strip()
    state_val = (state or "AZ").strip()
    zip_val = (str(zip_code) or "")[:5].strip()

    full_query = f"{clean_st}, {city_val}, {state_val}"
    if zip_val:
        full_query += f" {zip_val}"

    # 1. Mapbox Geocoding API if key configured (env or streamlit secrets)
    mapbox_token = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")
    try:
        if not mapbox_token and hasattr(st, "secrets") and "MAPBOX_API_KEY" in st.secrets:
            mapbox_token = st.secrets["MAPBOX_API_KEY"]
    except Exception:
        pass

    if mapbox_token:
        try:
            encoded = urllib.parse.quote(full_query)
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded}.json?access_token={mapbox_token}&country=us&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "TempeCareCompass/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                features = data.get("features", [])
                if features and "geometry" in features[0]:
                    coords = features[0]["geometry"]["coordinates"]  # [lon, lat]
                    return {
                        "latitude": float(coords[1]),
                        "longitude": float(coords[0]),
                        "source": "Mapbox Geocoding",
                        "formatted_address": full_query,
                    }
        except Exception:
            pass

    # 2. OpenStreetMap Nominatim Geocoding
    try:
        encoded = urllib.parse.quote(full_query)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "TempeCareCompass/1.0 (healthcare-navigation-app)"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            if data and len(data) > 0:
                return {
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"]),
                    "source": "Nominatim Geocoder",
                    "formatted_address": full_query,
                }
    except Exception:
        pass

    # 3. Fallback to Tempe ZIP Centroid with deterministic micro-jitter
    base_coords = TEMPE_ZIP_COORDINATES.get(zip_val, DEFAULT_TEMPE_COORDINATES)
    addr_hash = int(hashlib.md5(full_query.encode()).hexdigest()[:6], 16)
    jitter_lat = ((addr_hash % 200) - 100) * 0.00008
    jitter_lon = (((addr_hash // 200) % 200) - 100) * 0.00008

    return {
        "latitude": round(base_coords[0] + jitter_lat, 6),
        "longitude": round(base_coords[1] + jitter_lon, 6),
        "source": "Tempe Postal Centroid",
        "formatted_address": full_query,
    }
