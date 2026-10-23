"""Geospatial helpers: distances, GeoPandas conversion, neighbourhood summaries.

Distances use a vectorised haversine formula (pure NumPy), so GeoPandas is
optional. If GeoPandas is installed, `to_geodataframe` / `to_geojson` use it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DOWNTOWN = (47.6062, -122.3321)  # (lat, lon) of the demo city centre
EARTH_RADIUS_KM = 6371.0088
AMENITY_TYPES = ("downtown", "park", "school", "transit")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_distance_km(lat, lon, points: pd.DataFrame) -> np.ndarray:
    """Distance (km) from each (lat, lon) to the nearest row of `points`."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    best = np.full(lat.shape, np.inf)
    for p_lat, p_lon in zip(points["latitude"], points["longitude"]):
        best = np.minimum(best, haversine_km(lat, lon, p_lat, p_lon))
    return best


def add_distance_features(df: pd.DataFrame, amenities: pd.DataFrame) -> pd.DataFrame:
    """Adds dist_<type>_km columns for downtown, parks, schools and transit."""
    out = df.copy()
    for kind in AMENITY_TYPES:
        pts = amenities[amenities["amenity_type"] == kind]
        out[f"dist_{kind}_km"] = (
            nearest_distance_km(out["latitude"], out["longitude"], pts) if len(pts) else np.nan
        )
    return out


def to_geodataframe(df: pd.DataFrame):
    """Convert a listings frame to a GeoDataFrame (requires geopandas)."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )


def to_geojson(df: pd.DataFrame, path: str | Path, columns=("property_id", "neighborhood", "sale_price")) -> Path:
    """Write points to GeoJSON; uses GeoPandas when available, plain JSON otherwise."""
    path = Path(path)
    try:
        gdf = to_geodataframe(df)[list(columns) + ["geometry"]]
        gdf.to_file(path, driver="GeoJSON")
    except ImportError:
        feats = [
            {
                "type": "Feature",
                "properties": {c: (None if pd.isna(r[c]) else r[c]) for c in columns},
                "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
            }
            for _, r in df.iterrows()
        ]
        path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, default=str))
    return path


def neighborhood_price_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(price_per_sqft=df["sale_price"] / df["living_area"])
    return (
        d.groupby("neighborhood")
        .agg(
            sales=("sale_price", "size"),
            median_price=("sale_price", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .sort_values("median_price_per_sqft", ascending=False)
        .reset_index()
    )
