"""Synthetic housing-market generator (demo data only).

The generated market is FICTIONAL. It exists so the whole pipeline runs end to
end without internet access. Replace it with a real dataset (see README) before
drawing any conclusions about real prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geospatial import DOWNTOWN, add_distance_features

CITY = "Metro City"
# name, zip, latitude, longitude, price factor
NEIGHBORHOODS = [
    ("Harbor Point", "98101", 47.6150, -122.3400, 1.35),
    ("Old Town", "98104", 47.6000, -122.3290, 1.15),
    ("Midtown", "98122", 47.6120, -122.3050, 1.10),
    ("Lakeview", "98112", 47.6320, -122.2900, 1.45),
    ("Maple Heights", "98115", 47.6850, -122.3050, 1.05),
    ("Riverside", "98108", 47.5450, -122.3200, 0.80),
    ("Oak Park", "98118", 47.5500, -122.2750, 0.85),
    ("Sunset Hills", "98116", 47.5700, -122.3900, 1.00),
    ("Cedar Grove", "98133", 47.7300, -122.3450, 0.78),
    ("Westgate", "98199", 47.6500, -122.4000, 1.20),
    ("Pine Ridge", "98125", 47.7150, -122.2900, 0.88),
    ("Eastbrook", "98007", 47.6100, -122.1500, 1.25),
]


def make_amenities(seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = [("downtown", *DOWNTOWN)]
    for kind, k in (("park", 30), ("school", 24), ("transit", 18)):
        lat = rng.uniform(47.53, 47.74, k)
        lon = rng.uniform(-122.42, -122.14, k)
        rows += [(kind, a, b) for a, b in zip(lat, lon)]
    return pd.DataFrame(rows, columns=["amenity_type", "latitude", "longitude"])


def make_placeholder_amenities(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """For user data without an amenities file: PLACEHOLDER points inside the bounding box.
    Distances to these are NOT meaningful; supply a real data/raw/amenities.csv instead."""
    rng = np.random.default_rng(seed)
    lat_lo, lat_hi = df["latitude"].quantile([0.02, 0.98])
    lon_lo, lon_hi = df["longitude"].quantile([0.02, 0.98])
    rows = [("downtown", df["latitude"].median(), df["longitude"].median())]
    for kind, k in (("park", 30), ("school", 24), ("transit", 18)):
        rows += [(kind, a, b) for a, b in zip(rng.uniform(lat_lo, lat_hi, k), rng.uniform(lon_lo, lon_hi, k))]
    return pd.DataFrame(rows, columns=["amenity_type", "latitude", "longitude"])


def generate_listings(n: int = 8000, seed: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    amenities = make_amenities(seed)
    hoods = pd.DataFrame(NEIGHBORHOODS, columns=["neighborhood", "zip_code", "h_lat", "h_lon", "factor"])
    h = hoods.iloc[rng.integers(0, len(hoods), n)].reset_index(drop=True)

    lat = h["h_lat"].to_numpy() + rng.normal(0, 0.006, n)
    lon = h["h_lon"].to_numpy() + rng.normal(0, 0.008, n)
    ptype = rng.choice(config.PROPERTY_TYPES, n, p=[0.55, 0.20, 0.17, 0.08])
    ptype_s = pd.Series(ptype)

    mean = ptype_s.map({"Single Family": 2100, "Condo": 1000, "Townhouse": 1500, "Multi-Family": 2800}).to_numpy()
    sd = ptype_s.map({"Single Family": 700, "Condo": 330, "Townhouse": 420, "Multi-Family": 800}).to_numpy()
    area = np.clip(rng.normal(mean, sd), 450, 7000).round(-1).astype(int)
    bedrooms = np.clip(np.rint(area / 650 + rng.normal(0, 0.7, n)), 1, 8).astype(int)
    bathrooms = np.clip(np.round((0.55 * bedrooms + 0.4 + rng.normal(0, 0.5, n)) * 2) / 2, 1, 6)
    year_built = np.where(rng.random(n) < 0.25, rng.integers(2005, 2024, n), rng.integers(1920, 2005, n))
    lot = np.where(
        ptype == "Condo", 0,
        np.where(ptype == "Townhouse", rng.lognormal(7.6, 0.3, n), rng.lognormal(8.9, 0.45, n)),
    ).round().astype(int)
    sale_date = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 2191, n), unit="D")
    sale_year_frac = sale_date.year + (sale_date.dayofyear - 1) / 365.25

    geo = add_distance_features(pd.DataFrame({"latitude": lat, "longitude": lon}), amenities)
    type_factor = ptype_s.map({"Single Family": 1.0, "Condo": 1.12, "Townhouse": 1.03, "Multi-Family": 0.85}).to_numpy()
    loc_factor = 0.80 + 0.55 * np.exp(-geo["dist_downtown_km"].to_numpy() / 5.0)
    amen_factor = (
        1 - 0.035 * np.minimum(geo["dist_transit_km"], 4)
        - 0.020 * np.minimum(geo["dist_park_km"], 3)
        - 0.015 * np.minimum(geo["dist_school_km"], 3)
    ).to_numpy()
    age = np.clip(np.asarray(sale_date.year) - year_built, 0, None)
    age_factor = 1 - 0.0028 * np.minimum(age, 90) + np.where(age <= 5, 0.05, 0.0)
    time_factor = 1 + 0.055 * (np.asarray(sale_year_frac) - 2019)
    psf = 235 * h["factor"].to_numpy() * type_factor * loc_factor * amen_factor * age_factor * time_factor
    price = (area * psf + 9000 * bathrooms + 12 * lot) * rng.lognormal(0, 0.07, n)

    df = pd.DataFrame({
        "property_id": [f"P{100000 + i}" for i in range(n)],
        "city": CITY,
        "neighborhood": h["neighborhood"],
        "zip_code": h["zip_code"],
        "latitude": lat.round(6),
        "longitude": lon.round(6),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "living_area": area,
        "lot_size": lot.astype(float),
        "property_type": ptype,
        "year_built": year_built.astype(float),
        "sale_date": sale_date.strftime("%Y-%m-%d"),
        "sale_price": price.round(-2),
    })

    # ---- realistic dirt so the cleaning step has something to do ----
    def pick(frac):
        return rng.choice(n, int(n * frac), replace=False)

    df.loc[pick(0.020), "year_built"] = np.nan
    df.loc[pick(0.015), "bathrooms"] = np.nan
    df.loc[pick(0.010), "lot_size"] = np.nan
    df.loc[pick(0.005), "living_area"] = np.nan
    df.loc[pick(0.003), "sale_price"] = np.nan
    df.loc[pick(0.002), "bedrooms"] = 0
    df.loc[pick(0.002), "living_area"] = 99999
    df.loc[pick(0.002), "year_built"] = 1066
    out_idx = pick(0.004)
    df.loc[out_idx, "sale_price"] = df.loc[out_idx, "sale_price"] * rng.choice([8, 0.08], len(out_idx))
    dups = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dups]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, amenities


def main(n: int = 8000, seed: int = config.RANDOM_STATE) -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df, amenities = generate_listings(n, seed)
    df.to_csv(config.RAW_CSV, index=False)
    amenities.to_csv(config.AMENITIES_CSV, index=False)
    print(f"[generate] wrote {len(df):,} raw rows -> {config.RAW_CSV}")


if __name__ == "__main__":
    main()
