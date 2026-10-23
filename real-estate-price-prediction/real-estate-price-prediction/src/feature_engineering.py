"""Deterministic feature construction + the (fit-on-train-only) preprocessing transformer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .geospatial import add_distance_features

NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "living_area", "lot_size", "property_age", "area_per_bedroom",
    "sale_year", "latitude", "longitude",
    "dist_downtown_km", "dist_park_km", "dist_school_km", "dist_transit_km",
]
CATEGORICAL_FEATURES = ["neighborhood", "property_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Used to present importances the way the dashboard shows them.
FEATURE_GROUPS = {
    "Living area": ["living_area", "area_per_bedroom"],
    "Location": ["neighborhood", "latitude", "longitude", "dist_downtown_km",
                 "dist_park_km", "dist_school_km", "dist_transit_km"],
    "Bedrooms": ["bedrooms"],
    "Bathrooms": ["bathrooms"],
    "Property age": ["property_age"],
    "Property type": ["property_type"],
    "Lot size": ["lot_size"],
    "Sale date": ["sale_year"],
}


def build_features(df: pd.DataFrame, amenities: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Stateless, so it is safe to run before splitting."""
    out = df.copy()
    out["sale_date"] = pd.to_datetime(out["sale_date"])
    out["sale_year"] = out["sale_date"].dt.year + (out["sale_date"].dt.dayofyear - 1) / 365.25
    out["property_age"] = (out["sale_date"].dt.year - out["year_built"]).clip(lower=0)
    out["area_per_bedroom"] = out["living_area"] / out["bedrooms"].replace(0, np.nan)
    return add_distance_features(out, amenities)


def make_preprocessor() -> ColumnTransformer:
    """Imputation, scaling and encoding. Fitted inside the pipeline -> only ever sees training folds."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
