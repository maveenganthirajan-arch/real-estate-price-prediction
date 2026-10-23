"""Inference helpers used by the Streamlit app (and usable from scripts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .feature_engineering import FEATURES, build_features


def load_bundle():
    import joblib

    if not config.MODEL_PATH.exists():
        return None
    return joblib.load(config.MODEL_PATH)


def build_input_frame(bundle, neighborhood, bedrooms, bathrooms, living_area, property_type,
                      year_built, lot_size=None, sale_date=None) -> pd.DataFrame:
    hood = bundle["neighborhoods"].set_index("neighborhood").loc[neighborhood]
    if lot_size is None:
        lot_size = bundle["lot_defaults"].get(property_type, 0.0)
    row = pd.DataFrame([{
        "neighborhood": neighborhood, "city": hood["city"], "zip_code": hood["zip_code"],
        "latitude": hood["latitude"], "longitude": hood["longitude"],
        "bedrooms": bedrooms, "bathrooms": bathrooms, "living_area": living_area,
        "lot_size": lot_size, "property_type": property_type, "year_built": year_built,
        "sale_date": sale_date or bundle["reference_date"],
    }])
    return build_features(row, bundle["amenities"])[FEATURES]


def predict_price(bundle, neighborhood, bedrooms, bathrooms, living_area, property_type,
                  year_built, lot_size=None, sale_date=None) -> dict:
    X = build_input_frame(bundle, neighborhood, bedrooms, bathrooms, living_area, property_type,
                          year_built, lot_size, sale_date)
    price = float(bundle["model"].predict(X)[0])
    q = bundle["interval_log_halfwidth"]
    return {
        "price": price, "low": price * float(np.exp(-q)), "high": price * float(np.exp(q)),
        "level": bundle["interval_level"], "model": bundle["model_name"],
    }
