"""Rule-based cleaning: duplicates, impossible values, missing critical fields, outliers.

Only deterministic, label-free rules (plus a price-per-sqft outlier filter) are
applied here. Statistical imputation of the remaining gaps happens INSIDE the
modelling pipeline so that it is fitted on training data only (no leakage).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

NUMERIC_COLS = ["latitude", "longitude", "bedrooms", "bathrooms", "living_area", "lot_size", "year_built", "sale_price"]
CRITICAL = ["sale_price", "living_area", "latitude", "longitude", "neighborhood", "property_type", "sale_date"]


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()
    current_year = pd.Timestamp.today().year

    for c in ("neighborhood", "city", "property_type"):
        df[c] = df[c].astype("object").where(df[c].notna(), None)
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    df["property_type"] = df["property_type"].map(lambda v: v.title() if isinstance(v, str) else v)
    df["zip_code"] = df["zip_code"].astype("object").map(lambda v: str(v).split(".")[0].zfill(5) if pd.notna(v) else v)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

    # 1) duplicates
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=["property_id", "sale_date"])
    report["duplicates_removed"] = int(n0 - len(df))

    # 2) impossible values -> NaN (kept as gaps, not guessed)
    invalid = {
        "bedrooms": ~df["bedrooms"].between(1, 10) & df["bedrooms"].notna(),
        "bathrooms": ~df["bathrooms"].between(0.5, 10) & df["bathrooms"].notna(),
        "living_area": ~df["living_area"].between(300, 12000) & df["living_area"].notna(),
        "year_built": ~df["year_built"].between(1850, current_year) & df["year_built"].notna(),
        "lot_size": (df["lot_size"] < 0),
    }
    report["invalid_values_nulled"] = {k: int(m.sum()) for k, m in invalid.items()}
    for col, mask in invalid.items():
        df.loc[mask, col] = np.nan

    # 3) rows missing fields we cannot model without
    n0 = len(df)
    df = df.dropna(subset=CRITICAL)
    df = df[df["sale_price"] >= 10_000]
    report["dropped_missing_critical"] = int(n0 - len(df))

    # 4) price-per-sqft outliers (3 x IQR on log scale, per property type)
    n0 = len(df)
    lp = np.log(df["sale_price"] / df["living_area"])
    g = lp.groupby(df["property_type"])
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df = df[(lp >= q1 - 3 * iqr) & (lp <= q3 + 3 * iqr)]
    report["price_outliers_removed"] = int(n0 - len(df))

    df = df.sort_values("sale_date").reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["missing_after_cleaning"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    return df, report


def save(clean: pd.DataFrame, report: dict) -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_CSV, index=False)
    config.CLEANING_REPORT.write_text(json.dumps(report, indent=2))
