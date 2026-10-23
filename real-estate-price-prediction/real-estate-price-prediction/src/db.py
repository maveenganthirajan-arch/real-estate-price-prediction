"""SQL layer. Uses SQLite (stdlib) by default; the schema is PostgreSQL-compatible."""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import config


def _read_sql_file(name: str) -> str:
    return (config.SQL_DIR / name).read_text()


def connect(path=None) -> sqlite3.Connection:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path or config.DB_PATH)


def write_clean(clean: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """(Re)create the schema and load cleaned listings into normalised tables."""
    conn.executescript(_read_sql_file("schema.sql"))

    hoods = (
        clean.groupby("neighborhood")
        .agg(city=("city", "first"), zip_code=("zip_code", "first"),
             latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
        .rename(columns={"neighborhood": "name"})
    )
    hoods.insert(0, "neighborhood_id", range(1, len(hoods) + 1))
    hoods.to_sql("neighborhoods", conn, if_exists="append", index=False)

    d = clean.merge(hoods[["neighborhood_id", "name"]], left_on="neighborhood", right_on="name")
    props = d.drop_duplicates("property_id")[
        ["property_id", "neighborhood_id", "latitude", "longitude", "bedrooms", "bathrooms",
         "living_area", "lot_size", "property_type", "year_built"]
    ]
    props.to_sql("properties", conn, if_exists="append", index=False)

    sales = d[["property_id", "sale_date", "sale_price"]].copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"]).dt.strftime("%Y-%m-%d")
    sales.insert(0, "sale_id", range(1, len(sales) + 1))
    sales.to_sql("sales", conn, if_exists="append", index=False)
    conn.commit()


def load_training_frame(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect()
    try:
        df = pd.read_sql_query(_read_sql_file("training_query.sql"), conn, parse_dates=["sale_date"])
    finally:
        if own:
            conn.close()
    return df


def neighborhood_summary() -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql_query(_read_sql_file("neighborhood_summary.sql"), conn)
    finally:
        conn.close()


def monthly_trend() -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql_query(_read_sql_file("monthly_trend.sql"), conn)
    finally:
        conn.close()
