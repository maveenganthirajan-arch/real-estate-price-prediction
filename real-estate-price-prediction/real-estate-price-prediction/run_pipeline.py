"""One command to run everything: generate/load data -> clean -> SQL -> features -> train -> explain.

    python run_pipeline.py                       # synthetic demo data
    python run_pipeline.py --n 20000             # bigger synthetic set
    python run_pipeline.py --data my_sales.csv   # your own data (same columns, see README)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config, data_cleaning, db, generate_data, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000, help="rows of synthetic data to generate")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--data", type=str, default=None, help="path to your own listings CSV")
    ap.add_argument("--amenities", type=str, default=None, help="CSV: amenity_type,latitude,longitude")
    args = ap.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if args.data:
        raw = pd.read_csv(args.data, dtype={"zip_code": str})
        source = "user-provided"
        if args.amenities:
            pd.read_csv(args.amenities).to_csv(config.AMENITIES_CSV, index=False)
        else:
            print("[warn] no --amenities file: using PLACEHOLDER amenities; distance features are not meaningful.")
            generate_data.make_placeholder_amenities(raw).to_csv(config.AMENITIES_CSV, index=False)
    else:
        generate_data.main(args.n, args.seed)
        raw = pd.read_csv(config.RAW_CSV, dtype={"zip_code": str})
        source = "synthetic"

    clean, report = data_cleaning.clean_listings(raw)
    data_cleaning.save(clean, report)
    print(f"[clean] {report['rows_raw']:,} raw -> {report['rows_clean']:,} clean rows | {report}")

    conn = db.connect()
    db.write_clean(clean, conn)
    conn.close()
    print(f"[sql] loaded into {config.DB_PATH}")

    train_model.main(data_source=source)
    print("\nDone. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
