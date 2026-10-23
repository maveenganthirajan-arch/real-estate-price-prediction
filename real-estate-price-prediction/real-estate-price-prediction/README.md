# Real Estate Price Prediction & Market Analytics

Machine learning · SQL · geospatial features · explainability · Streamlit dashboard.
Layout reference: `docs/reference_dashboard.png`.

> **Important:** the bundled dataset is **synthetic** (a fictional city) so the project runs offline.
> Prices and metrics below say nothing about any real market. Swap in real sales data before drawing conclusions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # generate data -> clean -> SQL -> features -> train -> explain
streamlit run app.py            # open the dashboard
```

The pipeline works even without LightGBM / XGBoost / SHAP / GeoPandas installed: it falls back to
scikit-learn models, permutation importance and plain-JSON GeoJSON. Install everything for the full experience.
Re-run `run_pipeline.py` on your machine so `models/best_model.joblib` matches your scikit-learn version.

## What it does

| Stage | Where | Notes |
|---|---|---|
| Data | `src/generate_data.py` | Synthetic listings with deliberate dirt (dups, gaps, outliers) |
| Cleaning | `src/data_cleaning.py` | Dedupe, invalid values to NaN, drop rows missing critical fields, price/sqft outliers |
| SQL | `sql/*.sql`, `src/db.py` | Normalised `neighborhoods / properties / sales`; SQLite by default, PostgreSQL-compatible schema |
| Geospatial | `src/geospatial.py` | Haversine distance to downtown, parks, schools, transit; GeoPandas/GeoJSON helpers |
| Features | `src/feature_engineering.py` | Age, area per bedroom, sale time, distances; imputation/scaling/encoding inside the pipeline |
| Models | `src/train_model.py` | Linear Regression baseline, Random Forest, Gradient Boosting, plus XGBoost / LightGBM if installed |
| Explainability | `src/explainability.py` | SHAP (tree/linear) or permutation-importance fallback, grouped (e.g. Location = neighbourhood + coordinates + distances) |
| Dashboard | `app.py`, `src/visuals.py` | Inputs, estimate + interval, metrics, actual-vs-predicted, residuals, importance, model comparison, SQL market views, map |

## Leakage and validation safeguards

- **Time-ordered split** (oldest 70 % train, next 15 % validation, newest 15 % test), so the test measures predicting future sales.
- Imputation, scaling and encoding sit **inside** the sklearn `Pipeline`; they are fitted on training data only.
- Model selection uses time-series CV and **validation RMSE**, never the test set.
- The 90 % prediction interval is a split-conformal quantile of validation residuals (multiplicative). Coverage on the test set is reported, not assumed.
- Predicting **unfamiliar locations** would need geographic (group) validation; this project validates over time only.

## Results on the bundled synthetic data

Best model: **Gradient Boosting** (chosen by validation RMSE). Test R² 0.955, MAE $39,588, RMSE $55,241.
The nominal 90 % interval covered **84.7%** of test sales: slightly under target, because the market drifts after the calibration period.

| Model | CV RMSE | Val RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Linear Regression | $70,203 | $77,321 | $52,282 | $81,643 | 0.901 |
| Random Forest | $72,247 | $80,104 | $62,110 | $87,073 | 0.888 |
| Gradient Boosting | $49,497 | $49,257 | $39,588 | $55,241 | 0.955 |

Full numbers: `models/metrics.json`, `models/model_comparison.csv`.

## Using your own data

```bash
python run_pipeline.py --data my_sales.csv --amenities my_amenities.csv
```

`my_sales.csv` columns: `property_id, city, neighborhood, zip_code, latitude, longitude, bedrooms, bathrooms, living_area, lot_size, property_type, year_built, sale_date, sale_price`.
`my_amenities.csv` columns: `amenity_type, latitude, longitude` with types `downtown`, `park`, `school`, `transit`.
Without `--amenities`, placeholder points are generated and the distance features are meaningless.
Good public sources: King County House Sales, Ames Housing, or your local assessor / MLS export.
Adjust `PROPERTY_TYPES` in `src/config.py` and the validity bounds in `src/data_cleaning.py` for your market.

PostgreSQL: run `sql/schema.sql` in your database and point `src/db.py` at it (swap `sqlite3` for SQLAlchemy/psycopg2).

## Structure

```
real-estate-price-prediction/
├── data/{raw,processed}/      generated CSVs and the SQLite DB
├── notebooks/                 01_eda, 02_geospatial_analysis, 03_model_training
├── sql/                       schema.sql and analytical queries
├── src/                       data_cleaning, feature_engineering, train_model, predict, explainability, ...
├── models/                    best_model.joblib, metrics.json, feature_importance.json, test_predictions.csv
├── docs/reference_dashboard.png
├── app.py   run_pipeline.py   requirements.txt   README.md
```

## Limitations

Estimates are not appraisals. The model cannot see condition, renovations, views or sudden market shocks; tree models
do not extrapolate beyond the price range they were trained on; and no accuracy figure should be advertised until it is
reproduced on real data with proper validation.
