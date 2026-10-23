"""Train, compare and persist the price model.

Design choices that guard against over-optimistic results:
  * Time-ordered split (train < validation < test) -> simulates predicting FUTURE sales.
  * Imputation / scaling / encoding live inside the sklearn Pipeline -> fitted on training data only.
  * Model selection uses time-series CV (train) and the validation set, never the test set.
  * Prediction interval = split-conformal quantile of validation residuals (multiplicative / log scale).
  * Test set is touched once, for reporting.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from . import config, db
from .explainability import compute_importance
from .feature_engineering import FEATURES, build_features, make_preprocessor


def time_split(df: pd.DataFrame, val_frac: float = config.VAL_FRACTION, test_frac: float = config.TEST_FRACTION):
    df = df.sort_values("sale_date").reset_index(drop=True)
    n = len(df)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def candidate_models() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.6, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=config.RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=config.RANDOM_STATE,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE, verbose=-1,
        )
    except ImportError:
        pass
    return models


def make_pipeline(estimator) -> TransformedTargetRegressor:
    """Preprocessing + estimator, trained on log(price) but predicting dollars."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", make_preprocessor()), ("model", estimator)]),
        func=np.log, inverse_func=np.exp,
    )


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
    }


def main(data_source: str = "synthetic") -> dict:
    config.MODELS.mkdir(parents=True, exist_ok=True)
    amenities = pd.read_csv(config.AMENITIES_CSV)
    data = build_features(db.load_training_frame(), amenities)
    train, val, test = time_split(data)
    X_tr, y_tr = train[FEATURES], train["sale_price"]
    X_va, y_va = val[FEATURES], val["sale_price"]
    X_te, y_te = test[FEATURES], test["sale_price"]
    print(f"[train] rows train/val/test = {len(train):,}/{len(val):,}/{len(test):,}")

    cv = TimeSeriesSplit(n_splits=5)
    results, fitted = [], {}
    for name, est in candidate_models().items():
        pipe = make_pipeline(est)
        scores = cross_validate(
            pipe, X_tr, y_tr, cv=cv, n_jobs=1,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        pipe.fit(X_tr, y_tr)
        val_m = regression_metrics(y_va, pipe.predict(X_va))
        test_m = regression_metrics(y_te, pipe.predict(X_te))
        results.append({
            "model": name,
            "cv_rmse": float(-scores["test_rmse"].mean()), "cv_rmse_std": float(scores["test_rmse"].std()),
            "cv_mae": float(-scores["test_mae"].mean()), "cv_r2": float(scores["test_r2"].mean()),
            "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "val_r2": val_m["r2"],
            "test_mae": test_m["mae"], "test_rmse": test_m["rmse"], "test_r2": test_m["r2"],
            "test_mape_pct": test_m["mape_pct"],
        })
        fitted[name] = pipe
        print(f"[train] {name:<18} CV RMSE ${-scores['test_rmse'].mean():>10,.0f} | "
              f"val RMSE ${val_m['rmse']:>10,.0f} | val R2 {val_m['r2']:.3f}")

    # Selection uses validation RMSE only (the test set is never used to choose).
    best_name = min(results, key=lambda r: r["val_rmse"])["model"]
    best = fitted[best_name]
    best_row = next(r for r in results if r["model"] == best_name)

    # Split-conformal interval on the log scale from validation residuals.
    log_res = np.abs(np.log(y_va.to_numpy()) - np.log(best.predict(X_va)))
    k = min(1.0, np.ceil((len(log_res) + 1) * config.INTERVAL_LEVEL) / len(log_res))
    q = float(np.quantile(log_res, k))
    te_pred = best.predict(X_te)
    coverage = float(np.mean(np.abs(np.log(y_te.to_numpy()) - np.log(te_pred)) <= q))

    importance = compute_importance(best, X_te, y_te)
    config.IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2))

    pd.DataFrame({
        "property_id": test["property_id"].to_numpy(), "neighborhood": test["neighborhood"].to_numpy(),
        "sale_date": test["sale_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual": y_te.to_numpy(), "predicted": te_pred, "residual": y_te.to_numpy() - te_pred,
    }).to_csv(config.TEST_PRED_PATH, index=False)
    pd.DataFrame(results).to_csv(config.COMPARISON_PATH, index=False)

    metrics = {
        "best_model": best_name,
        "selection_rule": "lowest validation RMSE",
        "data_source": data_source,
        "test": {k: best_row[f"test_{k}"] for k in ("mae", "rmse", "r2")} | {"mape_pct": best_row["test_mape_pct"]},
        "validation": {k: best_row[f"val_{k}"] for k in ("mae", "rmse", "r2")},
        "cv": {"rmse": best_row["cv_rmse"], "rmse_std": best_row["cv_rmse_std"], "mae": best_row["cv_mae"], "r2": best_row["cv_r2"],
               "scheme": "TimeSeriesSplit(5) on training data"},
        "interval": {"level": config.INTERVAL_LEVEL, "log_halfwidth": q, "test_coverage": coverage},
        "models": results,
        "split": {
            "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
            "train_end": str(train["sale_date"].max().date()), "val_end": str(val["sale_date"].max().date()),
            "test_start": str(test["sale_date"].min().date()), "test_end": str(test["sale_date"].max().date()),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    bundle = {
        "model": best, "model_name": best_name, "features": FEATURES,
        "interval_log_halfwidth": q, "interval_level": config.INTERVAL_LEVEL,
        "metrics": metrics, "importance": importance, "data_source": data_source,
        "amenities": amenities,
        "neighborhoods": data.groupby("neighborhood").agg(
            city=("city", "first"), zip_code=("zip_code", "first"),
            latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        ).reset_index(),
        "property_types": sorted(data["property_type"].unique().tolist()),
        "lot_defaults": data.groupby("property_type")["lot_size"].median().fillna(0).to_dict(),
        "reference_date": str(data["sale_date"].max().date()),
        "year_range": [int(data["year_built"].min()), int(data["year_built"].max())],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    print(f"[train] best = {best_name} | test RMSE ${best_row['test_rmse']:,.0f} | test R2 {best_row['test_r2']:.3f} "
          f"| interval coverage on test {coverage:.1%} (target {config.INTERVAL_LEVEL:.0%})")
    return metrics


if __name__ == "__main__":
    main()
