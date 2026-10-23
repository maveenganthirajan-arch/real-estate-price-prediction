"""Feature importance: SHAP when installed (tree/linear models), permutation importance otherwise."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .feature_engineering import CATEGORICAL_FEATURES, FEATURE_GROUPS, FEATURES

SHAP_TREE_MODELS = {"RandomForestRegressor", "GradientBoostingRegressor", "LGBMRegressor", "XGBRegressor"}


def _base_feature(encoded_name: str) -> str:
    name = encoded_name.split("__", 1)[1]
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


def _shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per original feature (one-hot columns summed back). Values are in log-price units."""
    import shap  # optional dependency

    inner = model.regressor_
    prep, est = inner.named_steps["prep"], inner.named_steps["model"]
    Xs = X[FEATURES].sample(min(600, len(X)), random_state=config.RANDOM_STATE)
    Xt = prep.transform(Xs)
    kind = type(est).__name__
    if kind in SHAP_TREE_MODELS:
        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(Xt, check_additivity=False)
    elif kind == "LinearRegression":
        values = shap.LinearExplainer(est, Xt).shap_values(Xt)
    else:
        raise ValueError(f"SHAP not configured for {kind}")
    names = [_base_feature(n) for n in prep.get_feature_names_out()]
    return pd.Series(np.abs(values).mean(axis=0), index=names).groupby(level=0).sum()


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Increase in RMSE ($) when a feature is shuffled on held-out data."""
    r = permutation_importance(
        model, X[FEATURES], y, scoring="neg_root_mean_squared_error",
        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1,
    )
    return pd.Series(np.clip(r.importances_mean, 0, None), index=FEATURES)


def _table(series: pd.Series, label: str) -> list[dict]:
    s = series.sort_values(ascending=False)
    total, top = s.sum() or 1.0, s.max() or 1.0
    return [
        {label: k, "importance": float(v), "share_pct": float(100 * v / total), "relative_pct": float(100 * v / top)}
        for k, v in s.items()
    ]


def compute_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    method, note = "permutation", None
    try:
        s = _shap_importance(model, X_test)
        method = "shap"
    except Exception as exc:  # shap missing / unsupported model -> permutation fallback
        note = f"SHAP unavailable ({type(exc).__name__}: {exc}); used permutation importance."
        s = _permutation_importance(model, X_test, y_test)

    groups = {g: float(s.reindex(cols).fillna(0).sum()) for g, cols in FEATURE_GROUPS.items()}
    return {
        "method": method,
        "note": note,
        "unit": "mean |SHAP| (log-price)" if method == "shap" else "RMSE increase ($)",
        "features": _table(s, "feature"),
        "groups": _table(pd.Series(groups), "group"),
    }
