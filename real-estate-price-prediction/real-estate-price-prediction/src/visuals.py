"""Plotly figures for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE, TEAL, PURPLE = "#2f6fe4", "#2ec4b6", "#8e6bd8"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=330,
              font=dict(color="#1b2559"), paper_bgcolor="rgba(0,0,0,0)")


def actual_vs_predicted(pred: pd.DataFrame, max_points: int = 1500) -> go.Figure:
    d = pred.sample(min(max_points, len(pred)), random_state=0)
    fig = px.scatter(d, x="actual", y="predicted", opacity=0.55, color_discrete_sequence=[BLUE],
                     hover_data=["neighborhood"], labels={"actual": "Actual price ($)", "predicted": "Predicted price ($)"})
    lo, hi = float(min(d.actual.min(), d.predicted.min())), float(max(d.actual.max(), d.predicted.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal fit (y = x)",
                             line=dict(color="#c0392b", dash="dash")))
    fig.update_layout(**LAYOUT, title="Actual vs Predicted (held-out test set)", showlegend=True,
                      legend=dict(orientation="h", y=-0.25))
    return fig


def residuals(pred: pd.DataFrame, max_points: int = 1500) -> go.Figure:
    d = pred.sample(min(max_points, len(pred)), random_state=0)
    fig = px.scatter(d, x="predicted", y="residual", opacity=0.55, color_discrete_sequence=[PURPLE],
                     hover_data=["neighborhood"], labels={"predicted": "Predicted price ($)", "residual": "Residual ($)"})
    fig.add_hline(y=0, line_dash="dash", line_color="#c0392b")
    fig.update_layout(**LAYOUT, title="Residual analysis (actual − predicted)")
    return fig


def importance_bars(importance: dict, level: str = "groups", top: int = 8) -> go.Figure:
    key = "group" if level == "groups" else "feature"
    rows = importance[level][:top][::-1]
    fig = go.Figure(go.Bar(
        x=[r["share_pct"] for r in rows], y=[r[key] for r in rows], orientation="h",
        marker=dict(color=[BLUE] * len(rows)), text=[f"{r['share_pct']:.0f}%" for r in rows], textposition="outside",
    ))
    method = "SHAP" if importance["method"] == "shap" else "Permutation"
    fig.update_layout(**LAYOUT, title=f"Feature importance ({method}, share of total)",
                      xaxis_title="Share of total importance (%)", xaxis=dict(range=[0, max(r["share_pct"] for r in rows) * 1.2]))
    return fig


def model_comparison(models: list[dict]) -> go.Figure:
    d = sorted(models, key=lambda r: r["test_r2"])
    fig = go.Figure(go.Bar(
        x=[r["test_r2"] for r in d], y=[r["model"] for r in d], orientation="h",
        marker=dict(color=[TEAL if i == len(d) - 1 else BLUE for i in range(len(d))]),
        text=[f"R² {r['test_r2']:.3f} · RMSE ${r['test_rmse']/1000:,.0f}K" for r in d], textposition="inside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(**LAYOUT, title="Model comparison (test R²)", xaxis=dict(range=[0, 1]))
    return fig


def neighborhood_bars(summary: pd.DataFrame) -> go.Figure:
    d = summary.sort_values("avg_price_per_sqft")
    fig = px.bar(d, x="avg_price_per_sqft", y="neighborhood", orientation="h", color="avg_price",
                 color_continuous_scale="Blues", labels={"avg_price_per_sqft": "Avg price / sqft ($)", "neighborhood": "",
                                                          "avg_price": "Avg price ($)"})
    fig.update_layout(**{**LAYOUT, "height": 420}, title="Average price per sqft by neighbourhood")
    return fig


def monthly_trend(trend: pd.DataFrame) -> go.Figure:
    fig = px.line(trend, x="month", y="avg_price_per_sqft", markers=False, color_discrete_sequence=[BLUE],
                  labels={"month": "", "avg_price_per_sqft": "Avg price / sqft ($)"})
    fig.update_layout(**LAYOUT, title="Monthly average price per sqft")
    return fig


def price_map(df: pd.DataFrame, max_points: int = 3000) -> go.Figure:
    d = df.sample(min(max_points, len(df)), random_state=0).assign(price_per_sqft=lambda x: x.sale_price / x.living_area)
    kwargs = dict(lat="latitude", lon="longitude", color="price_per_sqft", color_continuous_scale="RdYlBu_r",
                  hover_name="neighborhood", hover_data={"sale_price": ":,.0f", "latitude": False, "longitude": False},
                  zoom=10, height=520, opacity=0.7, labels={"price_per_sqft": "$/sqft"})
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(d, map_style="open-street-map", **kwargs)
    else:
        fig = px.scatter_mapbox(d, mapbox_style="open-street-map", **kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
