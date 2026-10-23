"""Streamlit dashboard: Real Estate Price Prediction & Market Analytics.
Run:  streamlit run app.py   (after `python run_pipeline.py`)"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, db, predict, visuals

st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#e3edff,#f3f7ff);border:1px solid #dbe5fb;border-radius:16px;padding:18px 26px;margin-bottom:14px}
.hero h1{font-size:1.9rem;margin:0;color:#1b2f8a;letter-spacing:.5px}
.hero p{margin:4px 0 0;color:#5a6aa8;letter-spacing:3px;font-size:.78rem;text-transform:uppercase}
.price{font-size:3.2rem;font-weight:800;color:#1b2f8a;text-align:center;line-height:1.1;margin:6px 0}
.sub{text-align:center;color:#5a6aa8;font-size:.95rem}
.step{display:flex;gap:10px;align-items:flex-start;margin:7px 0}
.num{background:#3b7be8;color:#fff;border-radius:50%;min-width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem}
.step small{color:#6b79ad;display:block}
.tech span{display:inline-block;background:#eaf1ff;color:#1b2f8a;border-radius:8px;padding:3px 10px;margin:3px 4px 3px 0;font-size:.85rem}
</style>
""", unsafe_allow_html=True)


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def get_bundle():
    return predict.load_bundle()


@st.cache_data
def get_test_predictions():
    return pd.read_csv(config.TEST_PRED_PATH)


@st.cache_data
def get_clean():
    return pd.read_csv(config.CLEAN_CSV)


@st.cache_data
def get_sql_views():
    return db.neighborhood_summary(), db.monthly_trend()


bundle = get_bundle()
st.markdown('<div class="hero"><h1>REAL ESTATE PRICE PREDICTION &amp; MARKET ANALYTICS</h1>'
            '<p>Machine Learning · Data Analysis · Geospatial Insights</p></div>', unsafe_allow_html=True)

if bundle is None:
    st.error("No trained model found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()
if bundle["data_source"] == "synthetic":
    st.info("Demo mode: the model is trained on **synthetic** data for a fictional city. "
            "Prices are illustrative only. Re-run the pipeline with real sales data (see README).")

m = bundle["metrics"]
left, centre, right = st.columns([1.05, 2.6, 1.15], gap="medium")

# ---------------- left: inputs ----------------
with left:
    with st.container(border=True):
        st.subheader("Project Inputs")
        hoods = bundle["neighborhoods"].sort_values("neighborhood")
        labels = {f"{r.neighborhood} · {r.zip_code}": r.neighborhood for r in hoods.itertuples()}
        with st.form("inputs"):
            loc = st.selectbox("Location", list(labels), index=None, placeholder="Select neighbourhood or ZIP")
            bedrooms = st.selectbox("Bedrooms", list(range(1, 9)), index=2)
            baths = st.selectbox("Bathrooms", [x / 2 for x in range(2, 13)], index=2)
            area = st.number_input("Living Area (sqft)", 300, 12000, 2000, step=50)
            ptype = st.selectbox("Property Type", bundle["property_types"])
            yr_lo, yr_hi = bundle["year_range"]
            year = st.number_input("Year Built", yr_lo, yr_hi, min(2000, yr_hi), step=1, format="%d")
            go_btn = st.form_submit_button("Estimate Price", type="primary")
        if go_btn:
            if loc is None:
                st.warning("Please choose a location.")
            else:
                st.session_state["result"] = predict.predict_price(
                    bundle, labels[loc], bedrooms, baths, area, ptype, year)

# ---------------- centre: result + metrics + charts ----------------
with centre:
    with st.container(border=True):
        st.markdown("**🏠 PREDICTED PROPERTY VALUE**")
        res = st.session_state.get("result")
        if res:
            st.markdown(f'<div class="price">${res["price"]:,.0f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sub">{int(res["level"]*100)}% prediction interval: '
                        f'<b>${res["low"]:,.0f} – ${res["high"]:,.0f}</b> · model: {res["model"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="price">$ — — —</div><div class="sub">Model estimate · prediction interval</div>',
                        unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**MODEL EVALUATION** (held-out, most recent sales)")
        c = st.columns(5)
        c[0].metric("MAE", f"${m['test']['mae']:,.0f}")
        c[1].metric("RMSE", f"${m['test']['rmse']:,.0f}")
        c[2].metric("R²", f"{m['test']['r2']:.3f}")
        c[3].metric("Cross-val R²", f"{m['cv']['r2']:.3f}")
        c[4].metric("Interval coverage", f"{m['interval']['test_coverage']:.0%}",
                    help=f"Share of test sales inside the {int(m['interval']['level']*100)}% interval.")
    a, b = st.columns(2)
    pred_df = get_test_predictions()
    with a:
        with st.container(border=True):
            show(visuals.actual_vs_predicted(pred_df))
    with b:
        with st.container(border=True):
            show(visuals.residuals(pred_df))
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            show(visuals.importance_bars(bundle["importance"]))
    with b:
        with st.container(border=True):
            show(visuals.model_comparison(m["models"]))

# ---------------- right: tech stack + pipeline ----------------
with right:
    with st.container(border=True):
        st.subheader("Tech Stack")
        st.markdown('<div class="tech">' + "".join(f"<span>{t}</span>" for t in
                    ["Python", "SQL", "Pandas", "Scikit-learn", "LightGBM / XGBoost", "GeoPandas", "SHAP", "Streamlit"])
                    + "</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Project Pipeline")
        steps = [("Data Collection", "housing listings / sales"), ("SQL + Cleaning", "standardize, handle missing values"),
                 ("Geospatial Features", "location, distance, neighbourhood"),
                 ("Train Regression Models", "Linear, RF, GBM, XGBoost, LightGBM"),
                 ("Evaluate & Explain", "performance + SHAP"), ("Deploy Dashboard", "Streamlit app")]
        st.markdown("".join(f'<div class="step"><div class="num">{i}</div><div><b>{t}</b><small>{d}</small></div></div>'
                            for i, (t, d) in enumerate(steps, 1)), unsafe_allow_html=True)

# ---------------- market analytics ----------------
st.markdown("### Market Analytics")
t1, t2, t3, t4 = st.tabs(["Neighbourhood prices", "Price trend", "Map", "Model details"])
summary, trend = get_sql_views()
with t1:
    show(visuals.neighborhood_bars(summary))
    st.dataframe(summary.round(0), hide_index=True)
with t2:
    show(visuals.monthly_trend(trend))
with t3:
    show(visuals.price_map(get_clean()))
with t4:
    s = m["split"]
    st.write(f"**Best model:** {m['best_model']} (selected by {m['selection_rule']}). "
             f"Time-based split: train ≤ {s['train_end']}, validation ≤ {s['val_end']}, test {s['test_start']} → {s['test_end']}.")
    st.dataframe(pd.DataFrame(m["models"]).round(3), hide_index=True)
    if bundle["importance"].get("note"):
        st.caption(bundle["importance"]["note"])
    st.markdown("**Limitations:** the model learns from past sales and cannot see condition, renovations, views, or market "
                "shocks; tree models do not extrapolate beyond the price levels they have seen; the prediction interval is "
                "calibrated on validation data and may under-cover when the market shifts (check the coverage figure above).")

st.caption("📍 Use real market data; validate by location and time; estimates are not appraisals.")
