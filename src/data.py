# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data import load_revenue_by_product
from src.forecast import (
    make_product_forecasts,
    apply_scenario_overlay,
    quarterly_to_yearly,
    total_rollup,
)
from src.metrics import yoy_growth, mix_share

st.set_page_config(page_title="Google 10-Year Revenue Forecast", layout="wide")

st.title("Alphabet (Google) Revenue: 10-Year Forecast by Product")
st.caption(
    "Quarterly revenue lines (Search, YouTube Ads, Cloud, etc.) modeled and projected 10 years forward, with scenario levers by product."
)

with st.sidebar:
    st.header("Controls")
    force_refresh = st.checkbox("Force refresh data from source", value=False)
    horizon_years = st.slider("Forecast horizon (years)", 5, 15, 10)

@st.cache_data(show_spinner=False)
def get_data(force_refresh: bool) -> pd.DataFrame:
    return load_revenue_by_product(force_refresh=force_refresh)

@st.cache_data(show_spinner=True)
def get_baseline_forecasts(df: pd.DataFrame, horizon_years: int) -> pd.DataFrame:
    return make_product_forecasts(df, horizon_years=horizon_years)

df = get_data(force_refresh)
products = sorted(df["product"].unique())

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Baseline Forecast", "Scenarios", "Mix & Growth Story"])

with tab1:
    st.subheader("Raw quarterly revenue by product")
    st.dataframe(df.sort_values(["date", "product"]), use_container_width=True)

    p = st.selectbox("Quick chart product", products)
    d1 = df[df["product"] == p].sort_values("date")
    fig = px.line(d1, x="date", y="revenue_usd", title=f"{p}: Quarterly Revenue")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Baseline model (SARIMAX per product, quarterly seasonality)")
    fc = get_baseline_forecasts(df, horizon_years)

    p2 = st.selectbox("View forecast for product", products, key="p2")
    f2 = fc[fc["product"] == p2].sort_values("date")

    fig2 = px.line(f2, x="date", y="yhat", title=f"Forecast: {p2}")
    fig2.add_scatter(x=f2["date"], y=f2["yhat_lo"], mode="lines", name="80% low")
    fig2.add_scatter(x=f2["date"], y=f2["yhat_hi"], mode="lines", name="80% high")
    st.plotly_chart(fig2, use_container_width=True)

    yearly = quarterly_to_yearly(fc)
    total = total_rollup(yearly)
    figT = px.line(total.sort_values("year"), x="year", y="revenue_yhat", title="TOTAL Revenue (all products) – Baseline")
    st.plotly_chart(figT, use_container_width=True)

with tab3:
    st.subheader("Scenario builder (extra CAGR uplift by product)")
    st.write("Use these sliders to model revenue strategy improvements by product (pricing, growth, distribution, attach, etc.).")

    uplift = {}
    cols = st.columns(3)
    for i, p in enumerate(products):
        with cols[i % 3]:
            uplift[p] = st.slider(f"{p} uplift (extra CAGR)", -0.05, 0.10, 0.00, 0.005)

    fc_base = get_baseline_forecasts(df, horizon_years)
    fc_scn = apply_scenario_overlay(fc_base, uplift)

    y_base = total_rollup(quarterly_to_yearly(fc_base)).sort_values("year")
    y_scn  = total_rollup(quarterly_to_yearly(fc_scn)).sort_values("year")

    comp = y_base[["year", "revenue_yhat"]].merge(
        y_scn[["year", "revenue_yhat"]],
        on="year",
        suffixes=("_baseline", "_scenario"),
    )
    comp["delta"] = comp["revenue_yhat_scenario"] - comp["revenue_yhat_baseline"]

    st.metric("10Y total uplift (final year, scenario vs baseline)",
              f"${comp.iloc[-1]['delta']/1e9:,.1f}B")

    fig3 = px.line(comp, x="year", y=["revenue_yhat_baseline", "revenue_yhat_scenario"],
                   title="TOTAL Revenue: Baseline vs Scenario")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Mix and growth narrative")
    fc = get_baseline_forecasts(df, horizon_years)
    yearly = quarterly_to_yearly(fc)
    mix = mix_share(yearly)

    year_pick = st.selectbox("Pick a forecast year", sorted(mix["year"].unique()))
    m = mix[mix["year"] == year_pick].sort_values("revenue_yhat", ascending=False)

    figm = px.bar(m, x="product", y="mix_share", title=f"Revenue mix share – {year_pick}")
    st.plotly_chart(figm, use_container_width=True)

    # Historical YoY story (from raw data)
    hist_yoy = yoy_growth(df).dropna()
    p3 = st.selectbox("Historical YoY for product", sorted(hist_yoy["product"].unique()), key="p3")
    hy = hist_yoy[hist_yoy["product"] == p3]
    figy = px.line(hy, x="year", y="yoy", title=f"Historical YoY growth – {p3}")
    st.plotly_chart(figy, use_container_width=True)

st.caption(
    "Products reflect Alphabet reporting lines (Search & other, YouTube ads, Cloud, subscriptions/platforms/devices, etc.). "
    "Always sanity-check results against the latest Alphabet earnings release / 10-K."
)
