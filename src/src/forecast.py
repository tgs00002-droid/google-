# src/forecast.py
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def _prep_series(df: pd.DataFrame, product: str) -> pd.Series:
    s = (
        df[df["product"] == product]
        .set_index("date")["revenue_usd"]
        .asfreq("Q")  # quarterly
        .sort_index()
    )
    # Fill missing quarters conservatively
    s = s.interpolate(limit_direction="both")
    return s

def fit_sarimax_forecast(series: pd.Series, steps: int = 40) -> pd.DataFrame:
    """
    Quarterly forecast for 'steps' quarters (40 = 10 years).
    """
    # Small, robust SARIMAX spec for quarterly data:
    # (1,1,1) with seasonal (1,1,1,4)
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.2)  # ~80% interval for readability

    out = pd.DataFrame({
        "date": mean.index.to_timestamp(how="end"),
        "yhat": mean.values,
        "yhat_lo": ci.iloc[:, 0].values,
        "yhat_hi": ci.iloc[:, 1].values,
    })
    return out

def make_product_forecasts(df: pd.DataFrame, horizon_years: int = 10) -> pd.DataFrame:
    steps = horizon_years * 4
    products = sorted(df["product"].unique())

    all_rows = []
    for p in products:
        s = _prep_series(df, p)
        fc = fit_sarimax_forecast(s, steps=steps)
        fc["product"] = p
        all_rows.append(fc)

    return pd.concat(all_rows, ignore_index=True)

def apply_scenario_overlay(forecasts: pd.DataFrame, annual_uplift_by_product: dict[str, float]) -> pd.DataFrame:
    """
    annual_uplift_by_product: e.g. {"Google Cloud": 0.03} meaning +3% extra CAGR on top of baseline
    Applied smoothly across the horizon.
    """
    out = forecasts.copy()
    out["quarter_index"] = out.groupby("product").cumcount()
    out["year_frac"] = out["quarter_index"] / 4.0

    def adj(row):
        uplift = annual_uplift_by_product.get(row["product"], 0.0)
        factor = (1.0 + uplift) ** row["year_frac"]
        return factor

    factor = out.apply(adj, axis=1).astype(float)
    for col in ["yhat", "yhat_lo", "yhat_hi"]:
        out[col] = out[col] * factor

    return out.drop(columns=["quarter_index", "year_frac"])

def quarterly_to_yearly(dfq: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: date, product, yhat (and optional intervals)
    Output: year, product, revenue_yhat (sum of 4 quarters)
    """
    tmp = dfq.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    agg = tmp.groupby(["year", "product"], as_index=False).agg(
        revenue_yhat=("yhat", "sum"),
        revenue_lo=("yhat_lo", "sum"),
        revenue_hi=("yhat_hi", "sum"),
    )
    return agg

def total_rollup(yearly: pd.DataFrame) -> pd.DataFrame:
    tot = yearly.groupby("year", as_index=False).agg(
        revenue_yhat=("revenue_yhat", "sum"),
        revenue_lo=("revenue_lo", "sum"),
        revenue_hi=("revenue_hi", "sum"),
    )
    tot["product"] = "TOTAL"
    return tot
