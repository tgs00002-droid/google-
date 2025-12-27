# src/metrics.py
from __future__ import annotations
import pandas as pd

def yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    df tidy: date, product, revenue_usd
    """
    x = df.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    yearly = x.groupby(["year", "product"], as_index=False)["revenue_usd"].sum()
    yearly["yoy"] = yearly.groupby("product")["revenue_usd"].pct_change()
    return yearly

def mix_share(yearly: pd.DataFrame) -> pd.DataFrame:
    """
    yearly: year, product, revenue_yhat
    """
    x = yearly.copy()
    total = x.groupby("year")["revenue_yhat"].transform("sum")
    x["mix_share"] = x["revenue_yhat"] / total
    return x
