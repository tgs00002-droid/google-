# src/data.py
from __future__ import annotations

import os
import time
import pandas as pd
import requests

STOCKANALYSIS_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"

CACHE_DIR = os.path.join("data", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "revenue_by_product_quarterly.csv")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; revenue-forecast-project/1.0; +https://github.com/yourname)"
}

def load_revenue_by_product(force_refresh: bool = False, sleep_s: float = 0.5) -> pd.DataFrame:
    """
    Returns a tidy quarterly dataframe:
      date (quarter end), product, revenue_usd
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force_refresh:
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        return df

    time.sleep(sleep_s)
    r = requests.get(STOCKANALYSIS_URL, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    # The page contains an HTML table we can read.
    tables = pd.read_html(r.text)
    if not tables:
        raise RuntimeError("No tables found on source page.")

    # First table is typically the revenue-by-segment time series.
    wide = tables[0].copy()

    # Expect a first column like "Date" and then product columns
    wide.columns = [str(c).strip() for c in wide.columns]
    if "Date" not in wide.columns:
        raise RuntimeError(f"Unexpected columns: {wide.columns.tolist()}")

    wide["Date"] = pd.to_datetime(wide["Date"])
    # Convert values like "54.10B" or "675M" to numeric USD
    def parse_money(x) -> float:
        if pd.isna(x):
            return float("nan")
        s = str(x).strip().replace(",", "")
        mult = 1.0
        if s.endswith("B"):
            mult = 1e9
            s = s[:-1]
        elif s.endswith("M"):
            mult = 1e6
            s = s[:-1]
        elif s.endswith("K"):
            mult = 1e3
            s = s[:-1]
        return float(s) * mult

    product_cols = [c for c in wide.columns if c != "Date"]
    for c in product_cols:
        wide[c] = wide[c].apply(parse_money)

    tidy = wide.melt(
        id_vars=["Date"],
        value_vars=product_cols,
        var_name="product",
        value_name="revenue_usd",
    ).dropna()

    tidy = tidy.rename(columns={"Date": "date"}).sort_values(["product", "date"])
    tidy.to_csv(CACHE_PATH, index=False)
    return tidy
