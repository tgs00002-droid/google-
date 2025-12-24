"""
Master Finance Analytics App (Streamlit)
Portfolio-grade financial analytics and storytelling from a centralized master dataset.

Repo layout expected:
- app.py
- master.csv
- requirements.txt

This app is designed to:
- Turn raw financial and economic data into clear insights, not just charts
- Include historical trends, forecasts, and scenario analysis
- Explain why changes are happening, not only what happened
- Be understandable to non-technical users while still analytically rigorous
- Reflect hiring-manager expectations for clarity, scalability, and decision-driven design

Author: Thomas Selassie (customize in README, not required here)
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


# ============================================================
# 0) Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="Master Finance Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make the first render immediate to avoid "app in the oven" hangs
st.title("Master Finance Analytics")
st.caption("Decision-driven financial storytelling from a centralized master dataset.")


# ============================================================
# 1) Core app constants and defaults
# ============================================================

DEFAULT_CSV_PATH = "master.csv"

MIN_ROWS_FOR_FORECAST = 6
DEFAULT_FORECAST_HORIZON = 5

PCT_FORMAT = ".1%"
INT_FORMAT = ","
FLOAT_FORMAT = ",.2f"

MAX_TABLE_ROWS = 1000


# ============================================================
# 2) Utility formatting helpers
# ============================================================

def is_nan(x) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def fmt_money(x: float) -> str:
    if is_nan(x):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)

    if x >= 1e12:
        return f"{sign}${x/1e12:.2f}T"
    if x >= 1e9:
        return f"{sign}${x/1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}${x/1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x:,.0f}"
    return f"{sign}${x:.2f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if is_nan(x):
        return "—"
    return f"{100*float(x):.{digits}f}%"


def fmt_number(x: float, digits: int = 2) -> str:
    if is_nan(x):
        return "—"
    return f"{float(x):,.{digits}f}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# 3) Data model assumptions and detection
# ============================================================

@dataclass
class ColumnMap:
    time_col: str
    entity_col: Optional[str]
    metric_col: Optional[str]
    value_col: Optional[str]

    # Wide-format candidate metric columns
    wide_metric_cols: List[str]

    # Common canonical columns (if present)
    revenue_col: Optional[str]
    op_income_col: Optional[str]
    assets_col: Optional[str]
    close_col: Optional[str]

    # Optional precomputed columns
    yoy_cols: List[str]
    margin_cols: List[str]
    return_cols: List[str]
    index_cols: List[str]


def guess_time_column(df: pd.DataFrame) -> str:
    """
    Identify a time column. Priority:
    - 'date'
    - 'year'
    - any column containing 'date' or 'year'
    """
    cols = list(df.columns)
    lowered = {c: c.lower() for c in cols}

    for candidate in ["date", "year", "period", "quarter"]:
        for c in cols:
            if lowered[c] == candidate:
                return c

    for c in cols:
        lc = lowered[c]
        if "date" in lc:
            return c
    for c in cols:
        lc = lowered[c]
        if "year" in lc:
            return c

    return cols[0]


def guess_entity_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lowered = {c: c.lower() for c in cols}
    for candidate in ["entity", "ticker", "symbol", "company", "series", "name"]:
        for c in cols:
            if lowered[c] == candidate:
                return c
    for c in cols:
        lc = lowered[c]
        if "ticker" in lc or "symbol" in lc or "company" in lc:
            return c
    return None


def guess_long_format_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to detect long format:
    - metric column: 'metric', 'indicator', 'field'
    - value column: 'value', 'amount'
    """
    cols = list(df.columns)
    lowered = {c: c.lower() for c in cols}

    metric_col = None
    value_col = None

    for c in cols:
        if lowered[c] in ["metric", "indicator", "field", "kpi", "measure"]:
            metric_col = c
    for c in cols:
        if lowered[c] in ["value", "amount", "val", "numeric_value"]:
            value_col = c

    if metric_col and value_col:
        return metric_col, value_col

    # Secondary heuristic: object-like column with many repeats (metric),
    # numeric-like column (value).
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    if value_col is None and len(num_cols) == 1:
        value_col = num_cols[0]

    if metric_col is None:
        for c in obj_cols:
            nunique = df[c].nunique(dropna=True)
            if 5 <= nunique <= 200:
                metric_col = c
                break

    if metric_col and value_col:
        return metric_col, value_col

    return None, None


def find_best_match(cols: List[str], targets: List[str]) -> Optional[str]:
    lowered = {c: c.lower() for c in cols}
    for t in targets:
        for c in cols:
            if lowered[c] == t:
                return c
    for t in targets:
        for c in cols:
            if t in lowered[c]:
                return c
    return None


def detect_column_map(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)

    time_col = guess_time_column(df)
    entity_col = guess_entity_column(df)
    metric_col, value_col = guess_long_format_columns(df)

    # Determine wide metric columns if not long format
    wide_metric_cols = []
    if metric_col is None or value_col is None:
        exclude = {time_col}
        if entity_col:
            exclude.add(entity_col)

        candidates = [c for c in cols if c not in exclude]
        numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        wide_metric_cols = numeric_candidates

    # Common canonical metrics
    revenue_col = find_best_match(cols, ["revenue", "total_revenue", "sales"])
    op_income_col = find_best_match(cols, ["operating_income", "op_income", "ebit"])
    assets_col = find_best_match(cols, ["total_assets", "assets"])
    close_col = find_best_match(cols, ["close", "adj_close", "price", "close_price"])

    yoy_cols = [c for c in cols if "yoy" in c.lower() or "growth" in c.lower()]
    margin_cols = [c for c in cols if "margin" in c.lower()]
    return_cols = [c for c in cols if "return" in c.lower()]
    index_cols = [c for c in cols if "index" in c.lower()]

    return ColumnMap(
        time_col=time_col,
        entity_col=entity_col,
        metric_col=metric_col,
        value_col=value_col,
        wide_metric_cols=wide_metric_cols,
        revenue_col=revenue_col,
        op_income_col=op_income_col,
        assets_col=assets_col,
        close_col=close_col,
        yoy_cols=yoy_cols,
        margin_cols=margin_cols,
        return_cols=return_cols,
        index_cols=index_cols,
    )


# ============================================================
# 4) Data loading and cleaning
# ============================================================

@st.cache_data(show_spinner=False)
def load_master_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleanup of column names
    df.columns = [c.strip() for c in df.columns]

    # Attempt to parse dates for any date-like column
    for c in df.columns:
        lc = c.lower()
        if "date" in lc:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass

    # Attempt to parse year-like columns
    for c in df.columns:
        if c.lower() == "year":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def standardize_time(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Ensure a canonical time column exists and is sortable.
    If time is a date, convert to year for annual datasets but keep date if granular.
    """
    d = df.copy()

    if pd.api.types.is_datetime64_any_dtype(d[time_col]):
        # If has multiple entries per year, keep date; else use year
        years = d[time_col].dt.year
        if years.nunique(dropna=True) <= max(3, len(d) // 4):
            d["year"] = years.astype("Int64")
            return d, "year"
        else:
            return d, time_col

    # If integer year or string year
    if d[time_col].dtype == "object":
        # Try parse to int year
        parsed = pd.to_numeric(d[time_col], errors="coerce")
        if parsed.notna().mean() > 0.7:
            d["year"] = parsed.astype("Int64")
            return d, "year"

    if pd.api.types.is_integer_dtype(d[time_col]) or str(d[time_col].dtype).startswith("Int"):
        return d, time_col

    return d, time_col


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                continue
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ============================================================
# 5) Derived metrics: growth, margins, indexes, returns
# ============================================================

def compute_yoy(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return s.pct_change()


def compute_index(series: pd.Series, base: float = 100.0) -> pd.Series:
    s = series.astype(float)
    first = s.dropna().iloc[0] if s.dropna().shape[0] else np.nan
    if is_nan(first) or first == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    return base * (s / first)


def compute_margin(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = numer.astype(float)
    d = denom.astype(float)
    out = n / d.replace({0: np.nan})
    return out


def compute_annual_return(price: pd.Series) -> pd.Series:
    p = price.astype(float)
    return p.pct_change()


# ============================================================
# 6) Forecasting: explainable trend model + bootstrap interval
# ============================================================

@dataclass
class ForecastResult:
    horizon: int
    forecast_df: pd.DataFrame
    mape_backtest: Optional[float]
    model_note: str


def trend_forecast(
    x_time: pd.Series,
    y: pd.Series,
    horizon: int,
    log_scale: bool = True,
    boot_n: int = 400,
    interval_lo: float = 10.0,
    interval_hi: float = 90.0,
) -> Optional[ForecastResult]:
    """
    Explainable baseline model:
    - Linear regression on time vs log(y) if y positive (captures exponential-like growth)
    - Residual bootstrap for uncertainty intervals
    - Simple rolling backtest (MAPE) when enough history exists
    """
    df_fit = pd.DataFrame({"t": x_time, "y": y}).dropna()
    if len(df_fit) < MIN_ROWS_FOR_FORECAST:
        return None

    # Convert time to numeric
    t = df_fit["t"]
    if pd.api.types.is_datetime64_any_dtype(t):
        tx = t.view("int64") / 1e9
    else:
        tx = pd.to_numeric(t, errors="coerce")

    if tx.isna().mean() > 0.2:
        return None

    X = tx.values.reshape(-1, 1)
    yv = df_fit["y"].astype(float).values

    use_log = log_scale and np.all(yv > 0)

    if use_log:
        y_model = np.log(yv)
        model_note = "Model: linear trend on log scale (captures long-run growth)."
    else:
        y_model = yv
        model_note = "Model: linear trend on original scale (log not valid)."

    model = LinearRegression()
    model.fit(X, y_model)

    y_hat = model.predict(X)
    resid = y_model - y_hat

    # Future time points: if annual numeric, step by 1; else use equal spacing
    if not pd.api.types.is_datetime64_any_dtype(df_fit["t"]):
        t_last = float(tx.iloc[-1])
        steps = np.arange(1, horizon + 1)
        t_future = t_last + steps
        Xf = t_future.reshape(-1, 1)
        future_t_display = (pd.Series(t_future) if df_fit["t"].dtype != "Int64" else pd.Series(t_future).astype("Int64"))
    else:
        # For date series: use frequency based on median delta
        t_sorted = df_fit["t"].sort_values()
        deltas = t_sorted.diff().dropna().dt.days
        step_days = int(np.median(deltas)) if len(deltas) else 365
        last_date = t_sorted.iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=step_days * i) for i in range(1, horizon + 1)]
        Xf = (pd.Series(future_dates).view("int64") / 1e9).values.reshape(-1, 1)
        future_t_display = pd.Series(future_dates)

    pred = model.predict(Xf)

    rng = np.random.default_rng(42)
    sims = []
    if len(resid) >= 6:
        for _ in range(boot_n):
            sampled = rng.choice(resid, size=horizon, replace=True)
            sims.append(pred + sampled)
        sims = np.vstack(sims)
        lo = np.percentile(sims, interval_lo, axis=0)
        hi = np.percentile(sims, interval_hi, axis=0)
    else:
        lo = pred
        hi = pred

    if use_log:
        pred_y = np.exp(pred)
        lo_y = np.exp(lo)
        hi_y = np.exp(hi)
    else:
        pred_y = pred
        lo_y = lo
        hi_y = hi

    fc = pd.DataFrame(
        {
            "t": future_t_display,
            "forecast": pred_y,
            "lo": lo_y,
            "hi": hi_y,
        }
    )

    # Backtest: last 3 points, fit on earlier data if enough
    mape_val = None
    if len(df_fit) >= 10:
        k = 3
        df_train = df_fit.iloc[:-k].copy()
        df_test = df_fit.iloc[-k:].copy()

        t_train = df_train["t"]
        if pd.api.types.is_datetime64_any_dtype(t_train):
            tx_train = (t_train.view("int64") / 1e9).values.reshape(-1, 1)
            tx_test = (df_test["t"].view("int64") / 1e9).values.reshape(-1, 1)
        else:
            tx_train = pd.to_numeric(t_train, errors="coerce").values.reshape(-1, 1)
            tx_test = pd.to_numeric(df_test["t"], errors="coerce").values.reshape(-1, 1)

        y_train = df_train["y"].astype(float).values
        y_test = df_test["y"].astype(float).values

        if use_log and np.all(y_train > 0) and np.all(y_test > 0):
            y_train_m = np.log(y_train)
            y_test_true = y_test
            bt = LinearRegression().fit(tx_train, y_train_m)
            y_pred = np.exp(bt.predict(tx_test))
        else:
            bt = LinearRegression().fit(tx_train, y_train)
            y_pred = bt.predict(tx_test)

        # MAPE requires nonzero actuals
        mask = y_test_true != 0 if "y_test_true" in locals() else y_test != 0
        if mask.sum() >= 2:
            actual = (y_test_true if "y_test_true" in locals() else y_test)[mask]
            predv = y_pred[mask]
            mape_val = float(mean_absolute_percentage_error(actual, predv))

    return ForecastResult(horizon=horizon, forecast_df=fc, mape_backtest=mape_val, model_note=model_note)


def apply_scenario_adjustment(fc: pd.DataFrame, annual_uplift: float) -> pd.DataFrame:
    """
    Apply a scenario multiplicative uplift/drag across horizon:
    forecast_t = forecast_t * (1 + uplift)^k
    """
    out = fc.copy()
    if annual_uplift == 0:
        return out

    years = np.arange(1, len(out) + 1)
    factor = (1.0 + annual_uplift) ** years

    out["forecast"] = out["forecast"].astype(float) * factor
    out["lo"] = out["lo"].astype(float) * factor
    out["hi"] = out["hi"].astype(float) * factor
    return out


# ============================================================
# 7) Narrative generation: "why" and "so what"
# ============================================================

def infer_direction(recent: pd.Series) -> str:
    r = recent.dropna().astype(float)
    if len(r) < 3:
        return "stable"
    slope = r.diff().mean()
    if slope > 0:
        return "increasing"
    if slope < 0:
        return "decreasing"
    return "stable"


def relationship_statement(corr: float) -> str:
    if is_nan(corr):
        return "There is not enough data to estimate a reliable relationship between these two series."
    if corr >= 0.70:
        return "The relationship is strong and positive, suggesting fundamentals and market pricing have moved together in this window."
    if corr >= 0.35:
        return "The relationship is moderately positive, suggesting fundamentals matter but other drivers also influence pricing."
    if corr > -0.10:
        return "The relationship is weak, suggesting market pricing may be dominated by expectations, macro conditions, or valuation multiple changes."
    return "The relationship is negative, suggesting the market may be reacting to factors not captured by the selected fundamental metric."


def forecast_explanation(
    metric_label: str,
    hist: pd.Series,
    fc: pd.DataFrame,
    scenario_name: str,
    model_note: str,
    mape: Optional[float],
    units: str = "",
) -> str:
    h = hist.dropna().astype(float)
    if len(h) == 0:
        return "Forecast explanation unavailable because the historical series is empty."

    last_val = float(h.iloc[-1])
    recent = h.tail(5)
    direction = infer_direction(recent)

    end_forecast = float(fc["forecast"].iloc[-1])
    lo = float(fc["lo"].iloc[-1])
    hi = float(fc["hi"].iloc[-1])

    if abs(end_forecast) > 1e-9:
        band = (hi - lo) / abs(end_forecast)
    else:
        band = float("nan")

    uncertainty_word = "tight" if (not is_nan(band) and band < 0.25) else "wide"

    delta = (end_forecast - last_val) / (abs(last_val) + 1e-9)

    parts = []
    parts.append(f"Scenario: {scenario_name}.")
    parts.append(model_note)

    parts.append(
        f"Recent history is {direction}. "
        f"The end-of-horizon estimate is {units}{fmt_number(end_forecast, 2)} "
        f"with a {uncertainty_word} uncertainty range "
        f"({units}{fmt_number(lo, 2)} to {units}{fmt_number(hi, 2)})."
    )

    parts.append(
        f"Relative to the most recent actual value ({units}{fmt_number(last_val, 2)}), "
        f"the forecast implies approximately {fmt_pct(delta, 1)} movement by the horizon."
    )

    if mape is not None and not is_nan(mape):
        parts.append(f"Backtest error (MAPE, last few points): {fmt_pct(mape, 1)}. Use this as a realism check, not a guarantee.")

    return " ".join(parts)


def insight_bullets_for_fundamentals(
    df: pd.DataFrame,
    time_col: str,
    revenue_col: Optional[str],
    margin_col: Optional[str],
    op_income_col: Optional[str],
) -> List[str]:
    bullets = []
    d = df.sort_values(time_col).copy()

    if revenue_col and revenue_col in d.columns:
        rev = d[revenue_col].astype(float)
        direction = infer_direction(rev.tail(5))
        bullets.append(f"Revenue is {direction} recently. This anchors the scale of the business and planning capacity.")

        yoy = compute_yoy(rev)
        if yoy.dropna().shape[0] >= 2:
            latest_yoy = float(yoy.dropna().iloc[-1])
            bullets.append(f"Latest revenue growth is {fmt_pct(latest_yoy, 1)} year-over-year. This is the clearest acceleration/deceleration signal.")

    if op_income_col and revenue_col and op_income_col in d.columns and revenue_col in d.columns:
        margin = compute_margin(d[op_income_col], d[revenue_col])
        if margin.dropna().shape[0] >= 3:
            latest_margin = float(margin.dropna().iloc[-1])
            bullets.append(f"Operating margin is {fmt_pct(latest_margin, 1)}. Margin quality distinguishes efficient growth from expensive growth.")

    if margin_col and margin_col in d.columns:
        m = d[margin_col].astype(float)
        bullets.append(f"Margin trend is {infer_direction(m.tail(5))}. This is an early warning signal for cost pressure or improved efficiency.")

    if len(bullets) == 0:
        bullets.append("Add fundamentals columns like revenue and operating income to unlock margin and growth narratives.")

    return bullets


# ============================================================
# 8) Visual building blocks
# ============================================================

def line_chart(df: pd.DataFrame, x: str, y: str, title: str, yaxis_title: Optional[str] = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=yaxis_title if yaxis_title else y,
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, yaxis_format: Optional[str] = None) -> go.Figure:
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    if yaxis_format:
        fig.update_yaxes(tickformat=yaxis_format)
    return fig


def forecast_chart(hist_df: pd.DataFrame, x: str, y: str, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df[x], y=hist_df[y], mode="lines+markers", name="Actual"))

    fig.add_trace(go.Scatter(x=fc["t"], y=fc["forecast"], mode="lines+markers", name="Forecast"))

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([fc["t"].values, fc["t"].values[::-1]]),
            y=np.concatenate([fc["hi"].values, fc["lo"].values[::-1]]),
            fill="toself",
            line=dict(width=0),
            name="Uncertainty band",
            showlegend=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def dual_axis_revenue_margin(df: pd.DataFrame, x: str, revenue_col: str, margin_series: pd.Series, title: str) -> go.Figure:
    tmp = df[[x, revenue_col]].copy()
    tmp["margin"] = margin_series.values

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tmp[x], y=tmp[revenue_col], name="Revenue"))
    fig.add_trace(go.Scatter(x=tmp[x], y=tmp["margin"], name="Margin", mode="lines+markers", yaxis="y2"))

    fig.update_layout(
        title=title,
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Margin", overlaying="y", side="right", tickformat=".0%"),
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ============================================================
# 9) App navigation and controls
# ============================================================

df_raw = None
try:
    df_raw = load_master_csv(DEFAULT_CSV_PATH)
    st.success("Data loaded from master.csv")
except Exception as e:
    st.error("Could not load master.csv. Check that the file is in the repo root and is a valid CSV.")
    st.exception(e)
    st.stop()

colmap = detect_column_map(df_raw)

df_std, time_col = standardize_time(df_raw, colmap.time_col)
colmap = detect_column_map(df_std)  # refresh after time standardization

st.sidebar.header("Controls")

page = st.sidebar.radio(
    "Page",
    [
        "Executive summary",
        "Fundamentals",
        "Market performance",
        "Macro and indicators",
        "Forecasts and scenarios",
        "Data quality and dictionary",
        "Raw data view",
    ],
    index=0,
)

st.sidebar.subheader("Dataset format")
if colmap.metric_col and colmap.value_col:
    st.sidebar.write("Detected long format: metric/value columns present.")
else:
    st.sidebar.write("Detected wide format: numeric metric columns present.")


# ============================================================
# 10) Prepare analysis dataset (handles long and wide formats)
# ============================================================

def build_analysis_frame(df: pd.DataFrame, cm: ColumnMap, tcol: str) -> pd.DataFrame:
    """
    Returns a normalized analysis frame.
    If long format: keep as-is (time, entity, metric, value).
    If wide format: melt into long.
    """
    d = df.copy()

    # Ensure time col exists and is clean
    if tcol not in d.columns:
        d[tcol] = df[cm.time_col]

    if cm.metric_col and cm.value_col:
        keep = [tcol, cm.metric_col, cm.value_col]
        if cm.entity_col and cm.entity_col in d.columns:
            keep.insert(1, cm.entity_col)
        out = d[keep].copy()
        out.rename(columns={cm.metric_col: "metric", cm.value_col: "value"}, inplace=True)
        if cm.entity_col:
            out.rename(columns={cm.entity_col: "entity"}, inplace=True)
        else:
            out["entity"] = "ALL"
        out["metric"] = out["metric"].astype(str)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out

    # Wide format: melt numeric metric columns
    metric_cols = cm.wide_metric_cols
    if len(metric_cols) == 0:
        # As fallback, include all except time/entity
        exclude = {tcol}
        if cm.entity_col:
            exclude.add(cm.entity_col)
        metric_cols = [c for c in d.columns if c not in exclude]

    keep_cols = [tcol] + ([cm.entity_col] if cm.entity_col else [])
    long_df = d[keep_cols + metric_cols].copy()

    long_out = long_df.melt(
        id_vars=keep_cols,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    if cm.entity_col:
        long_out.rename(columns={cm.entity_col: "entity"}, inplace=True)
    else:
        long_out["entity"] = "ALL"

    long_out["metric"] = long_out["metric"].astype(str)
    long_out["value"] = pd.to_numeric(long_out["value"], errors="coerce")
    return long_out


analysis_long = build_analysis_frame(df_std, colmap, time_col)

# Entity filter (if meaningful)
entities = sorted(analysis_long["entity"].dropna().unique().tolist())
if len(entities) > 1:
    selected_entity = st.sidebar.selectbox("Entity", entities, index=0)
else:
    selected_entity = entities[0] if len(entities) else "ALL"

# Metric selection
all_metrics = sorted(analysis_long["metric"].dropna().unique().tolist())
selected_metric = st.sidebar.selectbox("Metric", all_metrics, index=0)

# Time filter
time_values = analysis_long[time_col].dropna().unique()
time_values_sorted = sorted(time_values.tolist())
if len(time_values_sorted) >= 2:
    tmin = time_values_sorted[0]
    tmax = time_values_sorted[-1]
    t_start, t_end = st.sidebar.select_slider("Time range", options=time_values_sorted, value=(tmin, tmax))
else:
    t_start, t_end = (time_values_sorted[0], time_values_sorted[0]) if len(time_values_sorted) else (None, None)

filtered = analysis_long.copy()
filtered = filtered[filtered["entity"] == selected_entity]
filtered = filtered[(filtered[time_col] >= t_start) & (filtered[time_col] <= t_end)]
filtered = filtered.sort_values(time_col)

metric_df = filtered[filtered["metric"] == selected_metric].copy()
metric_df = metric_df[[time_col, "value"]].dropna().sort_values(time_col)

# Provide a stable view frame
wide_for_entity = None
try:
    wide_for_entity = filtered.pivot_table(index=time_col, columns="metric", values="value", aggfunc="mean").reset_index()
except Exception:
    wide_for_entity = None


# ============================================================
# 11) Executive summary page
# ============================================================

if page == "Executive summary":
    st.header("Executive summary")

    # KPI cards based on canonical columns if present in the wide view
    # If we have a wide frame for the entity, we can compute revenue/margin/returns.
    if wide_for_entity is not None and len(wide_for_entity) >= 3:
        w = wide_for_entity.sort_values(time_col).copy()
        cols = list(w.columns)

        revenue_col = colmap.revenue_col if colmap.revenue_col in cols else find_best_match(cols, ["revenue"])
        op_income_col = colmap.op_income_col if colmap.op_income_col in cols else find_best_match(cols, ["operating_income", "op_income", "ebit"])
        close_col = colmap.close_col if colmap.close_col in cols else find_best_match(cols, ["close", "price", "adj_close"])

        # Compute derived series
        if revenue_col:
            w["revenue_yoy_calc"] = compute_yoy(w[revenue_col])
            w["revenue_index_calc"] = compute_index(w[revenue_col])
        if close_col:
            w["return_calc"] = compute_annual_return(w[close_col])
            w["stock_index_calc"] = compute_index(w[close_col])
        if revenue_col and op_income_col:
            w["op_margin_calc"] = compute_margin(w[op_income_col], w[revenue_col])

        latest = w.dropna(subset=[time_col]).iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        if revenue_col:
            c1.metric("Revenue (latest)", fmt_money(latest.get(revenue_col, np.nan)))
            c2.metric("Revenue YoY (latest)", fmt_pct(latest.get("revenue_yoy_calc", np.nan), 1))
        else:
            c1.metric("Revenue (latest)", "—")
            c2.metric("Revenue YoY (latest)", "—")

        if revenue_col and op_income_col:
            c3.metric("Operating margin (latest)", fmt_pct(latest.get("op_margin_calc", np.nan), 1))
        else:
            c3.metric("Operating margin (latest)", "—")

        if close_col:
            vol = w["return_calc"].dropna().std()
            c4.metric("Return volatility", fmt_pct(vol, 1) if not is_nan(vol) else "—")
        else:
            c4.metric("Return volatility", "—")

        st.subheader("What changed and why it matters")

        bullets = insight_bullets_for_fundamentals(
            w,
            time_col=time_col,
            revenue_col=revenue_col,
            margin_col="op_margin_calc" if "op_margin_calc" in w.columns else None,
            op_income_col=op_income_col,
        )
        for b in bullets:
            st.write(f"- {b}")

        st.divider()

        # Indexed growth comparison if available
        if "revenue_index_calc" in w.columns and "stock_index_calc" in w.columns:
            idx = w[[time_col, "revenue_index_calc", "stock_index_calc"]].copy()
            idx_long = idx.melt(id_vars=time_col, var_name="series", value_name="index_value")
            fig = px.line(idx_long, x=time_col, y="index_value", color="series", markers=True,
                          title="Indexed growth: fundamentals versus market (base = 100)")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add columns like revenue and close (price) to enable indexed comparisons and market linkage narratives.")

    else:
        st.info("Executive summary works best when the master dataset includes wide-format fundamentals and/or prices for the selected entity.")

    st.divider()

    st.subheader("Primary metric view")
    if len(metric_df) >= 2:
        fig = line_chart(metric_df, x=time_col, y="value", title=f"{selected_metric} over time")
        st.plotly_chart(fig, use_container_width=True)

        yoy = compute_yoy(metric_df["value"])
        if yoy.dropna().shape[0] >= 2:
            metric_yoy = pd.DataFrame({time_col: metric_df[time_col], "yoy": yoy})
            fig2 = bar_chart(metric_yoy.dropna(), x=time_col, y="yoy", title=f"{selected_metric} year-over-year change", yaxis_format=PCT_FORMAT)
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Not enough data points in the selected range to plot this metric.")


# ============================================================
# 12) Fundamentals page
# ============================================================

elif page == "Fundamentals":
    st.header("Fundamentals")

    if wide_for_entity is None or len(wide_for_entity) < 3:
        st.info("Fundamentals page requires wide metrics or a pivotable dataset. Try another entity or expand the dataset.")
        st.stop()

    w = wide_for_entity.sort_values(time_col).copy()
    cols = list(w.columns)

    # Choose fundamentals metrics interactively from available numeric columns
    numeric_cols = [c for c in cols if c != time_col and pd.api.types.is_numeric_dtype(w[c])]
    if len(numeric_cols) == 0:
        st.info("No numeric fundamentals columns detected for this entity.")
        st.stop()

    left, right = st.columns([2, 1])

    with right:
        st.subheader("Choose metrics")
        primary = st.selectbox("Primary fundamental metric", numeric_cols, index=0)
        secondary = st.selectbox("Secondary metric (optional)", ["None"] + numeric_cols, index=0)

        show_yoy = st.checkbox("Show YoY bars", value=True)
        show_indexed = st.checkbox("Show indexed (base=100)", value=False)

    with left:
        st.subheader("Trend")
        fig = line_chart(w, x=time_col, y=primary, title=f"{primary} trend")
        st.plotly_chart(fig, use_container_width=True)

        if show_yoy:
            yoy = compute_yoy(w[primary])
            tmp = pd.DataFrame({time_col: w[time_col], "yoy": yoy}).dropna()
            if len(tmp) >= 2:
                fig2 = bar_chart(tmp, x=time_col, y="yoy", title=f"{primary} YoY change", yaxis_format=PCT_FORMAT)
                st.plotly_chart(fig2, use_container_width=True)

        if show_indexed:
            idx = compute_index(w[primary])
            tmp = pd.DataFrame({time_col: w[time_col], "index": idx}).dropna()
            if len(tmp) >= 2:
                fig3 = line_chart(tmp, x=time_col, y="index", title=f"{primary} indexed (base=100)")
                st.plotly_chart(fig3, use_container_width=True)

        if secondary != "None":
            st.subheader("Primary versus secondary")
            tmp = w[[time_col, primary, secondary]].dropna()
            if len(tmp) >= 3:
                fig4 = px.line(tmp.melt(id_vars=time_col, var_name="metric", value_name="value"),
                               x=time_col, y="value", color="metric", markers=True,
                               title="Comparison")
                fig4.update_layout(legend_title_text="")
                st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Plain-English interpretation")

    series = w[primary].astype(float).dropna()
    if len(series) >= 4:
        direction = infer_direction(series.tail(5))
        latest = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) >= 2 else np.nan
        delta = (latest - prev) / (abs(prev) + 1e-9) if not is_nan(prev) else np.nan

        interpretation = (
            f"{primary} is {direction} in the most recent periods. "
            f"The latest value is {fmt_number(latest, 2)}. "
        )
        if not is_nan(delta):
            interpretation += f"That is a {fmt_pct(delta, 1)} change from the prior period. "

        interpretation += (
            "In decision terms, this metric should connect to budget allocation, capacity planning, or performance targets. "
            "A strong portfolio app shows not only the trend, but what actions the business should consider if the trend persists."
        )

        st.write(interpretation)
    else:
        st.write("Not enough history to generate a meaningful interpretation.")


# ============================================================
# 13) Market performance page
# ============================================================

elif page == "Market performance":
    st.header("Market performance")

    if wide_for_entity is None or len(wide_for_entity) < 3:
        st.info("Market performance requires price-like metrics (close/price) for the selected entity.")
        st.stop()

    w = wide_for_entity.sort_values(time_col).copy()
    cols = list(w.columns)

    price_col = find_best_match(cols, ["close", "adj_close", "price", "close_price", "avg_price"])
    if not price_col:
        st.info("No price-like column detected. Add a close/price column to enable returns, volatility, and market storytelling.")
        st.stop()

    w["return_calc"] = compute_annual_return(w[price_col])
    w["price_index"] = compute_index(w[price_col])

    c1, c2, c3 = st.columns(3)
    last_price = w[price_col].dropna().iloc[-1] if w[price_col].dropna().shape[0] else np.nan
    last_ret = w["return_calc"].dropna().iloc[-1] if w["return_calc"].dropna().shape[0] else np.nan
    vol = w["return_calc"].dropna().std()

    c1.metric("Latest price", fmt_number(last_price, 2) if not is_nan(last_price) else "—")
    c2.metric("Latest return", fmt_pct(last_ret, 1))
    c3.metric("Volatility", fmt_pct(vol, 1) if not is_nan(vol) else "—")

    fig = line_chart(w.dropna(subset=[price_col]), x=time_col, y=price_col, title=f"{price_col} trend")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = bar_chart(w.dropna(subset=["return_calc"]), x=time_col, y="return_calc", title="Returns by period", yaxis_format=PCT_FORMAT)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Market versus fundamentals linkage")

    # Attempt to use revenue if present
    revenue_col = find_best_match(cols, ["revenue", "total_revenue", "sales"])
    if revenue_col:
        tmp = w[[time_col, revenue_col, price_col]].dropna()
        if len(tmp) >= 6:
            corr = tmp[revenue_col].corr(tmp[price_col])
            st.write(relationship_statement(float(corr)))

            fig3 = px.scatter(tmp, x=revenue_col, y=price_col, trendline="ols",
                              title=f"{revenue_col} versus {price_col} (diagnostic relationship)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough overlapping data points to analyze linkage.")
    else:
        st.info("Add a revenue (or sales) column to connect market pricing to fundamentals.")


# ============================================================
# 14) Macro and indicators page
# ============================================================

elif page == "Macro and indicators":
    st.header("Macro and indicators")

    st.write(
        "This section is designed to connect macro indicators to performance. "
        "In a master dataset, macro metrics typically include inflation, interest rates, unemployment, GDP growth, "
        "and other forward-looking signals."
    )

    if wide_for_entity is None or len(wide_for_entity) < 3:
        st.info("Macro page requires multiple numeric metrics to compare and describe potential drivers.")
        st.stop()

    w = wide_for_entity.sort_values(time_col).copy()
    cols = [c for c in w.columns if c != time_col and pd.api.types.is_numeric_dtype(w[c])]

    if len(cols) < 2:
        st.info("Need at least two numeric metrics to compare indicators.")
        st.stop()

    primary = st.selectbox("Primary metric (what you want to explain)", cols, index=0)
    candidates = [c for c in cols if c != primary]
    drivers = st.multiselect("Potential drivers (macros/inputs)", candidates, default=candidates[: min(3, len(candidates))])

    if len(drivers) == 0:
        st.info("Select at least one driver metric.")
        st.stop()

    tmp = w[[time_col, primary] + drivers].dropna()
    if len(tmp) < 6:
        st.warning("Not enough overlapping data to estimate relationships.")
        st.stop()

    st.subheader("Correlation scan")
    corrs = []
    for dcol in drivers:
        corrs.append((dcol, float(tmp[primary].corr(tmp[dcol]))))
    corrs_df = pd.DataFrame(corrs, columns=["driver", "correlation"]).sort_values("correlation", ascending=False)

    st.dataframe(corrs_df, use_container_width=True)

    st.subheader("Driver visualization")
    plot_df = tmp[[time_col, primary] + drivers].copy()
    plot_long = plot_df.melt(id_vars=time_col, var_name="metric", value_name="value")
    fig = px.line(plot_long, x=time_col, y="value", color="metric", markers=True,
                  title="Primary metric and selected drivers (raw scale)")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Plain-English driver story")

    best_driver = corrs_df.iloc[0]["driver"]
    best_corr = float(corrs_df.iloc[0]["correlation"])
    story = (
        f"For the selected window, the strongest simple relationship is between {primary} and {best_driver} "
        f"(correlation {best_corr:.2f}). "
        f"This does not prove causation, but it helps prioritize which drivers to investigate first. "
        "A strong portfolio narrative explains plausible mechanisms (for example, rates affecting discount factors, "
        "inflation affecting costs, or GDP affecting demand) and then validates them with deeper analysis."
    )
    st.write(story)


# ============================================================
# 15) Forecasts and scenarios page
# ============================================================

elif page == "Forecasts and scenarios":
    st.header("Forecasts and scenarios")

    if wide_for_entity is None or len(wide_for_entity) < MIN_ROWS_FOR_FORECAST:
        st.info("Forecasting requires more history. Expand your dataset or widen the time range.")
        st.stop()

    w = wide_for_entity.sort_values(time_col).copy()
    num_cols = [c for c in w.columns if c != time_col and pd.api.types.is_numeric_dtype(w[c])]
    if len(num_cols) == 0:
        st.info("No numeric columns available to forecast.")
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_right:
        metric = st.selectbox("Metric to forecast", num_cols, index=0)
        horizon = st.slider("Horizon", 1, 10, DEFAULT_FORECAST_HORIZON)

        scenario = st.selectbox("Scenario", ["Baseline", "Optimistic", "Conservative"], index=0)

        uplift = 0.0
        if scenario == "Optimistic":
            uplift = st.slider("Optimistic uplift (annual)", 0.00, 0.25, 0.05, 0.01)
        elif scenario == "Conservative":
            uplift = -st.slider("Conservative drag (annual)", 0.00, 0.25, 0.05, 0.01)

        log_pref = st.checkbox("Use log-scale trend when possible", value=True)

    series_df = w[[time_col, metric]].dropna()
    if len(series_df) < MIN_ROWS_FOR_FORECAST:
        st.warning("Not enough usable points for forecasting after removing missing values.")
        st.stop()

    # Forecast
    fr = trend_forecast(series_df[time_col], series_df[metric], horizon=horizon, log_scale=log_pref)
    if fr is None:
        st.warning("Forecast model could not be fit. Check time column format and ensure numeric values.")
        st.stop()

    fc = apply_scenario_adjustment(fr.forecast_df, uplift)

    with col_left:
        fig = forecast_chart(series_df, x=time_col, y=metric, fc=fc, title=f"{metric} forecast ({scenario})")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast explanation")
    units = ""
    if "revenue" in metric.lower() or "income" in metric.lower() or "asset" in metric.lower():
        units = "$"

    explanation = forecast_explanation(
        metric_label=metric,
        hist=series_df[metric],
        fc=fc,
        scenario_name=scenario,
        model_note=fr.model_note,
        mape=fr.mape_backtest,
        units=units,
    )
    st.write(explanation)

    st.divider()
    st.subheader("Forecast table")
    show = fc.copy()
    show.rename(columns={"t": time_col}, inplace=True)

    # Friendly formatting
    show_out = show.copy()
    if units == "$":
        for c in ["forecast", "lo", "hi"]:
            show_out[c] = show_out[c].apply(lambda v: fmt_money(v))
    else:
        for c in ["forecast", "lo", "hi"]:
            show_out[c] = show_out[c].apply(lambda v: fmt_number(v, 2))

    st.dataframe(show_out, use_container_width=True)

    st.divider()
    st.subheader("What a hiring manager looks for here")

    st.write(
        "Strong signal: your forecast is explainable, your uncertainty is explicit, and your scenario assumptions are stated clearly. "
        "Weak signal: a black-box model with no interpretation or a forecast without a plausible mechanism. "
        "This app uses a simple trend model on purpose so the narrative remains credible and reviewable."
    )


# ============================================================
# 16) Data quality and dictionary page
# ============================================================

elif page == "Data quality and dictionary":
    st.header("Data quality and dictionary")

    st.subheader("Quick health checks")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_std):,}")
    c2.metric("Columns", f"{df_std.shape[1]:,}")

    missing_rate = float(df_std.isna().mean().mean()) if df_std.size else np.nan
    c3.metric("Average missing rate", fmt_pct(missing_rate, 1))

    numeric_cols = [c for c in df_std.columns if pd.api.types.is_numeric_dtype(df_std[c])]
    c4.metric("Numeric columns", f"{len(numeric_cols):,}")

    st.divider()

    st.subheader("Missingness by column (top)")
    miss = df_std.isna().mean().sort_values(ascending=False)
    miss_df = pd.DataFrame({"column": miss.index, "missing_rate": miss.values})
    st.dataframe(miss_df.head(30), use_container_width=True)

    st.divider()

    st.subheader("Data dictionary (auto-generated, customize for portfolio)")
    defs = {
        "revenue": "Total revenue for the period. Scale signal used for planning and growth evaluation.",
        "operating_income": "Operating profit. Helps separate business performance from financing/taxes.",
        "total_assets": "Balance sheet scale. Context for investment base and capacity.",
        "close": "End-of-period price. Reflects expectations, fundamentals, macro, and valuation multiples.",
        "annual_return": "Year-over-year return. Used to quantify performance and volatility.",
        "operating_margin": "Operating income divided by revenue. Quality and efficiency of growth.",
        "revenue_yoy": "Revenue growth rate versus the prior period.",
        "stock_yoy": "Price change versus the prior period.",
        "index": "Series normalized to a base value for comparability."
    }

    dd_rows = []
    for c in df_std.columns:
        lc = c.lower()
        definition = defs.get(lc, "Add a definition: what it measures, why it matters, and units.")
        dtype = str(df_std[c].dtype)
        example = ""
        if pd.api.types.is_numeric_dtype(df_std[c]):
            example = fmt_number(df_std[c].dropna().iloc[0], 2) if df_std[c].dropna().shape[0] else "—"
        else:
            example = str(df_std[c].dropna().iloc[0])[:50] if df_std[c].dropna().shape[0] else "—"

        dd_rows.append(
            {
                "field": c,
                "dtype": dtype,
                "definition": definition,
                "example": example,
            }
        )

    dd = pd.DataFrame(dd_rows)
    st.dataframe(dd, use_container_width=True)

    st.divider()

    st.subheader("Portfolio guidance")
    st.write(
        "To meet hiring-manager expectations, add a short section in your README describing: "
        "1) where each metric comes from (source and refresh cadence), "
        "2) how you handle missing values and revisions, "
        "3) what quality checks you run, and "
        "4) how the app ties metrics to real decisions."
    )


# ============================================================
# 17) Raw data view page
# ============================================================

elif page == "Raw data view":
    st.header("Raw data view")

    st.write(
        "This is a transparent view of the underlying dataset used to generate charts, metrics, and narratives. "
        "Transparency is a strong signal in financial analytics projects."
    )

    st.subheader("Preview")
    st.dataframe(df_std.head(50), use_container_width=True)

    st.subheader("Schema")
    schema = pd.DataFrame({"column": df_std.columns, "dtype": [str(df_std[c].dtype) for c in df_std.columns]})
    st.dataframe(schema, use_container_width=True)

    st.subheader("Download filtered long-format")
    downloadable = filtered.copy()
    csv_bytes = downloadable.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="filtered_long.csv", mime="text/csv")


# ============================================================
# 18) Footer: app notes
# ============================================================

st.divider()
st.caption(
    "Notes: This app intentionally emphasizes explainable models and decision narratives. "
    "For production, you would add metric governance, automated refresh, and monitoring."
)
