import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Master Finance Analytics (Portfolio App)",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning / typing
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df

def fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12: return f"{sign}${x/1e12:.2f}T"
    if x >= 1e9:  return f"{sign}${x/1e9:.2f}B"
    if x >= 1e6:  return f"{sign}${x/1e6:.2f}M"
    return f"{sign}${x:,.0f}"

def fmt_pct(x: float, digits=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100*float(x):.{digits}f}%"

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    return float(tmp.corr().iloc[0, 1])

def cagr(first: float, last: float, years: int) -> float:
    if years <= 0 or first is None or last is None:
        return np.nan
    if first <= 0 or last <= 0:
        return np.nan
    return (last / first) ** (1 / years) - 1

def trend_forecast(y: pd.Series, x_years: pd.Series, horizon: int = 5, log_scale: bool = True, n_boot: int = 300):
    """
    Simple, explainable forecast:
    - Fits linear regression on year vs log(y) (default) to model long-term growth
    - Produces uncertainty via bootstrap of residuals (honest + transparent)
    """
    df_fit = pd.DataFrame({"year": x_years, "y": y}).dropna()
    if len(df_fit) < 5:
        return None

    X = df_fit["year"].values.reshape(-1, 1)
    yv = df_fit["y"].values.astype(float)

    if log_scale:
        if np.any(yv <= 0):
            log_scale = False
        else:
            y_model = np.log(yv)
    else:
        y_model = yv

    model = LinearRegression()
    model.fit(X, y_model)
    y_hat = model.predict(X)
    resid = y_model - y_hat

    last_year = int(df_fit["year"].max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1)
    Xf = future_years.reshape(-1, 1)
    pred = model.predict(Xf)

    # Bootstrap intervals
    rng = np.random.default_rng(42)
    sims = []
    if len(resid) >= 5:
        for _ in range(n_boot):
            sampled = rng.choice(resid, size=len(future_years), replace=True)
            sims.append(pred + sampled)
        sims = np.vstack(sims)
        lo = np.percentile(sims, 10, axis=0)
        hi = np.percentile(sims, 90, axis=0)
    else:
        lo, hi = pred, pred

    if log_scale:
        pred_y = np.exp(pred)
        lo_y = np.exp(lo)
        hi_y = np.exp(hi)
    else:
        pred_y = pred
        lo_y = lo
        hi_y = hi

    out = pd.DataFrame({
        "year": future_years,
        "forecast": pred_y,
        "lo": lo_y,
        "hi": hi_y
    })
    return out

def build_story(df: pd.DataFrame) -> dict:
    """
    Turns metrics into a hiring-manager-friendly narrative.
    """
    df = df.sort_values("year").copy()

    latest = df.iloc[-1]
    first = df.iloc[0]
    years_span = int(latest["year"] - first["year"])

    # Core KPIs
    rev_cagr = cagr(first.get("revenue", np.nan), latest.get("revenue", np.nan), years_span)
    last_rev_yoy = latest.get("revenue_yoy", np.nan)
    last_margin = latest.get("operating_margin", np.nan)

    ann_ret = df["annual_return"].dropna()
    vol = float(ann_ret.std()) if len(ann_ret) >= 3 else np.nan

    corr_rev_stock = safe_corr(df.get("revenue", pd.Series(dtype=float)), df.get("close", pd.Series(dtype=float)))

    # Identify recent direction
    recent = df.tail(4)
    rev_trend = "up" if recent["revenue"].dropna().diff().mean() > 0 else "down"
    margin_trend = "up" if recent["operating_margin"].dropna().diff().mean() > 0 else "down"

    # Market vs fundamentals
    if not np.isnan(corr_rev_stock):
        if corr_rev_stock >= 0.6:
            linkage = "The stock and revenue have moved together strongly, which usually means fundamentals are a major driver of market perception."
        elif corr_rev_stock <= 0.0:
            linkage = "Revenue and the stock have not moved together reliably, which suggests valuation, sentiment, or macro conditions may be dominating the market story."
        else:
            linkage = "Revenue and the stock show a moderate relationship—fundamentals matter, but market expectations and macro factors also play a big role."
    else:
        linkage = "There isn’t enough data to reliably quantify the revenue-to-stock relationship, so the app focuses more on directional trends."

    # Narrative (plain English)
    story = {
        "headline": "Fundamentals + Market: what changed, and what it likely means",
        "bullets": [
            f"Revenue trend is **{rev_trend}** recently, and long-run growth is about **{fmt_pct(rev_cagr, 1)} CAGR** over the period shown." if not np.isnan(rev_cagr)
            else "Revenue growth is visible, but CAGR can’t be computed reliably with the current values.",
            f"Latest revenue growth is **{fmt_pct(last_rev_yoy, 1)} YoY**." if not np.isnan(last_rev_yoy)
            else "Latest revenue YoY is unavailable in the dataset.",
            f"Operating margin is **{fmt_pct(last_margin, 1)}**, trending **{margin_trend}** recently (profitability quality signal)." if not np.isnan(last_margin)
            else "Operating margin is unavailable in the dataset.",
            f"Return volatility is **{fmt_pct(vol, 1)}** (annualized, based on annual returns)." if not np.isnan(vol)
            else "Not enough return history to compute volatility.",
            linkage
        ],
        "corr_rev_stock": corr_rev_stock,
        "rev_cagr": rev_cagr,
        "vol": vol
    }
    return story

def explain_forecast(metric_name: str, history: pd.Series, forecast_df: pd.DataFrame, scenario_label: str) -> str:
    """
    A human-readable explanation for forecasts that doesn't sound like boilerplate.
    """
    last = history.dropna().iloc[-1] if history.dropna().shape[0] else np.nan
    recent = history.dropna().tail(4)
    recent_slope = recent.diff().mean() if len(recent) >= 3 else np.nan

    base = forecast_df["forecast"].iloc[-1]
    lo = forecast_df["lo"].iloc[-1]
    hi = forecast_df["hi"].iloc[-1]

    direction = "increase" if (not np.isnan(recent_slope) and recent_slope > 0) else "soften"
    uncertainty = "tight" if (hi - lo) / max(1e-9, abs(base)) < 0.25 else "wide"

    # Metric-specific language
    if metric_name in ["revenue", "operating_income", "total_assets"]:
        metric_phrase = "business fundamentals"
    elif metric_name in ["close", "avg_price", "open", "high", "low"]:
        metric_phrase = "market pricing"
    else:
        metric_phrase = "the metric"

    def pretty(v):
        if metric_name in ["revenue", "operating_income", "total_assets"]:
            return fmt_money(v)
        if metric_name in ["annual_return", "operating_margin", "revenue_yoy", "stock_yoy"]:
            return fmt_pct(v, 1)
        return f"{v:,.2f}"

    msg = (
        f"**{scenario_label} scenario:** Based on the historical trend, the model expects {metric_phrase} "
        f"to **{direction}** over the next few years. The end-of-horizon estimate is **{pretty(base)}**, "
        f"with a {uncertainty} range (**{pretty(lo)}** to **{pretty(hi)}**). "
        f"This range widens when recent history is choppier or when the metric has larger year-to-year swings."
    )

    # Add one more sentence that feels like an analyst
    if not np.isnan(last) and not np.isnan(base):
        change = (base - last) / (abs(last) + 1e-9)
        msg += f" Relative to the most recent actual value (**{pretty(last)}**), that implies about **{fmt_pct(change, 1)}** movement by the forecast horizon."

    return msg

# -----------------------------
# Load data
# -----------------------------
df = load_data("master.csv")

# Validate expected columns lightly (don’t hard fail)
required_hint = ["year", "revenue", "operating_income", "total_assets", "close"]
missing = [c for c in required_hint if c not in df.columns]
if missing:
    st.warning(f"Your dataset is missing some typical fields used in this app: {missing}. The app will still run with what’s available.")

df = df.sort_values("year")
story = build_story(df)

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("📌 Master Finance App")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Summary", "Fundamentals", "Market Performance", "Forecasts & Scenarios", "Data Dictionary"],
    index=0
)

st.sidebar.caption("Built to showcase: financial analytics, product thinking, and explainable insights.")

# -----------------------------
# Executive Summary
# -----------------------------
if page == "Executive Summary":
    st.title("📈 Master Finance Analytics — Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    # KPI cards
    rev_cagr = story["rev_cagr"]
    vol = story["vol"]
    last = df.iloc[-1]

    c1.metric("Revenue (latest)", fmt_money(last.get("revenue", np.nan)))
    c2.metric("Revenue CAGR", fmt_pct(rev_cagr, 1) if not np.isnan(rev_cagr) else "—")
    c3.metric("Operating Margin (latest)", fmt_pct(last.get("operating_margin", np.nan), 1))
    c4.metric("Return Volatility", fmt_pct(vol, 1) if not np.isnan(vol) else "—")

    st.subheader(story["headline"])
    for b in story["bullets"]:
        st.write("• " + b)

    st.divider()

    # Indexed comparison
    cols = []
    if "revenue_index" in df.columns:
        cols.append("revenue_index")
    if "stock_index" in df.columns:
        cols.append("stock_index")

    if cols:
        plot_df = df[["year"] + cols].copy()
        plot_long = plot_df.melt(id_vars="year", var_name="index_type", value_name="value")
        fig = px.line(plot_long, x="year", y="value", color="index_type", markers=True,
                      title="Indexed Growth: Fundamentals vs Market (Base = 100)")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add `revenue_index` and/or `stock_index` columns to show indexed comparisons (base=100).")

# -----------------------------
# Fundamentals
# -----------------------------
elif page == "Fundamentals":
    st.title("🏦 Fundamentals")

    metric = st.selectbox(
        "Choose a fundamentals metric",
        [m for m in ["revenue", "operating_income", "total_assets", "operating_margin", "revenue_yoy"] if m in df.columns],
        index=0
    )

    left, right = st.columns([2, 1])

    with left:
        fig = px.line(df, x="year", y=metric, markers=True, title=f"{metric.replace('_',' ').title()} Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Interpretation")
        latest_val = df[metric].dropna().iloc[-1] if df[metric].dropna().shape[0] else np.nan
        prev_val = df[metric].dropna().iloc[-2] if df[metric].dropna().shape[0] >= 2 else np.nan

        if metric in ["revenue", "operating_income", "total_assets"]:
            st.write(f"Latest: **{fmt_money(latest_val)}**")
            if not np.isnan(prev_val):
                delta = (latest_val - prev_val) / (abs(prev_val) + 1e-9)
                st.write(f"Change vs prior year: **{fmt_pct(delta, 1)}**")
            st.write("Why it matters: this is a core scale/health signal that should map to strategy, capacity, and investment decisions.")
        elif metric in ["operating_margin"]:
            st.write(f"Latest: **{fmt_pct(latest_val, 1)}**")
            st.write("Why it matters: margin is the ‘quality’ of growth—strong revenue growth with collapsing margins is a very different story than efficient growth.")
        elif metric in ["revenue_yoy"]:
            st.write(f"Latest: **{fmt_pct(latest_val, 1)}**")
            st.write("Why it matters: YoY highlights acceleration or deceleration—useful for planning and expectation-setting.")

    st.divider()

    if "revenue" in df.columns and "operating_margin" in df.columns:
        tmp = df[["year", "revenue", "operating_margin"]].copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=tmp["year"], y=tmp["revenue"], name="Revenue"))
        fig2.add_trace(go.Scatter(x=tmp["year"], y=tmp["operating_margin"], name="Operating Margin", yaxis="y2", mode="lines+markers"))
        fig2.update_layout(
            title="Revenue (Scale) + Operating Margin (Quality)",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Operating Margin", overlaying="y", side="right", tickformat=".0%"),
            legend_title_text=""
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Market Performance
# -----------------------------
elif page == "Market Performance":
    st.title("💹 Market Performance")

    price_cols = [c for c in ["open", "high", "low", "close", "avg_price"] if c in df.columns]
    if price_cols:
        metric = st.selectbox("Choose a market metric", price_cols, index=price_cols.index("close") if "close" in price_cols else 0)

        fig = px.line(df, x="year", y=metric, markers=True, title=f"{metric.title()} Over Time")
        st.plotly_chart(fig, use_container_width=True)

    if "annual_return" in df.columns:
        fig2 = px.bar(df, x="year", y="annual_return", title="Annual Return (YoY)", labels={"annual_return": "Return"})
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # Fundamentals vs market relationship
    st.subheader("Fundamentals vs Market: linkage check")
    if "revenue" in df.columns and "close" in df.columns:
        tmp = df[["year", "revenue", "close"]].dropna()
        if len(tmp) >= 5:
            fig3 = px.scatter(tmp, x="revenue", y="close", trendline="ols",
                              title="Revenue vs Stock Close (Relationship, not causation)")
            fig3.update_layout(xaxis_title="Revenue", yaxis_title="Close")
            st.plotly_chart(fig3, use_container_width=True)

            corr = safe_corr(df["revenue"], df["close"])
            st.write(f"Correlation (Revenue, Close): **{corr:.2f}**" if not np.isnan(corr) else "Correlation: —")
            st.caption("This is a quick diagnostic. A hiring manager wants to see that you can separate correlation from causation and discuss what else might be driving price (rates, multiples, expectations).")
        else:
            st.info("Not enough rows with both revenue and close to estimate a relationship reliably.")

# -----------------------------
# Forecasts & Scenarios
# -----------------------------
elif page == "Forecasts & Scenarios":
    st.title("🔮 Forecasts & Scenarios (Explainable)")

    forecastable = [c for c in ["revenue", "operating_income", "total_assets", "close"] if c in df.columns]
    if not forecastable:
        st.info("Add at least one of these columns to enable forecasting: revenue, operating_income, total_assets, close.")
    else:
        metric = st.selectbox("Metric to forecast", forecastable, index=0)
        horizon = st.slider("Forecast horizon (years)", 1, 10, 5)

        # Scenario control: adjust growth after baseline forecast
        scenario = st.selectbox("Scenario", ["Baseline", "Optimistic", "Conservative"], index=0)
        adj = 0.00
        if scenario == "Optimistic":
            adj = st.slider("Optimistic uplift (annual growth add-on)", 0.00, 0.20, 0.05, 0.01)
        elif scenario == "Conservative":
            adj = -st.slider("Conservative drag (annual growth reduction)", 0.00, 0.20, 0.05, 0.01)

        hist = df.set_index("year")[metric].astype(float)
        fc = trend_forecast(hist, hist.index.to_series(), horizon=horizon, log_scale=True)

        if fc is None:
            st.warning("Not enough data points to forecast this metric (need ~5+).")
        else:
            # Apply scenario adjustment as compounded growth on forecast path
            if adj != 0:
                base0 = float(fc["forecast"].iloc[0])
                years_ahead = np.arange(1, len(fc) + 1)
                growth_factor = (1 + adj) ** years_ahead
                fc["forecast"] = fc["forecast"] * growth_factor
                fc["lo"] = fc["lo"] * growth_factor
                fc["hi"] = fc["hi"] * growth_factor

            # Plot
            plot_hist = df[["year", metric]].copy()
            plot_fc = fc.copy()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_hist["year"], y=plot_hist[metric], mode="lines+markers", name="Actual"))

            fig.add_trace(go.Scatter(x=plot_fc["year"], y=plot_fc["forecast"], mode="lines+markers", name="Forecast"))
            fig.add_trace(go.Scatter(
                x=np.concatenate([plot_fc["year"], plot_fc["year"][::-1]]),
                y=np.concatenate([plot_fc["hi"], plot_fc["lo"][::-1]]),
                fill="toself",
                name="Uncertainty band (10–90%)",
                line=dict(width=0),
                showlegend=True
            ))

            fig.update_layout(title=f"{metric.replace('_',' ').title()} — Forecast ({scenario})",
                              xaxis_title="Year",
                              yaxis_title=metric.replace("_", " ").title())
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast explanation")
            st.write(explain_forecast(metric, hist, fc, scenario))

            st.divider()
            st.subheader("Scenario table (copy into a write-up)")
            show = fc.copy()
            show["forecast"] = show["forecast"].astype(float)
            show["lo"] = show["lo"].astype(float)
            show["hi"] = show["hi"].astype(float)

            # Pretty formatting
            if metric in ["revenue", "operating_income", "total_assets"]:
                show_fmt = show.copy()
                for c in ["forecast", "lo", "hi"]:
                    show_fmt[c] = show_fmt[c].apply(fmt_money)
            else:
                show_fmt = show.round(2)

            st.dataframe(show_fmt, use_container_width=True)

# -----------------------------
# Data Dictionary
# -----------------------------
elif page == "Data Dictionary":
    st.title("📚 Data Dictionary (Make the app self-explanatory)")

    dd = []
    defs = {
        "revenue": "Total revenue for the period. Scale signal used for planning and growth evaluation.",
        "operating_income": "Operating profit. Helps separate real business performance from financing/taxes.",
        "total_assets": "Balance sheet scale. Useful for context on investment base and capacity.",
        "close": "End-of-period price. Market expectations + fundamentals + macro all show up here.",
        "annual_return": "Year-over-year return based on close (or provided). Used to quantify performance/volatility.",
        "operating_margin": "Operating income / revenue. Quality and efficiency of growth.",
        "revenue_yoy": "Revenue growth rate vs prior year.",
        "stock_yoy": "Stock YoY change vs prior year.",
        "revenue_index": "Revenue normalized to 100 at start (comparability).",
        "stock_index": "Stock normalized to 100 at start (comparability).",
        "avg_price": "Average price over period (if computed)."
    }

    for col in df.columns:
        dd.append({
            "field": col,
            "definition": defs.get(col, "Add a definition here (recommended for portfolio quality)."),
            "example_use": "Chart, KPI, or forecast input"
        })

    st.dataframe(pd.DataFrame(dd), use_container_width=True)

    st.caption(
        "Hiring-manager signal: a clean dictionary shows you think about stakeholder clarity, metric governance, and scale."
    )
