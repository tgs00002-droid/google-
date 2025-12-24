"""
================================================================================
TITAN FINANCIAL INTELLIGENCE FRAMEWORK (v2.0.4)
================================================================================
CONFIDENTIAL: For Internal Strategy Review Only
AUTHOR: Senior Financial Analyst
DATE: 2023-10-27

DESCRIPTION:
This framework provides a holistic view of Google's (Alphabet) financial health.
It utilizes a modular architecture to separate data ingestion, quantitative 
modeling, and user interface rendering.

ARCHITECTURE:
1. DataIngestionLayer: Robust CSV parsing and type enforcement.
2. QuantitativeEngine: Statistical computing (VaR, Monte Carlo, CAGR).
3. VisualLayer: Plotly-based interactive rendering.
4. DashboardController: Streamlit application logic.

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import datetime

# ==============================================================================
# SECTION 1: SYSTEM CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="GOOGL Institutional Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional styling: No emojis, minimal clutter, high contrast
st.markdown("""
<style>
    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Metric Card Styling */
    div.stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f4;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: DATA INGESTION LAYER
# ==============================================================================

class DataIngestionLayer:
    """
    Responsible for loading, validating, and cleaning financial datasets.
    Enforces strict typing to prevent downstream calculation errors.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def execute_pipeline(self):
        """Runs the full data loading pipeline."""
        self._load_csv()
        self._validate_schema()
        self._enrich_data()
        return self.data

    def _load_csv(self):
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            st.error(f"CRITICAL ERROR: Failed to load data repository. {e}")
            st.stop()

    def _validate_schema(self):
        """Ensures essential columns exist."""
        required_cols = ['year', 'revenue', 'operating_income', 'total_assets', 'close']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            st.error(f"SCHEMA MISMATCH: Missing columns {missing}")
            st.stop()

    def _enrich_data(self):
        """Adds derived financial metrics."""
        # Convert to numeric, forcing errors to NaN then filling
        cols = ['revenue', 'operating_income', 'total_assets', 'close', 'annual_return']
        for c in cols:
            self.data[c] = pd.to_numeric(self.data[c], errors='coerce')
        
        self.data.fillna(method='ffill', inplace=True)
        self.data['date'] = pd.to_datetime(self.data['year'], format='%Y')
        
        # Advanced Metrics
        self.data['asset_turnover'] = self.data['revenue'] / self.data['total_assets']
        self.data['operating_roa'] = self.data['operating_income'] / self.data['total_assets']
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))

# ==============================================================================
# SECTION 3: QUANTITATIVE ANALYTICS ENGINE
# ==============================================================================

class QuantitativeEngine:
    """
    Performs heavy statistical and financial lifting.
    Includes Risk Modeling and Forecasting.
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data

    def calculate_cagr(self, metric: str, periods: int) -> float:
        """Computes Compound Annual Growth Rate."""
        start_val = self.df[metric].iloc[-periods-1] if len(self.df) > periods else self.df[metric].iloc[0]
        end_val = self.df[metric].iloc[-1]
        return (end_val / start_val) ** (1 / periods) - 1

    def calculate_var(self, confidence_level=0.95):
        """Calculates Value at Risk (VaR) using the Variance-Covariance method."""
        mean = np.mean(self.df['annual_return'])
        std_dev = np.std(self.df['annual_return'])
        var_pct = norm.ppf(1 - confidence_level, mean, std_dev)
        return var_pct

    def monte_carlo_forecast(self, simulations=1000, horizon=5):
        """
        Projects future revenue using Geometric Brownian Motion.
        Used for stress-testing growth assumptions.
        """
        last_rev = self.df['revenue'].iloc[-1]
        rev_growth = self.df['revenue'].pct_change().dropna()
        mu = rev_growth.mean()
        sigma = rev_growth.std()

        paths = np.zeros((horizon, simulations))
        paths[0] = last_rev

        for t in range(1, horizon):
            shock = np.random.normal(mu, sigma, simulations)
            paths[t] = paths[t-1] * (1 + shock)
        
        return paths

    def generate_linear_forecast(self, metric='revenue', years=3):
        """OLS Regression for deterministic trend analysis."""
        X = self.df['year'].values.reshape(-1, 1)
        y = self.df[metric].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_years = np.array(range(self.df['year'].max() + 1, self.df['year'].max() + 1 + years)).reshape(-1, 1)
        forecast = model.predict(future_years)
        
        return future_years.flatten(), forecast

# ==============================================================================
# SECTION 4: VISUALIZATION LAYER
# ==============================================================================

class VisualLayer:
    """
    Generates institutional-quality plots using Plotly.
    Focus is on data density and clarity.
    """
    
    @staticmethod
    def create_kpi_grid(df, quant_engine):
        """Renders top-level KPIs."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue (TTM)", f"${latest['revenue']/1e9:,.1f}B", f"{(latest['revenue']-prev['revenue'])/prev['revenue']:.1%}")
        c2.metric("Operating Income", f"${latest['operating_income']/1e9:,.1f}B", f"{(latest['operating_income']-prev['operating_income'])/prev['operating_income']:.1%}")
        c3.metric("Operating Margin", f"{latest['operating_margin']:.1%}", f"{(latest['operating_margin']-prev['operating_margin']):.2%} pts")
        c4.metric("Asset Turnover", f"{latest['asset_turnover']:.2f}x", "Efficiency Metric")

    @staticmethod
    def plot_efficiency_matrix(df):
        """Dual-axis chart comparing Scale (Assets) vs Efficiency (ROA)."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=df['year'], y=df['total_assets'], name='Total Assets', marker_color='#E8EAED'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['year'], y=df['operating_roa'], name='Operating ROA', line=dict(color='#1A73E8', width=3)), secondary_y=True)
        
        fig.update_layout(title="Capital Efficiency Analysis: Assets vs Return on Assets", template="simple_white", height=450)
        return fig

    @staticmethod
    def plot_volatility_cone(df):
        """Visualizes stock price deviation."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['year'], y=df['close'], mode='lines+markers', name='Stock Price', line=dict(color='#202124')))
        
        # Simple Bollinger Band approximation for visual context
        rolling_mean = df['close'].rolling(window=3).mean()
        rolling_std = df['close'].rolling(window=3).std()
        
        fig.add_trace(go.Scatter(x=df['year'], y=rolling_mean + (2*rolling_std), mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['year'], y=rolling_mean - (2*rolling_std), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(26, 115, 232, 0.1)', name='Volatility Band (2σ)'))
        
        fig.update_layout(title="Historical Price Action & Volatility Regime", template="simple_white", height=450)
        return fig

    @staticmethod
    def plot_monte_carlo(paths, start_year):
        """Renders stochastic simulation paths."""
        fig = go.Figure()
        
        # Plot simulation density
        subset = paths[:, :100] # Limit to 100 lines for performance
        years = list(range(start_year, start_year + len(paths)))
        
        for i in range(subset.shape[1]):
            fig.add_trace(go.Scatter(x=years, y=subset[:, i], mode='lines', line=dict(color='rgba(189, 193, 198, 0.1)'), showlegend=False))
            
        # Plot Mean
        mean_path = np.mean(paths, axis=1)
        fig.add_trace(go.Scatter(x=years, y=mean_path, mode='lines', name='Mean Forecast', line=dict(color='#137333', width=3)))
        
        fig.update_layout(title="Monte Carlo Revenue Simulation (1000 Iterations)", template="simple_white", height=450)
        return fig

# ==============================================================================
# SECTION 5: MAIN CONTROLLER
# ==============================================================================

def main():
    # 1. Header & Branding
    st.sidebar.image("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", width=180)
    st.sidebar.markdown("### Financial Intelligence Unit")
    st.sidebar.markdown("---")
    
    # 2. Navigation
    view_selection = st.sidebar.radio("Analytics Module:", 
        ["Executive Summary", "Operating Efficiency", "Risk & Volatility", "Strategic Forecasting"])
    
    # 3. Data Load
    pipeline = DataIngestionLayer("master.csv")
    df = pipeline.execute_pipeline()
    quant = QuantitativeEngine(df)

    # 4. Dashboard Logic
    if view_selection == "Executive Summary":
        st.header("Executive Summary: Fiscal Performance")
        st.markdown("Top-level assessment of growth, margin expansion, and capital allocation efficiency.")
        st.markdown("---")
        VisualLayer.create_kpi_grid(df, quant)
        
        st.subheader("Revenue vs Operating Income Trajectory")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df['year'], y=df['revenue'], name='Revenue', marker_color='#8AB4F8'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['year'], y=df['operating_margin'], name='Op Margin', line=dict(color='#D93025', width=3)), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    elif view_selection == "Operating Efficiency":
        st.header("Operating Efficiency & Capital Structure")
        st.markdown("Deep dive into Asset Turnover and Return on Assets (ROA).")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(VisualLayer.plot_efficiency_matrix(df), use_container_width=True)
            st.caption("**Insight:** Declining ROA amidst rising Assets suggests diminishing marginal returns on capital expenditure.")
        
        with c2:
            st.subheader("Cost Structure Analysis")
            # Calculate implied costs
            df['implied_costs'] = df['revenue'] - df['operating_income']
            fig_cost = px.area(df, x='year', y=['operating_income', 'implied_costs'], title="Revenue Decomposition: Costs vs Profit")
            fig_cost.update_layout(template="simple_white")
            st.plotly_chart(fig_cost, use_container_width=True)

    elif view_selection == "Risk & Volatility":
        st.header("Market Risk & Drawdown Analysis")
        var_95 = quant.calculate_var(0.95)
        st.warning(f"Projected Value at Risk (95% Confidence): {var_95:.2%} annual downside deviation.")
        
        st.plotly_chart(VisualLayer.plot_volatility_cone(df), use_container_width=True)
        
        st.subheader("Correlation Matrix (Macro Factors)")
        corr = df[['revenue', 'total_assets', 'close', 'annual_return']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Greys', title="Factor Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

    elif view_selection == "Strategic Forecasting":
        st.header("Strategic Forecasting Models")
        st.markdown("Stochastic and Deterministic modeling to bound future performance scenarios.")
        
        tab1, tab2 = st.tabs(["Monte Carlo (Stochastic)", "Linear Regression (Deterministic)"])
        
        with tab1:
            st.markdown("**Methodology:** Geometric Brownian Motion simulated over 1,000 iterations based on historical volatility.")
            paths = quant.monte_carlo_forecast()
            st.plotly_chart(VisualLayer.plot_monte_carlo(paths, df['year'].max()), use_container_width=True)
            
        with tab2:
            st.markdown("**Methodology:** Ordinary Least Squares (OLS) regression on historical top-line revenue.")
            years, pred = quant.generate_linear_forecast()
            
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=df['year'], y=df['revenue'], name='Historical', mode='lines+markers'))
            fig_reg.add_trace(go.Scatter(x=years, y=pred, name='OLS Forecast', line=dict(dash='dash', color='black')))
            st.plotly_chart(fig_reg, use_container_width=True)

if __name__ == "__main__":
    main()
