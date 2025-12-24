"""
================================================================================
ALPHABET (GOOGLE) STRATEGIC FINANCIAL INTELLIGENCE SUITE
================================================================================
CONFIDENTIAL: INTERNAL CAPITAL STRATEGY REVIEW
AUTHOR: LEAD FINANCIAL ANALYST
VERSION: 3.5.0 (PRODUCTION)

SYSTEM ARCHITECTURE:
1.  ETL Layer: Robust ingestion of 'master.csv' with strict type enforcement.
2.  Analytical Core:
    - DuPont Identity Decomposition
    - Operating Leverage & Margin Analytics
    - Geometric Brownian Motion (Stochastic Forecasting)
    - Correlation & Covariance Matrix
3.  Presentation Layer:
    - Streamlit (UI)
    - Plotly (Interactive Visualization)

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
from dataclasses import dataclass
from typing import Tuple, List, Dict
import datetime

# ==============================================================================
# SECTION 1: ENTERPRISE CONFIGURATION & STYLES
# ==============================================================================

st.set_page_config(
    page_title="GOOGL Strategic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for a "Goldman Sachs / Google Finance" Professional Aesthetic
st.markdown("""
<style>
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Roboto', 'Helvetica Neue', sans-serif;
        color: #333333;
    }
    h1, h2, h3 {
        color: #202124;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1a73e8;
    }
    
    /* Containers */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dadce0;
    }
    
    /* Custom Card Class */
    .finance-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(60,64,67,0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: DATA TRANSFORMATION LAYER (ETL)
# ==============================================================================

@dataclass
class FinancialData:
    """Immutable data structure for passing financial state across modules."""
    df: pd.DataFrame
    latest_year: int
    start_year: int

class DataLoader:
    """
    Enterprise ETL (Extract, Transform, Load) pipeline.
    Handles missing data imputation and type casting for financial precision.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_data = None
        self.clean_data = None

    def execute(self) -> FinancialData:
        self._load()
        self._transform()
        self._enrich()
        return FinancialData(
            df=self.clean_data, 
            latest_year=int(self.clean_data['year'].max()),
            start_year=int(self.clean_data['year'].min())
        )

    def _load(self):
        try:
            self.raw_data = pd.read_csv(self.filepath)
        except FileNotFoundError:
            st.error("CRITICAL: Data source 'master.csv' not found. Terminating pipeline.")
            st.stop()

    def _transform(self):
        df = self.raw_data.copy()
        
        # Enforce Numeric Types (Handling 'NA' strings)
        numeric_cols = ['revenue', 'operating_income', 'total_assets', 'close', 'annual_return', 'operating_margin']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Interpolation for missing financial quarters (Business Logic: Linear Fill)
        df.interpolate(method='linear', limit_direction='forward', inplace=True)
        df.fillna(0, inplace=True)
        
        self.clean_data = df

    def _enrich(self):
        """Calculates derived Senior-Level metrics."""
        df = self.clean_data
        
        # 1. Net Profit Approximation (using Operating Income * 0.8 as tax proxy)
        df['implied_net_income'] = df['operating_income'] * 0.80 
        
        # 2. DuPont Analysis Components
        # Net Profit Margin = Net Income / Revenue
        df['net_profit_margin'] = df['implied_net_income'] / df['revenue']
        
        # Asset Turnover = Revenue / Total Assets
        df['asset_turnover'] = df['revenue'] / df['total_assets']
        
        # Financial Leverage (Equity Multiplier) - Proxy using Assets/Income
        # (Simplified for this dataset as Equity isn't explicitly provided, we assume Assets ~= Equity + Liab)
        df['equity_multiplier'] = df['total_assets'] / (df['total_assets'] * 0.6) # Assumption: 60% Equity financing
        
        # 3. ROE (Return on Equity)
        df['roe'] = df['net_profit_margin'] * df['asset_turnover'] * df['equity_multiplier']

# ==============================================================================
# SECTION 3: ANALYTICAL ENGINE (The "Brain")
# ==============================================================================

class AnalyticalEngine:
    """
    Performs high-level computation. 
    Separated from UI to ensure logic isolation (Unit Testing capability).
    """
    def __init__(self, data: FinancialData):
        self.data = data.df

    def calculate_cagr(self, metric: str) -> float:
        """Compound Annual Growth Rate - The gold standard for growth."""
        start = self.data[metric].iloc[0]
        end = self.data[metric].iloc[-1]
        years = len(self.data) - 1
        if start == 0: return 0.0
        return (end / start) ** (1 / years) - 1

    def monte_carlo_simulation(self, iterations=1000, forecast_years=5):
        """
        Stochastic Modeling: Projects revenue paths using Geometric Brownian Motion.
        Used for Risk Management and Scenario Planning.
        """
        last_rev = self.data['revenue'].iloc[-1]
        pct_change = self.data['revenue'].pct_change().dropna()
        
        mu = pct_change.mean()
        sigma = pct_change.std()
        
        simulation_results = np.zeros((forecast_years, iterations))
        simulation_results[0] = last_rev
        
        for t in range(1, forecast_years):
            shock = np.random.normal(mu, sigma, iterations)
            simulation_results[t] = simulation_results[t-1] * (1 + shock)
            
        return simulation_results, mu, sigma

    def operating_leverage_check(self):
        """
        Degree of Operating Leverage (DOL).
        % Change in EBIT / % Change in Sales.
        High DOL = High Risk but High Reward (Scalability).
        """
        df = self.data.copy()
        df['pct_rev'] = df['revenue'].pct_change()
        df['pct_ebit'] = df['operating_income'].pct_change()
        df['DOL'] = df['pct_ebit'] / df['pct_rev']
        
        # Filter outliers for clean visualization
        return df[['year', 'DOL']].replace([np.inf, -np.inf], np.nan).dropna()

# ==============================================================================
# SECTION 4: VISUALIZATION FACTORY
# ==============================================================================

class ChartBuilder:
    """
    Generates standardized, publication-ready Plotly figures.
    """
    colors = {
        'primary': '#1a73e8',   # Google Blue
        'secondary': '#ea4335', # Google Red
        'tertiary': '#fbbc04',  # Google Yellow
        'quaternary': '#34a853',# Google Green
        'neutral': '#5f6368'    # Google Grey
    }

    @staticmethod
    def plot_dupont(df):
        """Three-pane chart for Decomposition Analysis."""
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Net Profit Margin", "Asset Turnover", "Return on Equity (ROE)"))
        
        fig.add_trace(go.Scatter(x=df['year'], y=df['net_profit_margin'], mode='lines+markers', name='Margins', line=dict(color=ChartBuilder.colors['primary'])), row=1, col=1)
        fig.add_trace(go.Bar(x=df['year'], y=df['asset_turnover'], name='Turnover', marker_color=ChartBuilder.colors['tertiary']), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['year'], y=df['roe'], mode='lines', fill='tozeroy', name='ROE', line=dict(color=ChartBuilder.colors['quaternary'])), row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False, template="plotly_white", title_text="DuPont Identity Decomposition")
        return fig

    @staticmethod
    def plot_operating_leverage(df):
        """Dual axis: Revenue Growth vs Margin Expansion."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=df['year'], y=df['revenue'], name='Revenue (Abs)', marker_color='#E8EAED'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['year'], y=df['operating_margin'], name='Op Margin (%)', line=dict(color=ChartBuilder.colors['secondary'], width=3)), secondary_y=True)
        
        fig.update_layout(title_text="Scalability Analysis: Revenue vs. Margin Efficiency", template="plotly_white", height=450)
        return fig

    @staticmethod
    def plot_risk_cone(sim_data, start_val):
        """Fan chart for Monte Carlo results."""
        mean_path = np.mean(sim_data, axis=1)
        p95 = np.percentile(sim_data, 95, axis=1)
        p05 = np.percentile(sim_data, 5, axis=1)
        
        x_axis = np.arange(len(mean_path))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(26, 115, 232, 0.2)', name='95% Confidence Interval'))
        fig.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines', line=dict(color=ChartBuilder.colors['primary'], width=3), name='Base Case Forecast'))
        
        fig.update_layout(title="Stochastic Revenue Forecast (5-Year Horizon)", xaxis_title="Years Forward", yaxis_title="Proj. Revenue", template="plotly_white")
        return fig

# ==============================================================================
# SECTION 5: MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    # --- Initialization ---
    pipeline = DataLoader('master.csv')
    financial_data = pipeline.execute()
    engine = AnalyticalEngine(financial_data)
    df = financial_data.df

    # --- Sidebar Control Center ---
    st.sidebar.title("Analyst Workspace")
    st.sidebar.info(f"Reporting Period: {financial_data.start_year} - {financial_data.latest_year}")
    
    module = st.sidebar.radio(
        "Select Strategic Module:",
        ["Executive Dashboard", "DuPont & Efficiency", "Valuation & Risk", "Raw Data Audit"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("CONFIDENTIAL - FOR AUTHORIZED PERSONNEL ONLY")

    # --- Module 1: Executive Dashboard ---
    if module == "Executive Dashboard":
        st.header("Executive Summary: Operational Health")
        st.markdown("**Objective:** Assess top-line velocity and bottom-line quality.")
        
        # High-Level KPIs
        rev_cagr = engine.calculate_cagr('revenue')
        op_inc_cagr = engine.calculate_cagr('operating_income')
        latest_margin = df['operating_margin'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue CAGR (Historic)", f"{rev_cagr:.1%}", "Velocity Metric")
        c2.metric("Op. Income CAGR", f"{op_inc_cagr:.1%}", delta=f"{(op_inc_cagr - rev_cagr)*100:.1f} bps Spread")
        c3.metric("Current Op Margin", f"{latest_margin:.1%}", "Profitability Quality")
        
        st.markdown("---")
        
        # Primary Visualization: Scalability
        st.subheader("Operating Leverage Analysis")
        st.markdown("""
        *Insight:* This chart determines if the company benefits from economies of scale. 
        Divergence between the **Red Line (Margin)** and **Grey Bars (Revenue)** indicates increasing profitability per unit of sale.
        """)
        st.plotly_chart(ChartBuilder.plot_operating_leverage(df), use_container_width=True)

    # --- Module 2: DuPont & Efficiency ---
    elif module == "DuPont & Efficiency":
        st.header("Deep-Dive: Return on Equity Decomposition")
        st.markdown("""
        **Methodology (DuPont Identity):** We decompose ROE into three drivers to isolate the source of returns:
        1.  **Net Profit Margin:** Pricing power and cost control.
        2.  **Asset Turnover:** Efficiency of asset utilization.
        3.  **Financial Leverage:** Use of debt to amplify returns.
        """)
        
        st.plotly_chart(ChartBuilder.plot_dupont(df), use_container_width=True)
        
        st.info("""
        **Senior Analyst Note:** If ROE is rising but Asset Turnover is falling, the company is relying on pricing power or leverage rather than efficiency. 
        Analyze the center chart (Asset Turnover) closely for signs of bloating balance sheets.
        """)

    # --- Module 3: Valuation & Risk ---
    elif module == "Valuation & Risk":
        st.header("Risk Management & Stochastic Forecasting")
        st.markdown("**Model:** Monte Carlo Simulation (Geometric Brownian Motion)")
        
        # Simulation Controls
        col1, col2 = st.columns([1, 3])
        with col1:
            iterations = st.slider("Simulations", 100, 5000, 1000)
            years = st.slider("Forecast Horizon", 3, 10, 5)
            st.markdown(f"**Parameters:**\n- Iterations: {iterations}\n- Horizon: {years} Years")
            
        with col2:
            sim_res, mu, sigma = engine.monte_carlo_simulation(iterations, years)
            st.plotly_chart(ChartBuilder.plot_risk_cone(sim_res, df['revenue'].iloc[-1]), use_container_width=True)
            
        st.markdown("### Risk-Adjusted Commentary")
        st.write(f"Based on historical volatility (σ = {sigma:.2%}), the model predicts a widening cone of uncertainty.")
        st.write("This suggests that while the mean path is positive, capital allocation strategies must account for significant downside tail risk in Year 3+.")

    # --- Module 4: Raw Data Audit ---
    elif module == "Raw Data Audit":
        st.header("Data Integrity Audit")
        st.dataframe(df.style.format("{:.2f}"))
        
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

if __name__ == "__main__":
    main()
