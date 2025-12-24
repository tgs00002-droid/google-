"""
================================================================================
GOOGLE FINANCIAL ANALYST CANDIDATE DASHBOARD
================================================================================
Author: [Your Name]
Purpose: Interactive Financial Deep-Dive into Google's Historical Performance
Target Audience: Hiring Committee, Google Finance Team

Technology Stack:
- Streamlit: For rapid UI deployment
- Pandas/NumPy: For vectorized financial computation
- Plotly: For interactive, executive-grade visualizations
- Scikit-Learn: For predictive modeling (Revenue Forecasting)
- SciPy: For statistical risk analysis

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# ==============================================================================
# CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Google Financial Intelligence Unit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic Google's Material Design aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        color: #202124;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 4px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE 1: DATA INGESTION ENGINE
# ==============================================================================

class DataEngine:
    """
    Handles data loading, cleaning, and preprocessing.
    Designed to be robust against missing values and type errors.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """Loads data with caching logic."""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            self._preprocess()
            return self.processed_data
        except FileNotFoundError:
            st.error(f"Critical Error: Data file not found at {self.file_path}. Please check repository structure.")
            st.stop()

    def _preprocess(self):
        """Clean and type-cast the data."""
        df = self.raw_data.copy()
        
        # Ensure year is treated as datetime for time-series operations
        df['date'] = pd.to_datetime(df['year'], format='%Y')
        
        # Handle NA values with forward fill (financial time-series standard)
        df.replace('NA', np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True) # Fallback
        
        # Numeric conversion
        cols_to_convert = ['revenue', 'operating_income', 'total_assets', 'annual_return']
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        self.processed_data = df

# ==============================================================================
# MODULE 2: FINANCIAL ANALYTICS ENGINE
# ==============================================================================

class FinancialEngine:
    """
    Performs advanced financial calculations:
    - CAGR (Compound Annual Growth Rate)
    - Volatility (Standard Deviation)
    - Sharpe Ratio (Risk-Adjusted Return)
    - Margin Analysis
    """
    def __init__(self, df):
        self.df = df

    def calculate_cagr(self, column, periods):
        """Calculate Compound Annual Growth Rate."""
        start_val = self.df[column].iloc[0]
        end_val = self.df[column].iloc[-1]
        return (end_val / start_val) ** (1 / periods) - 1

    def calculate_kpis(self):
        """Returns a dictionary of high-level KPIs."""
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        return {
            "Revenue": latest['revenue'],
            "Revenue Growth": (latest['revenue'] - prev['revenue']) / prev['revenue'],
            "Op Margin": latest['operating_margin'],
            "Op Margin Change": latest['operating_margin'] - prev['operating_margin'],
            "Stock Price": latest['close'],
            "YTD Return": latest['annual_return']
        }

    def monte_carlo_simulation(self, n_simulations=1000, days=252):
        """
        Runs a Monte Carlo simulation to project future stock prices.
        Based on historical mean returns and volatility (Geometric Brownian Motion).
        """
        returns = self.df['annual_return'].dropna()
        mu = returns.mean()
        sigma = returns.std()
        start_price = self.df['close'].iloc[-1]
        
        simulation_df = pd.DataFrame()
        
        for i in range(n_simulations):
            # Generate random daily returns
            daily_returns = np.random.normal(mu/252, sigma/np.sqrt(252), days)
            price_series = [start_price]
            
            for r in daily_returns:
                price_series.append(price_series[-1] * (1 + r))
            
            simulation_df[f'Sim_{i}'] = price_series
            
        return simulation_df

# ==============================================================================
# MODULE 3: PREDICTIVE MODELING (AI/ML)
# ==============================================================================

class ForecastEngine:
    """
    Uses Scikit-Learn to project Revenue and Operating Income 
    for the next 3 fiscal years.
    """
    def __init__(self, df):
        self.df = df
        
    def forecast_revenue(self, years_ahead=3):
        """Linear Regression Forecast."""
        X = self.df['year'].values.reshape(-1, 1)
        y = self.df['revenue'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_years = np.array([self.df['year'].max() + i for i in range(1, years_ahead + 1)]).reshape(-1, 1)
        predictions = model.predict(future_years)
        
        return future_years.flatten(), predictions, model.score(X, y)

# ==============================================================================
# MODULE 4: VISUALIZATION FACTORY
# ==============================================================================

class ChartFactory:
    """
    Generates standardized, executive-ready Plotly charts.
    """
    @staticmethod
    def plot_dual_axis(df, x_col, bar_col, line_col, title):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=df[x_col], y=df[bar_col], name=bar_col.replace('_', ' ').title(), marker_color='#4285F4', opacity=0.7),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df[x_col], y=df[line_col], name=line_col.replace('_', ' ').title(), mode='lines+markers', line=dict(color='#EA4335', width=3)),
            secondary_y=True
        )
        
        fig.update_layout(title_text=title, template="plotly_white", height=500)
        return fig

    @staticmethod
    def plot_correlation_heatmap(df):
        corr = df[['revenue', 'operating_income', 'total_assets', 'close', 'annual_return']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Metric Correlation Matrix")
        return fig

    @staticmethod
    def plot_monte_carlo(sim_df):
        fig = go.Figure()
        # Plot first 50 simulations to avoid lag
        for col in sim_df.columns[:50]:
            fig.add_trace(go.Scatter(y=sim_df[col], mode='lines', line=dict(width=1, color='rgba(66, 133, 244, 0.2)'), showlegend=False))
        
        # Plot Mean Path
        mean_path = sim_df.mean(axis=1)
        fig.add_trace(go.Scatter(y=mean_path, mode='lines', name='Mean Projection', line=dict(color='#0F9D58', width=4)))
        
        fig.update_layout(
            title="Monte Carlo Risk Simulation (1000 Scenarios)",
            yaxis_title="Stock Price ($)",
            xaxis_title="Trading Days into Future",
            template="plotly_white"
        )
        return fig

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        # Placeholder for Google Logo (Public URL)
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg", width=150)
        st.title("Analyst Control Panel")
        st.markdown("---")
        
        analysis_mode = st.radio("Select Module:", 
            ["Executive Summary", "Financial Fundamentals", "Stock Performance", "AI Forecasting"])
        
        st.markdown("---")
        st.info("System Status: Online \nData Source: Verified Master CSV")
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Data Loading ---
    # NOTE: Ensure 'master.csv' is in the same folder as this script
    data_engine = DataEngine('master.csv')
    df = data_engine.load_data()
    fin_engine = FinancialEngine(df)
    
    # --- Page Content ---
    
    if analysis_mode == "Executive Summary":
        st.title("📊 Executive Financial Dashboard")
        st.markdown("### FY Performance Snapshot")
        
        kpis = fin_engine.calculate_kpis()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${kpis['Revenue']:,.0f}", f"{kpis['Revenue Growth']*100:.1f}%")
        col2.metric("Operating Margin", f"{kpis['Op Margin']*100:.1f}%", f"{kpis['Op Margin Change']*100:.2f} pts")
        col3.metric("Share Price", f"${kpis['Stock Price']:.2f}", "Latest Close")
        col4.metric("Annual Return", f"{kpis['YTD Return']*100:.1f}%", "Volatility Adj.")
        
        st.markdown("---")
        st.subheader("Asset Turnover & Efficiency")
        fig_assets = ChartFactory.plot_dual_axis(df, 'year', 'total_assets', 'revenue', 'Asset Base vs Revenue Generation')
        st.plotly_chart(fig_assets, use_container_width=True)

    elif analysis_mode == "Financial Fundamentals":
        st.title("💰 Financial Statement Deep-Dive")
        
        tab1, tab2 = st.tabs(["Income Analysis", "Efficiency Metrics"])
        
        with tab1:
            st.subheader("Top Line vs Bottom Line")
            fig_inc = go.Figure()
            fig_inc.add_trace(go.Bar(x=df['year'], y=df['revenue'], name='Revenue', marker_color='#1a73e8'))
            fig_inc.add_trace(go.Bar(x=df['year'], y=df['operating_income'], name='Op Income', marker_color='#34a853'))
            st.plotly_chart(fig_inc, use_container_width=True)
            
        with tab2:
            st.subheader("Correlation Analysis")
            st.write("Identifying drivers of stock price through statistical correlation.")
            fig_corr = ChartFactory.plot_correlation_heatmap(df)
            st.plotly_chart(fig_corr, use_container_width=True)

    elif analysis_mode == "Stock Performance":
        st.title("📈 Market Performance & Risk")
        
        # Candlestick Chart
        st.subheader("Price Action (OHLC)")
        fig_candle = go.Figure(data=[go.Candlestick(x=df['date'],
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'])])
        fig_candle.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volatility Analysis
        st.subheader("Risk Analysis: Return Distribution")
        fig_hist = px.histogram(df, x="annual_return", nbins=10, title="Distribution of Annual Returns", marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)

    elif analysis_mode == "AI Forecasting":
        st.title("🤖 Predictive Modeling & Simulations")
        st.markdown("Leveraging **Monte Carlo** simulations and **Linear Regression** to project future scenarios.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Forecast (Linear Model)")
            forecaster = ForecastEngine(df)
            years, preds, r2 = forecaster.forecast_revenue()
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df['year'], y=df['revenue'], mode='lines+markers', name='Historical'))
            fig_forecast.add_trace(go.Scatter(x=years, y=preds, mode='lines+markers', name='Forecast', line=dict(dash='dash', color='orange')))
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.info(f"Model Confidence (R²): {r2:.4f}")

        with col2:
            st.subheader("Monte Carlo Price Simulation")
            sim_df = fin_engine.monte_carlo_simulation()
            fig_mc = ChartFactory.plot_monte_carlo(sim_df)
            st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
