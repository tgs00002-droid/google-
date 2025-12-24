import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 1. Configuration & Style
# ==========================================
st.set_page_config(
    page_title="Strategic Financial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional "Google-like" look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4285F4;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .highlight { color: #4285F4; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Data Engine (The "Brain")
# ==========================================
class FinancialEngine:
    """
    Handles all ETL (Extract, Transform, Load) and metric engineering.
    Decoupling logic from presentation makes this scalable.
    """
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.df = self._process_data()

    def _process_data(self):
        df = self.raw_data.copy()
        df = df.sort_values('year')
        
        # --- Feature Engineering ---
        # 1. Rule of 40 (Growth + Margin)
        # Note: Assuming margin is decimal (0.25) and yoy is decimal (0.15)
        df['rule_of_40'] = (df['revenue_yoy'] * 100) + (df['operating_margin'] * 100)
        
        # 2. Operating Leverage (Op Income Growth / Revenue Growth)
        df['op_income_yoy'] = df['operating_income'].pct_change()
        df['operating_leverage'] = df['op_income_yoy'] / df['revenue_yoy']
        
        # 3. Clean large numbers for display
        df['revenue_B'] = df['revenue'] / 1e9
        df['op_income_B'] = df['operating_income'] / 1e9
        
        return df

    def get_latest_metrics(self):
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        return {
            'year': latest['year'],
            'revenue': latest['revenue_B'],
            'rev_growth': latest['revenue_yoy'],
            'margin': latest['operating_margin'],
            'rule_40': latest['rule_of_40'],
            'delta_rev': (latest['revenue'] - prev['revenue']) / 1e9
        }

    def generate_forecast(self, base_growth, bull_scenario_adj, bear_scenario_adj, years=3):
        """Generates 3 scenarios: Base, Bull, and Bear case."""
        last_year = self.df['year'].max()
        last_rev = self.df['revenue'].iloc[-1]
        
        scenarios = []
        for year_idx in range(1, years + 1):
            curr_year = last_year + year_idx
            
            # Base Case
            base_rev = last_rev * ((1 + base_growth) ** year_idx)
            scenarios.append({'Year': curr_year, 'Revenue': base_rev, 'Scenario': 'Base Case'})
            
            # Bull Case
            bull_rev = last_rev * ((1 + base_growth + bull_scenario_adj) ** year_idx)
            scenarios.append({'Year': curr_year, 'Revenue': bull_rev, 'Scenario': 'Bull Case'})
            
            # Bear Case
            bear_rev = last_rev * ((1 + base_growth - bear_scenario_adj) ** year_idx)
            scenarios.append({'Year': curr_year, 'Revenue': bear_rev, 'Scenario': 'Bear Case'})
            
        return pd.DataFrame(scenarios)

# ==========================================
# 3. Visualization Class
# ==========================================
class ChartBuilder:
    """
    Dedicated class for generating Plotly charts.
    Ensures consistent styling across the app.
    """
    @staticmethod
    def plot_divergence(df):
        fig = go.Figure()
        
        # Revenue Line
        fig.add_trace(go.Scatter(
            x=df['year'], y=df['revenue_index'],
            name='Fundamental Growth (Revenue)',
            line=dict(color='#4285F4', width=3)
        ))
        
        # Stock Line
        fig.add_trace(go.Scatter(
            x=df['year'], y=df['stock_index'],
            name='Market Valuation (Stock Price)',
            line=dict(color='#34A853', width=3, dash='dot')
        ))
        
        fig.update_layout(
            title="Valuation Reality Check: Fundamentals vs. Hype",
            xaxis_title="Year",
            yaxis_title="Index (Base=100)",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

    @staticmethod
    def plot_rule_of_40(df):
        fig = px.scatter(
            df, x='revenue_yoy', y='operating_margin',
            size='revenue_B', color='year',
            title="The 'Rule of 40' Trade-off: Growth vs. Profitability",
            labels={'revenue_yoy': 'Revenue Growth', 'operating_margin': 'Operating Margin'},
            template="plotly_white",
            color_continuous_scale=px.colors.sequential.Bluered
        )
        # Add the threshold line
        fig.add_shape(type="line", x0=0, y0=0.4, x1=0.4, y0=0,
                      line=dict(color="Gray", width=2, dash="dash"))
        return fig

    @staticmethod
    def plot_forecast(hist_df, pred_df):
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(
            x=hist_df['year'], y=hist_df['revenue'],
            name='Historical Revenue',
            line=dict(color='black', width=3)
        ))
        
        # Forecast Scenarios
        colors = {'Base Case': '#4285F4', 'Bull Case': '#34A853', 'Bear Case': '#EA4335'}
        
        for scenario in pred_df['Scenario'].unique():
            subset = pred_df[pred_df['Scenario'] == scenario]
            fig.add_trace(go.Scatter(
                x=subset['Year'], y=subset['Revenue'],
                name=scenario,
                line=dict(color=colors[scenario], width=2, dash='dash' if scenario != 'Base Case' else 'solid')
            ))
            
        fig.update_layout(
            title="Forward-Looking Scenarios (Driver-Based)",
            yaxis_title="Revenue ($)",
            template="plotly_white"
        )
        return fig

# ==========================================
# 4. Main App Logic
# ==========================================
def main():
    # Load Data
    try:
        engine = FinancialEngine('master.csv')
    except FileNotFoundError:
        st.error("Error: 'master.csv' not found. Please upload the file.")
        return

    # Sidebar: Controls
    st.sidebar.title("🎮 Analyst Controls")
    st.sidebar.subheader("Forecast Assumptions")
    
    growth_input = st.sidebar.slider("Base Revenue Growth Rate", 0.0, 0.30, 0.12, 0.01, format="%.2f")
    macro_impact = st.sidebar.selectbox("Macro Economic Outlook", ["Neutral", "Recessionary (-5%)", "Expansionary (+5%)"])
    
    # Adjust scenario based on macro selection
    bull_adj = 0.05
    bear_adj = 0.05
    if macro_impact == "Recessionary (-5%)":
        growth_input -= 0.05
    elif macro_impact == "Expansionary (+5%)":
        growth_input += 0.05

    # --- HEADER SECTION ---
    st.title("📊 Strategic Financial Intelligence Dashboard")
    st.markdown("A centralized view of fundamentals, market valuation, and forward-looking scenarios.")
    
    # Top Level Metrics
    metrics = engine.get_latest_metrics()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FY Revenue", f"${metrics['revenue']:.1f}B", f"{metrics['rev_growth']:.1%}")
    c2.metric("Op Margin", f"{metrics['margin']:.1%}", "Stable")
    c3.metric("Rule of 40 Score", f"{metrics['rule_40']:.0f}", "Target > 40")
    c4.metric("Market Sentiment", "Neutral", "Beta: 1.05")

    st.markdown("---")

    # --- SECTION 1: HISTORICAL ANALYSIS ---
    st.header("1. Historical Performance & Valuation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chart: Divergence
        fig_div = ChartBuilder.plot_divergence(engine.df)
        st.plotly_chart(fig_div, use_container_width=True)
    
    with col2:
        st.subheader("💡 Analyst Insight")
        st.markdown(f"""
        <div class="metric-card">
        <b>Valuation Decoupling:</b><br>
        Over the last cycle, we observed a clear divergence between 
        <span class="highlight">Revenue Growth</span> (Fundamentals) and 
        <span class="highlight">Stock Price</span> (Sentiment).
        <br><br>
        While revenue grew at a steady CAGR, stock volatility was driven 
        primarily by multiple expansion rather than earnings surprises.
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: EFFICIENCY ---
    st.header("2. The Efficiency Frontier")
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.markdown(f"""
        ### The "Rule of 40"
        This chart plots **Growth vs. Margin**.
        
        * **Top Right:** High Growth, High Profit (Unicorns)
        * **Bottom Left:** Warning Zone
        
        *Current Position:* The company is maturing, moving from the top-right 
        towards a stable, cash-cow position (lower right).
        """)
        
    with col4:
        fig_rule40 = ChartBuilder.plot_rule_of_40(engine.df)
        st.plotly_chart(fig_rule40, use_container_width=True)

    # --- SECTION 3: FORECASTING ---
    st.header("3. Forward-Looking Scenarios")
    
    # Generate Data
    forecast_df = engine.generate_forecast(
        base_growth=growth_input, 
        bull_scenario_adj=bull_adj, 
        bear_scenario_adj=bear_adj
    )
    
    col5, col6 = st.columns([2, 1])
    
    with col5:
        fig_forecast = ChartBuilder.plot_forecast(engine.df, forecast_df)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    with col6:
        st.subheader("Scenario Logic")
        st.info(f"""
        **Base Case ({growth_input:.1%}):** Assumes steady-state search volume and ad pricing.
        
        **Bull Case (+5%):** Driven by AI-monetization upside and macro recovery.
        
        **Bear Case (-5%):** Driven by regulatory headwinds or margin compression.
        """)
        
        st.write("Full Forecast Data:")
        st.dataframe(forecast_df.set_index('Year'), height=150)

if __name__ == "__main__":
    main()
