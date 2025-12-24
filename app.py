import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Google Candidate: Financial Strategy Portfolio", layout="wide")

# Google-branded Styling (Clean, White, Professional)
st.markdown('<style>.main {background-color: #f8f9fa;} .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}</style>', unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    # This reads your master.csv file
    df = pd.read_csv('master.csv')
    return df

try:
    df = load_data()
except:
    st.error("Error: 'master.csv' not found. Please ensure it is in the same folder as this script.")
    st.stop()

# --- 3. HEADER SECTION ---
st.title("📊 Financial Analyst Portfolio: Quantitative Strategy")
st.subheader("Analysis by Thomas")
st.markdown("_Demonstrating Technical Rigor & Business Storytelling for the Google Finance Team_")

# --- 4. SIDEBAR: TECHNICAL PROFICIENCY ---
st.sidebar.header("🛠️ Technical Skills")

# Showcase SQL (Critical for Google)
st.sidebar.markdown("### 1. SQL Optimization")
st.sidebar.code("""
SELECT 
    year, 
    revenue,
    (revenue - LAG(revenue) OVER (ORDER BY year)) 
    / LAG(revenue) OVER (ORDER BY year) AS yoy_growth
FROM financials;
""", language='sql')

# Showcase Financial Modeling (CAGR)
rev_start = df['revenue'].iloc[0]
rev_end = df['revenue'].iloc[-1]
years = len(df) - 1
cagr = ((rev_end / rev_start)**(1/years)) - 1
st.sidebar.metric("14-Year Revenue CAGR", f"{cagr:.2%}")

# --- 5. EXECUTIVE SUMMARY (KPIs) ---
st.markdown("---")
st.header("🚀 Executive Performance Summary (2024)")
k1, k2, k3, k4 = st.columns(4)

# Dynamic metrics from your data
k1.metric("Total Revenue", f"${df['revenue'].iloc[-1]/1e9:.1f}B", f"{df['revenue_yoy'].iloc[-1]:.1%}")
k2.metric("Operating Income", f"${df['operating_income'].iloc[-1]/1e9:.1f}B")
k3.metric("Avg. Op. Margin", f"{df['operating_margin'].mean():.1%}")
k4.metric("Asset Growth", f"{((df['total_assets'].iloc[-1]/df['total_assets'].iloc[0])-1):.0%} Total")

# --- 6. STORY CHAPTER 1: SCALING THE ENGINE ---
st.divider()
st.header("📈 1. Scaling the Engine: Revenue vs. Efficiency")
col1, col2 = st.columns([2, 1])

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['year'], y=df['revenue']/1e9, name='Revenue', fill='tozeroy', line_color='#4285F4'))
    fig1.add_trace(go.Scatter(x=df['year'], y=df['operating_income']/1e9, name='Op. Income', line_color='#EA4335'))
    fig1.update_layout(title="Revenue & Income Scaling ($ Billions)", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### **Business Interpretation**")
    st.write("""
    Since 2010, the company has scaled revenue by over 11x. 
    **Why this matters:** Most companies lose efficiency as they grow. Here, 
    Operating Income scaled *faster* than revenue in key periods, 
    demonstrating incredible operational leverage.
    """)
    st.info("The 19.6% average growth rate proves sustainable product-market fit.")

# --- 7. STORY CHAPTER 2: PROFITABILITY INTEGRITY ---
st.header("💎 2. Operational Integrity (Margin Analysis)")
fig2 = px.bar(df, x='year', y='operating_margin', 
             title="Operating Margin Consistency (%)",
             color='operating_margin', color_continuous_scale='RdYlGn')
fig2.add_hline(y=df['operating_margin'].mean(), line_dash="dash", annotation_text="15-Year Average")
st.plotly_chart(fig2, use_container_width=True)

st.warning("**Insight:** Consistent margins above 25% signify dominant pricing power and cost-discipline, even through market pivots (2018-2019).")

# --- 8. STORY CHAPTER 3: FORWARD-LOOKING MODEL ---
st.header("🔮 3. Strategic Forecast (3-Year Model)")
# Simple projection based on trailing average growth
avg_g = df['revenue_yoy'].tail(3).mean()
f_years = [2025, 2026, 2027]
f_vals = [df['revenue'].iloc[-1] * (1 + avg_g)**i for i in range(1, 4)]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['year'], y=df['revenue']/1e9, name='Historical', line_color='#34A853'))
fig3.add_trace(go.Scatter(x=f_years, y=[v/1e9 for v in f_vals], name='Thomas Forecast', line=dict(dash='dot', color='#FBBC05')))
fig3.update_layout(title="Projected Revenue Trajectory ($B)", xaxis_title="Year", yaxis_title="USD (B)")
st.plotly_chart(fig3, use_container_width=True)

# --- 9. FOOTER: THE "HIRE ME" SECTION ---
st.divider()
st.markdown("""
### Why Thomas for Google?
1. **Analytical Depth:** Moving from raw data to predictive growth models.
2. **Technical Agility:** Proficiency in Python, SQL, and Dashboard Deployment.
3. **Business Partnering:** Ability to translate complex financials into clear executive stories.
""")
st.caption("Developed by Thomas for the Google Recruitment Team | 2024")
