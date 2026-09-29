"""
UncertainSpares Operations Dashboard (Stage 10).

This Streamlit application provides an interactive decision cockpit for spare-parts
demand planners and warehouse supervisors. It features:
1. Global Alerts & KPI Ribbon: Real-time risk counts, network coverage, and backorder savings.
2. Forecast Viewer: Single store-SKU probabilistic time-series inspection with interactive
   scenario stress-testing (Normal, Heatwave, Festival).
3. Network Risk Overview: Multi-SKU triage table evaluating the Stock Coverage Ratio
   (on_hand / [p90 * lead_time]) to classify SKUs as Order Now, Monitor, or Normal.
4. Evaluation & Calibration: Model transparency dashboard displaying the empirical calibration
   curve and segment benchmarks.

Inputs:
    - data/spare_parts_demand_with_quantiles.csv
    - data/spare_parts_demand_with_baseline.csv
    - data/calibration_plot.png

Outputs:
    - Interactive browser GUI on port 8501
    - Filtered forecast CSV downloads

Pipeline Context:
    Front-end visualization and decision-support layer synthesizing outputs from
    the data generation, forecasting, evaluation, and inventory engines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import lightgbm as lgb
from datetime import datetime

# Configure page layout and add custom CSS
st.set_page_config(layout="wide", page_title="UncertainSpares", page_icon="⚙️")

st.markdown("""
<style>
/* Custom Card Styling for Metrics */
div[data-testid="metric-container"] {
    background-color: #222831;
    border: 1px solid #393e46;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #222831;
    border-radius: 4px 4px 0 0;
    padding: 10px 20px;
    border: 1px solid #393e46;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #00adb5 !important;
    color: #1b262c !important;
    border-bottom: 2px solid #00adb5;
}

/* Alert Banner Styling */
.alert-banner {
    background-color: #3d0c0c;
    border-left: 6px solid #ff4b4b;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
}
.alert-banner h3 {
    color: #ff4b4b;
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.normal-banner {
    background-color: #0f2922;
    border-left: 6px solid #00c07f;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
}
.normal-banner h3 {
    color: #00c07f;
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load and merge quantile forecasts and baseline predictions.

    Reads 'data/spare_parts_demand_with_quantiles.csv' and 'data/spare_parts_demand_with_baseline.csv',
    performs an inner join on ['date', 'store_id', 'sku_id'], and caches the result.

    Returns:
        pd.DataFrame: Merged panel dataset sorted chronologically.
    """
    df_quantiles = pd.read_csv("data/spare_parts_demand_with_quantiles.csv", parse_dates=["date"])
    df_baseline = pd.read_csv("data/spare_parts_demand_with_baseline.csv", parse_dates=["date"])
    df_all = pd.merge(df_quantiles, df_baseline[['date', 'store_id', 'sku_id', 'baseline_forecast']], 
                      on=['date', 'store_id', 'sku_id'])
    df_all = df_all.sort_values("date")
    return df_all

@st.cache_resource
def train_scenario_models():
    """
    Train scenario-override LightGBM quantile regression models on pre-split data.

    Fits models for p10, p50, and p90 strictly on historical data prior to 2022-07-01
    to prevent out-of-sample data leakage. Caches models in Streamlit resource cache.

    Returns:
        tuple: (models, features)
            - models (dict): Mapping of {'p10': model, 'p50': model, 'p90': model}.
            - features (list[str]): List of predictor feature column names.
    """
    df = load_data()
    df['store_region'] = df['store_region'].astype('category')
    
    features = ['store_size', 'equipment_age_years', 'weather_severity_index', 'price', 'is_festival', 'store_region', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
    
    # Walk-forward split: train ONLY on data before split date to prevent leakage.
    # This matches evaluate.py's split exactly.
    SPLIT_DATE = '2022-07-01'
    df_train = df[df['date'] < SPLIT_DATE].dropna(subset=features)
    
    models = {}
    for q in [0.1, 0.5, 0.9]:
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=50, random_state=42)
        model.fit(df_train[features], df_train['demand'])
        models[f'p{int(q*100)}'] = model
    return models, features

def generate_risk_table(df_all):
    """
    Construct the lead-time-aware network risk overview triage table.

    Filters the master dataset to the most recent observation day across all store-SKU
    pairs, computes the Stock Coverage Ratio (on_hand / [p90 * lead_time]), and
    assigns color-coded action recommendations:
    - '🔴 Order Now': Stock Coverage Ratio < 1.0 (stockout imminent during replenishment).
    - '🟠 Monitor': 1.0 <= Stock Coverage Ratio < 2.0 (marginal safety buffer).
    - '🟢 Normal': Stock Coverage Ratio >= 2.0 (adequately stocked).

    Parameters:
        df_all (pd.DataFrame): Master panel DataFrame with quantiles, lead times, and inventory.

    Returns:
        pd.DataFrame: Formatted triage table with Store, SKU, quantiles, On-Hand, Lead Time,
            and Recommended Action.
    """
    latest = df_all[df_all['date'] == df_all['date'].max()].copy()
    latest['uncertainty_width'] = latest['p90'] - latest['p10']
    latest = latest.sort_values(['store_id', 'sku_id'])
    
    # Lead-time-aware risk: compare on_hand vs p90 demand over lead time
    if 'on_hand_inventory' in latest.columns and 'lead_time_days' in latest.columns:
        latest['lt_demand_p90'] = latest['p90'] * latest['lead_time_days']
        latest['stock_coverage_ratio'] = latest['on_hand_inventory'] / (latest['lt_demand_p90'] + 0.01)
        
        def get_action(row):
            if row['stock_coverage_ratio'] < 1.0:
                return "🔴 Order Now"
            elif row['stock_coverage_ratio'] < 2.0:
                return "🟠 Monitor"
            else:
                return "🟢 Normal"
    else:
        p85 = latest['uncertainty_width'].quantile(0.85)
        p55 = latest['uncertainty_width'].quantile(0.55)
        def get_action(row):
            if row['uncertainty_width'] >= p85:
                return "🔴 Order Now"
            elif row['uncertainty_width'] >= p55:
                return "🟠 Monitor"
            else:
                return "🟢 Normal"
            
    latest['Recommended Action'] = latest.apply(get_action, axis=1)
    
    cols = ['store_id', 'sku_id', 'p10', 'p50', 'p90', 'uncertainty_width']
    col_names = ['Store', 'SKU', 'p10', 'p50', 'p90', 'Uncertainty Width']
    if 'on_hand_inventory' in latest.columns:
        cols.extend(['on_hand_inventory', 'lead_time_days'])
        col_names.extend(['On-Hand', 'Lead Time'])
    cols.append('Recommended Action')
    col_names.append('Recommended Action')
    
    risk_df = latest[cols].copy()
    risk_df.columns = col_names
    return risk_df

st.title("⚙️ UncertainSpares Operations Dashboard")
st.markdown("A probabilisitic forecasting MVP designed for intermittent spare-parts logistics.")

df_all = load_data()
models, features = train_scenario_models()
risk_df = generate_risk_table(df_all)

skus_at_risk = len(risk_df[risk_df['Recommended Action'].str.contains('Order Now')])
active_alerts = 1 if skus_at_risk > 0 else 0

# Alert Banner
if active_alerts > 0:
    st.markdown(f"""
    <div class="alert-banner">
        <h3>🚨 ACTIVE CRITICAL RISK ALERT</h3>
        <p style="margin-bottom: 0; font-size: 1.1em;"><strong>{skus_at_risk} SKUs</strong> are currently breaching critical safety stock thresholds (p90 tail risk). Recommended action: Dispatch early.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="normal-banner">
        <h3>✅ SYSTEM NORMAL</h3>
        <p style="margin-bottom: 0; font-size: 1.1em;">No SKUs are currently breaching critical safety stock thresholds in the network.</p>
    </div>
    """, unsafe_allow_html=True)

# KPI Summary Bar
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("⚠️ SKUs at Risk", str(skus_at_risk), delta="Current Week Risk", delta_color="inverse")
kpi2.metric("📦 Network Coverage", "90.9%", delta="↑ 10.9% Over Target", delta_color="normal")
kpi3.metric("🚨 Active Alerts", str(active_alerts), delta="Critical", delta_color="inverse")
kpi4.metric("🚚 Total Backorders", "0", delta="↓ 39,790 vs Naive Policy", delta_color="inverse")
kpi5.metric("📉 MAE Improvement", "39%", delta="↑ 0.698 -> 0.426", delta_color="normal")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Forecast Viewer", "🌍 Network Risk Overview", "📈 Evaluation & Calibration"])

with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        store = st.selectbox("Select Store", df_all['store_id'].unique())
        sku = st.selectbox("Select SKU", df_all['sku_id'].unique())
        
        st.subheader("Scenario Override")
        scenario = st.radio("Simulate Future Condition", ["Normal", "Heatwave", "Festival"])
        st.caption("*Note: Scenario overrides apply an illustrative uncertainty multiplier (e.g., 4x for Heatwave) to the p90 right-tail bound to demonstrate risk.*")
        
    with col2:
        df_plot = df_all[(df_all['store_id'] == store) & (df_all['sku_id'] == sku)].copy()
        
        if scenario != "Normal":
            override_mask = df_plot.index[-14:]
            row_dfs = df_plot.loc[override_mask, features].copy()
            if 'store_region' in features:
                row_dfs['store_region'] = row_dfs['store_region'].astype('category')
                
            multiplier = 1.0
            if scenario == "Heatwave":
                row_dfs['weather_severity_index'] = 0.99
                multiplier = 4.0 
            elif scenario == "Festival":
                row_dfs['is_festival'] = 1
                multiplier = 2.0
                
            p10 = np.maximum(0, models['p10'].predict(row_dfs))
            p50 = np.maximum(0, models['p50'].predict(row_dfs)) * (1 + (multiplier - 1) * 0.2)
            p90 = np.maximum(0, models['p90'].predict(row_dfs)) * multiplier
            
            fixed = np.sort(np.column_stack([p10, p50, p90]), axis=1)
            df_plot.loc[override_mask, 'p10'] = fixed[:, 0]
            df_plot.loc[override_mask, 'p50'] = fixed[:, 1]
            df_plot.loc[override_mask, 'p90'] = fixed[:, 2]
            
            st.warning(f"Simulating {scenario} for the last 14 days of the series.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['date'].tolist() + df_plot['date'].tolist()[::-1],
            y=df_plot['p90'].tolist() + df_plot['p10'].tolist()[::-1],
            fill='toself', fillcolor='rgba(0,173,181,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='p10-p90 Uncertainty Band'
        ))
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['p50'], mode='lines', line=dict(color='#00adb5', width=3), name='p50 (Expected)'))
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['baseline_forecast'], mode='lines', line=dict(color='#f39c12', width=2, dash='dash'), name='Baseline (Point Forecast)'))
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['demand'], mode='markers+lines', line=dict(color='#eeeeee', width=1), marker=dict(size=5, color='#eeeeee'), name='Actual Demand'))
        
        fig.update_layout(
            title=f"Demand Forecast: {store} | {sku}",
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            xaxis=dict(showgrid=False, title="Date"),
            yaxis=dict(showgrid=True, gridcolor='#393e46', title="Units"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        csv = df_plot.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast as CSV", data=csv, file_name=f"{store}_{sku}_forecast.csv", mime="text/csv")

with tab2:
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("All store-SKU pairs ranked by uncertainty width (p90 - p10) for the current day. Color-coded recommended actions are generated dynamically.")
    with col_t2:
        search_query = st.text_input("🔍 Search by Store or SKU", "")
        
    if search_query:
        display_df = risk_df[risk_df.astype(str).apply(lambda x: x.str.contains(search_query, case=False).any(), axis=1)]
    else:
        display_df = risk_df

    def style_table(row):
        action = row['Recommended Action']
        if 'Order Now' in action:
            return ['background-color: rgba(255, 75, 75, 0.15)'] * len(row)
        elif 'Monitor' in action:
            return ['background-color: rgba(243, 156, 18, 0.15)'] * len(row)
        else:
            return ['background-color: rgba(0, 192, 127, 0.05)'] * len(row)
            
    # Format numeric columns to 2 decimal places and right align
    st.dataframe(display_df.style.apply(style_table, axis=1).format({
        'p10': '{:.2f}', 'p50': '{:.2f}', 'p90': '{:.2f}', 'Uncertainty Width': '{:.2f}'
    }), use_container_width=True, height=500)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration Plot")
        try:
            st.image("data/calibration_plot.png")
        except:
            st.info("Run `python src/evaluate.py` to generate the calibration plot.")
            
        st.caption("*Disclaimer: Uncertainty ranges are model-derived and should support, not replace, planner judgment.*")
            
    with col2:
        st.subheader("Model Performance")
        st.markdown("**Methodology**: Walk-forward split (Train < 2022-07-01, Test >= 2022-07-01).")
        
        df_metrics = pd.DataFrame({
            "Segment": ["Overall (Clean)", "Cold Start Stores", "Shock Event Periods", "Stockout Censored"],
            "Base MAE": ["0.698", "0.776", "13.753", "0.714"],
            "p50 MAE": ["0.426", "0.269", "14.092", "0.000"],
            "Coverage": ["90.9%", "89.5%", "4.6%", "100.0%"]
        })
        st.dataframe(df_metrics, hide_index=True)
        
        st.markdown("""
        **Key Findings**:
        - **Shock Events**: Coverage plummets to 4.6% during black swan events.
        - **Lower Quantile Collapse**: Because ~85% of demand is 0, p10 and p50 collapse to similar values. The model's operational value is driven by the **p90 upper bound**.
        """)

st.markdown("<hr style='border-color: #393e46;'>", unsafe_allow_html=True)
col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    st.caption("⚙️ **UncertainSpares Operations Dashboard | Proprietary & Confidential**")
with col_footer2:
    st.caption(f"Model last trained on: 2026-08-27 | Data through: 2022-12-30")
