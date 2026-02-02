"""
Fraud Detection Dashboard - Streamlit
Real-time fraud detection monitoring and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import os
from datetime import datetime, timedelta

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="🚨 Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Database Connection
# ==========================================
@st.cache_resource
def get_db_engine():
    """Get database connection"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        st.error("❌ DATABASE_URL not configured")
        st.stop()
    return create_engine(db_url)

engine = get_db_engine()

# ==========================================
# Data Loading Functions
# ==========================================
@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_payments(limit=1000):
    """Load recent payments"""
    query = f"""
        SELECT 
            id,
            trans_num,
            amt,
            merchant,
            category,
            city,
            state,
            transaction_time,
            is_fraud,
            created_at
        FROM payments
        ORDER BY id DESC
        LIMIT {limit}
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=30)
def load_predictions(limit=1000):
    """Load recent predictions"""
    query = f"""
        SELECT 
            p.trans_num,
            p.amt,
            p.merchant,
            p.category,
            p.city,
            p.state,
            pred.is_fraud as predicted_fraud,
            pred.fraud_probability,
            pred.predicted_at
        FROM predictions pred
        JOIN payments p ON pred.trans_num = p.trans_num
        ORDER BY pred.predicted_at DESC
        LIMIT {limit}
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=30)
def load_fraud_alerts():
    """Load fraud alerts"""
    query = """
        SELECT 
            fa.id,
            fa.trans_num,
            p.amt,
            p.merchant,
            p.city,
            p.state,
            pred.fraud_probability,
            fa.email_sent,
            fa.email_sent_at,
            fa.created_at
        FROM fraud_alerts fa
        JOIN payments p ON fa.trans_num = p.trans_num
        LEFT JOIN predictions pred ON fa.trans_num = pred.trans_num
        ORDER BY fa.created_at DESC
        LIMIT 100
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=30)
def get_statistics():
    """Get overall statistics"""
    query = """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as total_frauds,
            AVG(amt) as avg_amount,
            SUM(amt) as total_amount
        FROM payments
    """
    return pd.read_sql(query, engine).iloc[0]

# ==========================================
# Header
# ==========================================
st.title("🚨 Fraud Detection Dashboard")
st.markdown("**Real-time monitoring of credit card fraud detection system**")

# ==========================================
# Sidebar - Filters
# ==========================================
st.sidebar.header("🔍 Filters")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Time range
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 100 transactions", "Last 500 transactions", "Last 1000 transactions"],
    index=1
)

limit_map = {
    "Last 100 transactions": 100,
    "Last 500 transactions": 500,
    "Last 1000 transactions": 1000
}
limit = limit_map[time_range]

# Fraud filter
show_only_frauds = st.sidebar.checkbox("Show only frauds", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Auto-refresh**: Every 30 seconds")
st.sidebar.markdown(f"**Last update**: {datetime.now().strftime('%H:%M:%S')}")

# ==========================================
# Load Data
# ==========================================
with st.spinner("Loading data..."):
    stats = get_statistics()
    predictions_df = load_predictions(limit)
    fraud_alerts_df = load_fraud_alerts()

# ==========================================
# KPIs - Top Metrics
# ==========================================
st.header("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{int(stats['total_transactions']):,}",
        delta=f"{len(predictions_df)} recent"
    )

with col2:
    fraud_rate = (stats['total_frauds'] / stats['total_transactions'] * 100) if stats['total_transactions'] > 0 else 0
    st.metric(
        label="Fraud Rate",
        value=f"{fraud_rate:.2f}%",
        delta=f"{int(stats['total_frauds'])} frauds",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Average Amount",
        value=f"${stats['avg_amount']:.2f}",
        delta="per transaction"
    )

with col4:
    st.metric(
        label="Total Volume",
        value=f"${stats['total_amount']:,.0f}",
        delta="all time"
    )

st.markdown("---")

# ==========================================
# Charts Section
# ==========================================
st.header("📈 Analytics")

if not predictions_df.empty:
    
    # Row 1: Fraud Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Fraud vs Legitimate")
        
        fraud_counts = predictions_df['predicted_fraud'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[fraud_counts.get(False, 0), fraud_counts.get(True, 0)],
            hole=0.4,
            marker_colors=['#00cc96', '#ef553b']
        )])
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Fraud Probability Distribution")
        
        fig_hist = px.histogram(
            predictions_df,
            x='fraud_probability',
            nbins=20,
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 2: Time Series
    st.subheader("📉 Fraud Detection Over Time")
    
    # Prepare time series data
    predictions_df['predicted_at'] = pd.to_datetime(predictions_df['predicted_at'])
    predictions_df['date'] = predictions_df['predicted_at'].dt.date
    
    daily_frauds = predictions_df.groupby(['date', 'predicted_fraud']).size().unstack(fill_value=0)
    
    fig_line = go.Figure()
    
    if False in daily_frauds.columns:
        fig_line.add_trace(go.Scatter(
            x=daily_frauds.index,
            y=daily_frauds[False],
            mode='lines+markers',
            name='Legitimate',
            line=dict(color='#00cc96', width=2)
        ))
    
    if True in daily_frauds.columns:
        fig_line.add_trace(go.Scatter(
            x=daily_frauds.index,
            y=daily_frauds[True],
            mode='lines+markers',
            name='Fraud',
            line=dict(color='#ef553b', width=2)
        ))
    
    fig_line.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Transactions",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Row 3: Geographic and Category Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Frauds by State")
        
        frauds_by_state = predictions_df[predictions_df['predicted_fraud'] == True].groupby('state').size().sort_values(ascending=False).head(10)
        
        fig_bar = px.bar(
            x=frauds_by_state.values,
            y=frauds_by_state.index,
            orientation='h',
            color_discrete_sequence=['#ef553b']
        )
        fig_bar.update_layout(
            xaxis_title="Number of Frauds",
            yaxis_title="State",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🏪 Frauds by Category")
        
        frauds_by_category = predictions_df[predictions_df['predicted_fraud'] == True].groupby('category').size().sort_values(ascending=False).head(10)
        
        fig_bar2 = px.bar(
            x=frauds_by_category.values,
            y=frauds_by_category.index,
            orientation='h',
            color_discrete_sequence=['#ab63fa']
        )
        fig_bar2.update_layout(
            xaxis_title="Number of Frauds",
            yaxis_title="Category",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar2, use_container_width=True)
    
    # Row 4: Amount Analysis
    st.subheader("💰 Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Legitimate Amount", f"${predictions_df[predictions_df['predicted_fraud'] == False]['amt'].mean():.2f}")
    
    with col2:
        st.metric("Avg Fraud Amount", f"${predictions_df[predictions_df['predicted_fraud'] == True]['amt'].mean():.2f}")
    
    fig_box = px.box(
        predictions_df,
        x='predicted_fraud',
        y='amt',
        color='predicted_fraud',
        color_discrete_map={False: '#00cc96', True: '#ef553b'},
        labels={'predicted_fraud': 'Fraud Status', 'amt': 'Amount ($)'}
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.warning("⚠️ No predictions data available yet. Wait for the ML pipeline to run.")

st.markdown("---")

# ==========================================
# Recent Transactions Table
# ==========================================
st.header("📋 Recent Predictions")

if not predictions_df.empty:
    
    # Filter
    display_df = predictions_df.copy()
    if show_only_frauds:
        display_df = display_df[display_df['predicted_fraud'] == True]
    
    # Format for display
    display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.1%}")
    display_df['predicted_fraud'] = display_df['predicted_fraud'].apply(lambda x: "🚨 FRAUD" if x else "✅ LEGIT")
    display_df['amt'] = display_df['amt'].apply(lambda x: f"${x:.2f}")
    
    # Select columns
    display_cols = ['trans_num', 'amt', 'merchant', 'category', 'city', 'state', 'predicted_fraud', 'fraud_probability', 'predicted_at']
    
    st.dataframe(
        display_df[display_cols].head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "trans_num": "Transaction ID",
            "amt": "Amount",
            "merchant": "Merchant",
            "category": "Category",
            "city": "City",
            "state": "State",
            "predicted_fraud": "Status",
            "fraud_probability": "Fraud Prob.",
            "predicted_at": "Predicted At"
        }
    )
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Data (CSV)",
        data=csv,
        file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.info("ℹ️ No predictions available yet.")

st.markdown("---")

# ==========================================
# Fraud Alerts Section
# ==========================================
st.header("🚨 Fraud Alerts")

if not fraud_alerts_df.empty:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", len(fraud_alerts_df))
    
    with col2:
        emails_sent = fraud_alerts_df['email_sent'].sum()
        st.metric("Emails Sent", emails_sent)
    
    with col3:
        pending = len(fraud_alerts_df) - emails_sent
        st.metric("Pending", pending)
    
    st.subheader("Recent Alerts")
    
    # Format
    alerts_display = fraud_alerts_df.copy()
    alerts_display['amt'] = alerts_display['amt'].apply(lambda x: f"${x:.2f}")
    alerts_display['fraud_probability'] = alerts_display['fraud_probability'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    alerts_display['email_sent'] = alerts_display['email_sent'].apply(lambda x: "✅ Sent" if x else "⏳ Pending")
    
    display_cols = ['trans_num', 'amt', 'merchant', 'city', 'state', 'fraud_probability', 'email_sent', 'created_at']
    
    st.dataframe(
        alerts_display[display_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "trans_num": "Transaction ID",
            "amt": "Amount",
            "merchant": "Merchant",
            "city": "City",
            "state": "State",
            "fraud_probability": "Probability",
            "email_sent": "Email Status",
            "created_at": "Alert Time"
        }
    )

else:
    st.info("ℹ️ No fraud alerts yet. System is monitoring...")

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Fraud Detection System</strong> - Powered by Machine Learning</p>
    <p>Data refreshes every 30 seconds | Model: RandomForest | Deployment: HuggingFace Spaces</p>
</div>
""", unsafe_allow_html=True)
