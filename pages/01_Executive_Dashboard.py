"""Executive Analytics Dashboard"""
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.metrics import calculate_kpis, calculate_churn_by_segment, calculate_tenure_band_distribution, generate_insights
from src.visualization.charts import (
    create_bar_chart, create_histogram, create_box_plot,
    create_pie_chart, COLORS, get_chart_template
)
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header

# Page config
st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Executive Overview",
    "📊",
    "Executive Analytics Dashboard",
    "A high-level view of customer health, churn patterns, and retention opportunities.",
)

# Load data
@st.cache_data
def load_data():
    cleaned_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"
    if not cleaned_path.exists():
        st.error("Dataset not found. Please run data ingestion first.")
        return None
    return pd.read_csv(cleaned_path)

df = load_data()

if df is None:
    st.stop()

# Session state for filters
if "filters" not in st.session_state:
    st.session_state.filters = {
        "contract": None,
        "payment_method": None,
        "internet_service": None,
        "churn_status": None
    }

# Sidebar filters
st.sidebar.header("Filters")

contract_options = ["All"] + sorted(df["Contract"].unique().tolist())
selected_contract = st.sidebar.selectbox("Contract Type", contract_options)

payment_options = ["All"] + sorted(df["PaymentMethod"].unique().tolist())
selected_payment = st.sidebar.selectbox("Payment Method", payment_options)

internet_options = ["All"] + sorted(df["InternetService"].unique().tolist())
selected_internet = st.sidebar.selectbox("Internet Service", internet_options)

churn_options = ["All", "Yes", "No"]
selected_churn = st.sidebar.selectbox("Churn Status", churn_options)

# Apply filters
df_filtered = df.copy()

if selected_contract != "All":
    df_filtered = df_filtered[df_filtered["Contract"] == selected_contract]

if selected_payment != "All":
    df_filtered = df_filtered[df_filtered["PaymentMethod"] == selected_payment]

if selected_internet != "All":
    df_filtered = df_filtered[df_filtered["InternetService"] == selected_internet]

if selected_churn != "All":
    df_filtered = df_filtered[df_filtered["Churn"] == selected_churn]

# Reset filters button
if st.sidebar.button("Reset Filters"):
    st.session_state.filters = {
        "contract": None,
        "payment_method": None,
        "internet_service": None,
        "churn_status": None
    }
    st.rerun()

# KPI Cards
st.header("Key Performance Indicators")

kpis = calculate_kpis(df_filtered)

# Recalculate KPIs for filtered data
filtered_kpis = {
    "total_customers": len(df_filtered),
    "churned_customers": (df_filtered["Churn"] == "Yes").sum(),
    "retained_customers": (df_filtered["Churn"] == "No").sum(),
    "churn_rate": ((df_filtered["Churn"] == "Yes").sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0,
    "avg_tenure": df_filtered["tenure"].mean(),
    "avg_monthly_charges": df_filtered["MonthlyCharges"].mean()
}

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", filtered_kpis["total_customers"])

with col2:
    st.metric("Churned Customers", filtered_kpis["churned_customers"])

with col3:
    st.metric("Churn Rate", f"{filtered_kpis['churn_rate']:.1f}%")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Avg Tenure", f"{filtered_kpis['avg_tenure']:.1f} months")

with col5:
    st.metric("Avg Monthly Charges", f"${filtered_kpis['avg_monthly_charges']:.2f}")

with col6:
    st.metric("Retained Customers", filtered_kpis["retained_customers"])

# Charts
st.header("Churn Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn by Contract Type")
    contract_churn = calculate_churn_by_segment(df_filtered, "Contract")
    fig_contract = create_bar_chart(
        contract_churn.reset_index(),
        x="Contract",
        y="churn_rate",
        title="Churn Rate by Contract Type",
        color="Contract"
    )
    st.plotly_chart(fig_contract, width='stretch')

with col2:
    st.subheader("Churn by Payment Method")
    payment_churn = calculate_churn_by_segment(df_filtered, "PaymentMethod")
    fig_payment = create_bar_chart(
        payment_churn.reset_index(),
        x="PaymentMethod",
        y="churn_rate",
        title="Churn Rate by Payment Method",
        color="PaymentMethod"
    )
    st.plotly_chart(fig_payment, width='stretch')

col3, col4 = st.columns(2)

with col3:
    st.subheader("Churn by Internet Service")
    internet_churn = calculate_churn_by_segment(df_filtered, "InternetService")
    fig_internet = create_bar_chart(
        internet_churn.reset_index(),
        x="InternetService",
        y="churn_rate",
        title="Churn Rate by Internet Service",
        color="InternetService"
    )
    st.plotly_chart(fig_internet, width='stretch')

with col4:
    st.subheader("Churn Distribution")
    churn_dist = df_filtered["Churn"].value_counts()
    fig_pie = create_pie_chart(
        pd.DataFrame({"Churn": churn_dist.index, "count": churn_dist.values}),
        names="Churn",
        values="count",
        title="Churn Distribution"
    )
    st.plotly_chart(fig_pie, width='stretch')

# Tenure Analysis
st.header("Tenure Analysis")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Tenure Band Distribution")
    tenure_bands = calculate_tenure_band_distribution(df_filtered)
    fig_tenure = create_bar_chart(
        tenure_bands.reset_index(),
        x="TenureBand",
        y="total_customers",
        title="Customer Count by Tenure Band",
        color="TenureBand"
    )
    st.plotly_chart(fig_tenure, width='stretch')

with col6:
    st.subheader("Monthly Charges Distribution by Churn")
    fig_charges = create_histogram(
        df_filtered,
        x="MonthlyCharges",
        title="Monthly Charges Distribution",
        color="Churn",
        nbins=30
    )
    st.plotly_chart(fig_charges, width='stretch')

# Business Insights
st.header("Business Insights")

insights = generate_insights(df_filtered)

for i, insight in enumerate(insights, 1):
    st.markdown(f"{i}. {insight}")

# Data export
st.header("Export Data")

if st.button("Download Filtered Data"):
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )
