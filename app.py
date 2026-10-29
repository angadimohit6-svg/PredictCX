"""Customer Intelligence AI - Main Application Entry Point"""
import streamlit as st
from pathlib import Path

from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header
from src.data.app_data import load_customer_data_or_stop

# Page configuration
st.set_page_config(
    page_title="Customer Intelligence AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
configure_dashboard_ui()

project_root = Path(__file__).parent
render_sidebar_navigation(project_root)
df = load_customer_data_or_stop(project_root)

# Dataset status
cleaned_data_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"
db_path = project_root / "data" / "processed" / "customer_churn.db"

st.sidebar.success(f"✅ Dataset loaded: {len(df)} customers")

if db_path.exists():
    st.sidebar.success("✅ Database initialized")
else:
    st.sidebar.warning("⚠️ Database not initialized")

st.sidebar.markdown("---")

# Model status
models_path = project_root / "models"
results_path = models_path / "training_results.json"

if results_path.exists():
    import json
    with open(results_path, "r") as f:
        results = json.load(f)
    best_model = results.get("best_model", "Unknown")
    st.sidebar.success(f"✅ Model trained: {best_model}")
else:
    st.sidebar.warning("⚠️ Models not trained")

st.sidebar.markdown("---")

# Footer
st.sidebar.markdown("""
### Data Attribution
IBM Telco Customer Churn Dataset
[GitHub Repository](https://github.com/IBM/telco-customer-churn-on-icp4d)

### Model Version
Churn Prediction v1.0
""")

# Main page content
render_page_header(
    "Customer Intelligence",
    "🎯",
    "Customer Intelligence AI",
    "Understand customer behavior, predict churn, and build smarter retention strategies.",
)
st.markdown("""
Explore the platform:
- **Executive Analytics**: High-level KPIs and trends
- **Customer Explorer**: Search and filter customer data
- **Churn Prediction**: Predict churn for individual customers
- **Model Lab**: Compare ML model performance
- **Explainable AI**: Understand model predictions with SHAP
- **SQL Analytics**: Query customer data with SQL
- **Retention Simulator**: Simulate retention campaign outcomes

Choose a workspace from the sidebar to get started.
""")

# Quick stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churned = (df["Churn"] == "Yes").sum()
    st.metric("Churned Customers", churned)

with col3:
    churn_rate = (churned / len(df)) * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

with col4:
    avg_tenure = df["tenure"].mean()
    st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
