"""SQL Analytics Workspace"""
import streamlit as st
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_operations import (
    get_total_customers, get_churn_rate, get_churn_by_contract,
    get_churn_by_payment_method, get_churn_by_internet_service,
    get_tenure_summary, get_monthly_charges_summary,
    get_customer_segment_counts, search_customers
)
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header

# Page config
st.set_page_config(
    page_title="SQL Analytics",
    page_icon="💾",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Data Workspace",
    "💾",
    "SQL Analytics Workspace",
    "Explore customer data with ready-made queries, custom SQL, and validation metrics.",
)

# Check if database exists
db_path = project_root / "data" / "processed" / "customer_churn.db"

if not db_path.exists():
    st.error("Database not found. Please initialize the database first.")
    st.stop()

# Predefined queries
st.header("Predefined Queries")

query_options = {
    "Total Customers": "SELECT COUNT(*) as total FROM customers",
    "Overall Churn Rate": "SELECT SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate FROM customers",
    "Churn by Contract": "SELECT Contract, COUNT(*) as total, SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned, SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate FROM customers GROUP BY Contract",
    "Churn by Payment Method": "SELECT PaymentMethod, COUNT(*) as total, SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned, SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate FROM customers GROUP BY PaymentMethod",
    "Tenure Summary": "SELECT AVG(tenure) as avg_tenure, MIN(tenure) as min_tenure, MAX(tenure) as max_tenure FROM customers",
    "Monthly Charges Summary": "SELECT AVG(MonthlyCharges) as avg_charges, MIN(MonthlyCharges) as min_charges, MAX(MonthlyCharges) as max_charges FROM customers",
    "Customer Segments": "SELECT Contract, InternetService, COUNT(*) as count FROM customers GROUP BY Contract, InternetService ORDER BY count DESC"
}

selected_query = st.selectbox("Select Query", list(query_options.keys()))

if selected_query:
    query = query_options[selected_query]

    st.code(query, language="sql")

    try:
        result = search_customers()  # This will execute the query
        # Actually, we need to use execute_query directly
        from src.database.db_operations import execute_query
        result = execute_query(query)

        st.subheader("Results")
        st.dataframe(result, width='stretch')

        # Download results
        csv = result.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name=f"{selected_query.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error executing query: {e}")
        st.exception(e)

# Custom SQL query
st.header("Custom SQL Query")

st.info("""
You can write custom SQL queries to explore the data.
Available table: `customers`

Available columns:
- customerID, gender, SeniorCitizen, Partner, Dependents
- tenure, PhoneService, MultipleLines
- InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
- StreamingTV, StreamingMovies
- Contract, PaperlessBilling, PaymentMethod
- MonthlyCharges, TotalCharges, Churn
""")

custom_query = st.text_area(
    "Enter SQL Query",
    value="SELECT * FROM customers LIMIT 10",
    height=150
)

if st.button("Execute Custom Query"):
    try:
        result = execute_query(custom_query)
        st.subheader("Results")
        st.dataframe(result, width='stretch')

        # Download results
        csv = result.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="custom_query_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error executing query: {e}")
        st.exception(e)

# Interactive analytics
st.header("Interactive Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn by Contract")
    contract_data = get_churn_by_contract()
    st.dataframe(contract_data, width='stretch')

with col2:
    st.subheader("Churn by Payment Method")
    payment_data = get_churn_by_payment_method()
    st.dataframe(payment_data, width='stretch')

col3, col4 = st.columns(2)

with col3:
    st.subheader("Churn by Internet Service")
    internet_data = get_churn_by_internet_service()
    st.dataframe(internet_data, width='stretch')

with col4:
    st.subheader("Customer Segments")
    segment_data = get_customer_segment_counts()
    st.dataframe(segment_data, width='stretch')

# Summary statistics
st.header("Summary Statistics")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Tenure Summary")
    tenure_data = get_tenure_summary()
    st.dataframe(tenure_data, width='stretch')

with col6:
    st.subheader("Monthly Charges Summary")
    charges_data = get_monthly_charges_summary()
    st.dataframe(charges_data, width='stretch')

# Validation against Python calculations
st.header("SQL vs Python Validation")

st.info("Comparing SQL query results with equivalent Python calculations")

try:
    # Total customers
    sql_total = get_total_customers()
    df = pd.read_csv(project_root / "data" / "processed" / "telco_customer_churn_clean.csv")
    python_total = len(df)

    st.metric("Total Customers (SQL)", sql_total)
    st.metric("Total Customers (Python)", python_total)

    if abs(sql_total - python_total) == 0:
        st.success("✅ SQL and Python calculations match")
    else:
        st.warning(f"⚠️ Difference: {abs(sql_total - python_total)}")

    # Churn rate
    sql_churn = get_churn_rate()
    python_churn = (df["Churn"] == "Yes").sum() / len(df) * 100

    st.metric("Churn Rate (SQL)", f"{sql_churn:.2f}%")
    st.metric("Churn Rate (Python)", f"{python_churn:.2f}%")

    if abs(sql_churn - python_churn) < 0.01:
        st.success("✅ SQL and Python calculations match")
    else:
        st.warning(f"⚠️ Difference: {abs(sql_churn - python_churn):.4f}%")

except Exception as e:
    st.error(f"Error during validation: {e}")
    st.exception(e)
