"""Customer Explorer - Search and filter customer data"""
import streamlit as st
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_operations import search_customers
from src.models.inference import batch_predict
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header
from src.data.app_data import load_customer_data_or_stop
from src.models.deployment import ensure_model_artifacts_or_stop

# Page config
st.set_page_config(
    page_title="Customer Explorer",
    page_icon="🔍",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Customer Intelligence",
    "🔍",
    "Customer Explorer",
    "Search, segment, and inspect customer profiles with optional churn-risk predictions.",
)

# Load data
df = load_customer_data_or_stop(project_root)

# Get unique values for filters
contract_types = sorted(df["Contract"].unique().tolist())
payment_methods = sorted(df["PaymentMethod"].unique().tolist())
internet_services = sorted(df["InternetService"].unique().tolist())

# Sidebar filters
st.sidebar.header("Search & Filters")

# Customer ID search
customer_id_search = st.sidebar.text_input("Search by Customer ID")

# Categorical filters
contract_filter = st.sidebar.selectbox("Contract Type", ["All"] + contract_types)
payment_filter = st.sidebar.selectbox("Payment Method", ["All"] + payment_methods)
internet_filter = st.sidebar.selectbox("Internet Service", ["All"] + internet_services)
churn_filter = st.sidebar.selectbox("Churn Status", ["All", "Yes", "No"])

# Numerical filters
min_tenure, max_tenure = st.sidebar.slider(
    "Tenure Range (months)",
    min_value=int(df["tenure"].min()),
    max_value=int(df["tenure"].max()),
    value=(int(df["tenure"].min()), int(df["tenure"].max()))
)

min_charges, max_charges = st.sidebar.slider(
    "Monthly Charges Range ($)",
    min_value=float(df["MonthlyCharges"].min()),
    max_value=float(df["MonthlyCharges"].max()),
    value=(float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()))
)

# Apply filters
df_filtered = df.copy()

if customer_id_search:
    df_filtered = df_filtered[df_filtered["customerID"].str.contains(customer_id_search, case=False, na=False)]

if contract_filter != "All":
    df_filtered = df_filtered[df_filtered["Contract"] == contract_filter]

if payment_filter != "All":
    df_filtered = df_filtered[df_filtered["PaymentMethod"] == payment_filter]

if internet_filter != "All":
    df_filtered = df_filtered[df_filtered["InternetService"] == internet_filter]

if churn_filter != "All":
    df_filtered = df_filtered[df_filtered["Churn"] == churn_filter]

df_filtered = df_filtered[
    (df_filtered["tenure"] >= min_tenure) &
    (df_filtered["tenure"] <= max_tenure) &
    (df_filtered["MonthlyCharges"] >= min_charges) &
    (df_filtered["MonthlyCharges"] <= max_charges)
]

# Reset filters button
if st.sidebar.button("Reset Filters"):
    st.rerun()

# Display results count
st.subheader(f"Showing {len(df_filtered)} customers")

# Add predictions on request, training models first on fresh deployments.
add_predictions = st.checkbox("Add Churn Predictions", value=False)

if add_predictions:
    ensure_model_artifacts_or_stop(project_root)
    try:
        df_filtered = batch_predict(df_filtered)
        st.success("Predictions added successfully")
    except Exception as e:
        st.error(f"Error adding predictions: {e}")

# Display customer table
if len(df_filtered) > 0:
    # Select columns to display
    display_columns = ["customerID", "Contract", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]

    if add_predictions and "churn_probability" in df_filtered.columns:
        display_columns.extend(["churn_probability", "risk_band"])

    # Reorder columns
    df_display = df_filtered[display_columns + [col for col in df_filtered.columns if col not in display_columns]]

    # Display with pagination
    page_size = 20
    page = st.number_input("Page", min_value=1, max_value=(len(df_filtered) // page_size) + 1, value=1)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(
        df_display.iloc[start_idx:end_idx],
        width='stretch',
        height=400
    )

    # Customer detail view
    st.header("Customer Detail View")

    if len(df_filtered) > 0:
        selected_customer = st.selectbox(
            "Select Customer for Detail View",
            options=df_filtered["customerID"].tolist()
        )

        if selected_customer:
            customer_data = df_filtered[df_filtered["customerID"] == selected_customer].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Customer Profile")
                st.write(f"**Customer ID:** {customer_data['customerID']}")
                st.write(f"**Contract:** {customer_data['Contract']}")
                st.write(f"**Tenure:** {customer_data['tenure']} months")
                st.write(f"**Monthly Charges:** ${customer_data['MonthlyCharges']:.2f}")
                st.write(f"**Total Charges:** ${customer_data['TotalCharges']:.2f}")

            with col2:
                st.subheader("Services")
                st.write(f"**Phone Service:** {customer_data['PhoneService']}")
                st.write(f"**Internet Service:** {customer_data['InternetService']}")
                st.write(f"**Payment Method:** {customer_data['PaymentMethod']}")
                st.write(f"**Paperless Billing:** {customer_data['PaperlessBilling']}")

            if add_predictions and "churn_probability" in customer_data:
                st.subheader("Churn Prediction")
                proba = customer_data['churn_probability']
                risk = customer_data['risk_band']

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Churn Probability", f"{proba * 100:.1f}%")

                with col_b:
                    st.metric("Risk Band", risk)

                with col_c:
                    if proba >= 0.7:
                        st.markdown("🔴 **High Risk**")
                    elif proba >= 0.4:
                        st.markdown("🟡 **Medium Risk**")
                    else:
                        st.markdown("🟢 **Low Risk**")

    # Export
    st.header("Export Data")

    if st.button("Download Filtered Data"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="customer_explorer_results.csv",
            mime="text/csv"
        )
else:
    st.info("No customers match the current filters. Try adjusting your search criteria.")
