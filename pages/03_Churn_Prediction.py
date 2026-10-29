"""Individual Churn Prediction Workspace"""
import streamlit as st
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference import predict_single_customer
from src.explainability.shap_explainer import get_explainer, explain_prediction, generate_plain_english_explanation, SHAP_AVAILABLE
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header
from src.data.app_data import load_customer_data_or_stop

# Page config
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎲",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Predictive Intelligence",
    "🎲",
    "Individual Churn Prediction",
    "Estimate churn risk for a customer profile and understand the factors behind the score.",
)

# Load data for reference
df = load_customer_data_or_stop(project_root)

# Check if model is available
models_path = project_root / "models"
results_path = models_path / "training_results.json"

if not results_path.exists():
    st.error("Model not trained. Please train models first using the Model Lab.")
    st.stop()

# Get unique values for dropdowns
contract_types = sorted(df["Contract"].unique().tolist())
payment_methods = sorted(df["PaymentMethod"].unique().tolist())
internet_services = sorted(df["InternetService"].unique().tolist())

# Input form
st.header("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
    contract = st.selectbox("Contract Type", contract_types)
    payment_method = st.selectbox("Payment Method", payment_methods)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

st.header("Additional Services")

col3, col4 = st.columns(2)

with col3:
    internet_service = st.selectbox("Internet Service", internet_services)
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

with col4:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Predict button
st.header("Prediction")

if st.button("Predict Churn", type="primary"):
    # Build customer features dictionary
    customer_features = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "Yes" if phone_service == "Yes" else "No",
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        # Get prediction
        result = predict_single_customer(customer_features)

        # Display results
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            proba = result["churn_probability"]
            st.metric("Churn Probability", f"{proba * 100:.1f}%")

        with col_b:
            risk = result["risk_band"]
            st.metric("Risk Band", risk)

        with col_c:
            if proba >= 0.7:
                st.markdown("🔴 **High Risk**")
            elif proba >= 0.4:
                st.markdown("🟡 **Medium Risk**")
            else:
                st.markdown("🟢 **Low Risk**")

        # Create gauge chart
        from src.visualization.charts import create_gauge_chart
        fig_gauge = create_gauge_chart(
            value=proba * 100,
            title="Churn Probability",
            min_val=0,
            max_val=100
        )
        st.plotly_chart(fig_gauge, width='stretch')

        # SHAP explanation if available
        if SHAP_AVAILABLE:
            st.header("Explainable AI")

            try:
                from src.models.training import load_model, get_best_model_name

                model_name = get_best_model_name()
                pipeline, metadata = load_model(model_name)

                # Create DataFrame for explanation
                customer_df = pd.DataFrame([customer_features])

                # Get background data
                background_sample = df.sample(min(100, len(df)), random_state=42)

                # Create explainer
                explainer = get_explainer(pipeline, background_sample)

                # Get explanation
                explanation = explain_prediction(explainer, customer_df, pipeline)

                # Plain English explanation
                st.subheader("Explanation")
                plain_explanation = generate_plain_english_explanation(explanation)
                st.markdown(plain_explanation)

                # Feature contributions
                st.subheader("Feature Contributions")

                contributions_df = pd.DataFrame(explanation["all_contributions"])

                col_pos, col_neg = st.columns(2)

                with col_pos:
                    st.markdown("**Factors Increasing Churn Risk**")
                    positive = contributions_df[contributions_df["contribution"] > 0].head(5)
                    if len(positive) > 0:
                        st.dataframe(positive[["feature", "value", "contribution"]], width='stretch')
                    else:
                        st.info("No positive factors")

                with col_neg:
                    st.markdown("**Factors Decreasing Churn Risk**")
                    negative = contributions_df[contributions_df["contribution"] < 0].tail(5)
                    if len(negative) > 0:
                        st.dataframe(negative[["feature", "value", "contribution"]], width='stretch')
                    else:
                        st.info("No negative factors")

            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")
        else:
            st.info("SHAP is not installed. Install with: pip install shap")

        # Download prediction
        st.header("Export")

        import json
        prediction_export = {
            "customer_features": customer_features,
            "prediction": result
        }

        st.download_button(
            label="Download Prediction (JSON)",
            data=json.dumps(prediction_export, indent=2),
            file_name="churn_prediction.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.exception(e)

# Disclaimer
st.header("Disclaimer")
st.warning("""
**Important Notes:**
- This prediction is based on a machine learning model trained on historical data.
- Model predictions may be incorrect and should support, not replace, business judgment.
- The model may not generalize to new customer segments or changing market conditions.
- Use these predictions as one input in your retention decision-making process.
""")
