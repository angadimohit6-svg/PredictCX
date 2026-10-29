"""Explainable AI Workspace - SHAP Explanations"""
import streamlit as st
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.training import load_model, get_best_model_name
from src.explainability.shap_explainer import (
    get_explainer, get_feature_importance, explain_prediction,
    generate_plain_english_explanation, SHAP_AVAILABLE
)
from src.visualization.charts import create_feature_importance_plot
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header
from src.data.app_data import load_customer_data_or_stop

# Page config
st.set_page_config(
    page_title="Explainable AI",
    page_icon="🔬",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Model Transparency",
    "🔬",
    "Explainable AI Workspace",
    "Explore the customer signals behind model behavior with global and individual explanations.",
)

# Check SHAP availability
if not SHAP_AVAILABLE:
    st.error("SHAP is not installed. Install with: pip install shap")
    st.stop()

# Load data
df = load_customer_data_or_stop(project_root)

# Check if model is available
models_path = project_root / "models"
results_path = models_path / "training_results.json"

if not results_path.exists():
    st.error("Model not trained. Please train models first using the Model Lab.")
    st.stop()

# Load model
model_name = get_best_model_name()
pipeline, metadata = load_model(model_name)

st.success(f"Using model: {model_name}")

# Tab navigation
tab1, tab2 = st.tabs(["Global Feature Importance", "Individual Prediction Explanation"])

with tab1:
    st.header("Global Feature Importance")
    st.markdown("Understand which features are most important for the model overall")

    st.info("This may take a moment to calculate...")

    try:
        # Sample background data for explainer
        background_sample = df.sample(min(100, len(df)), random_state=42)

        # Create explainer
        with st.spinner("Creating explainer..."):
            explainer = get_explainer(pipeline, background_sample)

        # Get feature importance
        with st.spinner("Calculating feature importance..."):
            importance_df = get_feature_importance(explainer, background_sample, pipeline)

        # Display feature importance
        fig_importance = create_feature_importance_plot(importance_df, top_n=15)
        st.plotly_chart(fig_importance, width='stretch')

        # Display as table
        st.subheader("Feature Importance Table")
        st.dataframe(importance_df.head(15), width='stretch')

        # Explanation
        st.header("Understanding SHAP Values")
        st.markdown("""
        **What are SHAP values?**
        - SHAP (SHapley Additive exPlanations) values measure the contribution of each feature to a prediction
        - Values can be positive (increases churn probability) or negative (decreases churn probability)
        - The magnitude indicates the strength of the feature's influence

        **Global Feature Importance**
        - Shows the average absolute SHAP value for each feature across the dataset
        - Features with higher importance have a larger impact on predictions on average
        - This helps identify which factors the model considers most important

        **Important Notes:**
        - Feature importance does not imply causation
        - Correlated features can share importance
        - Explanations depend on the model and reference data
        - Use these insights to guide investigation, not as definitive proof
        """)

    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        st.exception(e)

with tab2:
    st.header("Individual Prediction Explanation")
    st.markdown("Explain a single customer's prediction")

    # Customer selection
    st.subheader("Select Customer")

    customer_options = ["Select a customer..."] + df["customerID"].tolist()[:100]  # Limit to first 100 for performance
    selected_customer_id = st.selectbox("Customer ID", customer_options)

    if selected_customer_id != "Select a customer...":
        try:
            # Get customer data
            customer_data = df[df["customerID"] == selected_customer_id].iloc[0:1]

            # Get prediction
            from src.models.inference import predict_churn_probability
            proba = predict_churn_probability(customer_data, model_name)[0]

            # Display prediction
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Churn Probability", f"{proba * 100:.1f}%")

            with col2:
                if proba >= 0.7:
                    risk = "High"
                elif proba >= 0.4:
                    risk = "Medium"
                else:
                    risk = "Low"
                st.metric("Risk Band", risk)

            with col3:
                st.metric("Model", model_name)

            # Get SHAP explanation
            st.subheader("SHAP Explanation")

            with st.spinner("Calculating SHAP values..."):
                background_sample = df.sample(min(100, len(df)), random_state=42)
                explainer = get_explainer(pipeline, background_sample)
                explanation = explain_prediction(explainer, customer_data, pipeline)

            # Plain English explanation
            st.subheader("Plain-English Explanation")
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

            # All contributions
            st.subheader("All Feature Contributions")
            st.dataframe(contributions_df, width='stretch')

            # Customer profile
            st.subheader("Customer Profile")
            customer_display = customer_data.drop(columns=["customerID", "Churn"])
            st.dataframe(customer_display.T, width='stretch')

        except Exception as e:
            st.error(f"Error generating explanation: {e}")
            st.exception(e)

    else:
        st.info("Select a customer to view their prediction explanation")

# Disclaimer
st.header("Disclaimer")
st.warning("""
**Important Notes about SHAP Explanations:**
- SHAP describes how features contribute to the model's prediction relative to a reference
- Feature contributions are not proof of causation
- Explanations depend on the model, reference data, and feature representation
- Correlated features can complicate attribution
- The model may be incorrect even if the explanation seems reasonable
- Use these insights to support, not replace, business judgment
""")
