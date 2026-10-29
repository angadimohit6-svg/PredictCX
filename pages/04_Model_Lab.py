"""Model Training and Comparison Lab"""
import streamlit as st
from pathlib import Path
import pandas as pd
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.training import train_and_evaluate_all, load_model, get_best_model_name
from src.visualization.charts import (
    create_confusion_matrix_plot, create_roc_curve,
    create_feature_importance_plot, COLORS
)
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header

# Page config
st.set_page_config(
    page_title="Model Lab",
    page_icon="🧪",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Machine Learning",
    "🧪",
    "Model Training & Comparison Lab",
    "Compare model quality, explore classification trade-offs, and manage training runs.",
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

# Check if models are already trained
models_path = project_root / "models"
results_path = models_path / "training_results.json"

if results_path.exists():
    with open(results_path, "r") as f:
        existing_results = json.load(f)
    st.success(f"Models already trained. Best model: {existing_results['best_model']}")

    if st.button("Retrain Models"):
        results_path.unlink()
        st.rerun()
else:
    st.info("No trained models found. Click below to train models.")

    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                results = train_and_evaluate_all(df, save_artifacts=True)
                st.success("Model training complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training models: {e}")
                st.exception(e)

# Display results if available
if results_path.exists():
    with open(results_path, "r") as f:
        results = json.load(f)

    st.header("Model Comparison")

    # Create comparison table
    model_comparison = []
    for model_name, model_data in results["models"].items():
        model_comparison.append({
            "Model": model_name,
            "CV ROC-AUC": f"{model_data['cv_metrics']['cv_mean']:.4f} ± {model_data['cv_metrics']['cv_std']:.4f}",
            "Test ROC-AUC": f"{model_data['test_metrics']['roc_auc']:.4f}",
            "Test F1": f"{model_data['test_metrics']['f1']:.4f}",
            "Test Precision": f"{model_data['test_metrics']['precision']:.4f}",
            "Test Recall": f"{model_data['test_metrics']['recall']:.4f}",
            "Test PR-AUC": f"{model_data['test_metrics']['pr_auc']:.4f}"
        })

    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df, width='stretch')

    # Model selection
    st.header("Model Details")

    selected_model = st.selectbox(
        "Select Model to View Details",
        options=list(results["models"].keys())
    )

    if selected_model:
        model_data = results["models"][selected_model]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ROC-AUC", f"{model_data['test_metrics']['roc_auc']:.4f}")

        with col2:
            st.metric("F1 Score", f"{model_data['test_metrics']['f1']:.4f}")

        with col3:
            st.metric("PR-AUC", f"{model_data['test_metrics']['pr_auc']:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = model_data['test_metrics']['confusion_matrix']
        fig_cm = create_confusion_matrix_plot(cm, ["Retained", "Churned"])
        st.plotly_chart(fig_cm, width='stretch')

        # Explanation of metrics
        st.subheader("Metric Explanation")

        st.markdown("""
        **Confusion Matrix:**
        - **True Negatives (Top-Left):** Customers correctly predicted as retained
        - **False Positives (Top-Right):** Customers incorrectly predicted as churned (Type I error)
        - **False Negatives (Bottom-Left):** Customers incorrectly predicted as retained (Type II error)
        - **True Positives (Bottom-Right):** Customers correctly predicted as churned

        **Business Impact:**
        - **False Positives:** Wasted retention efforts on customers who would have stayed
        - **False Negatives:** Missed opportunities to prevent churn of at-risk customers
        """)

        # Threshold exploration
        st.subheader("Threshold Exploration")

        threshold = st.slider(
            "Classification Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

        st.info(f"Current threshold: {threshold:.2f}")
        st.info("Adjusting the threshold changes the trade-off between precision and recall.")
        st.info("- Higher threshold = Fewer false positives, more false negatives")
        st.info("- Lower threshold = Fewer false negatives, more false positives")

        # Feature importance (if available)
        try:
            pipeline, metadata = load_model(selected_model)

            if metadata.get("feature_names"):
                st.subheader("Feature Importance")

                importance_df = pd.DataFrame({
                    "feature": metadata["feature_names"],
                    "importance": [1.0] * len(metadata["feature_names"])  # Placeholder
                })

                # For actual models, we'd need to extract real feature importance
                # This is a simplified version
                st.info("Feature importance visualization requires model-specific implementation.")

        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")

# Training configuration
st.header("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric("Random State", "42")
    st.metric("Test Size", "20%")

with col2:
    st.metric("CV Folds", "5")
    st.metric("CV Strategy", "Stratified K-Fold")

# Model descriptions
st.header("Model Descriptions")

st.markdown("""
**Dummy Classifier (Baseline):**
- Makes predictions using stratified random sampling
- Serves as a baseline to compare against more sophisticated models
- Expected ROC-AUC: ~0.5 (random guessing)

**Logistic Regression:**
- Linear model that models the probability of churn
- Interpretable coefficients show feature direction and magnitude
- Works well with linear relationships between features and target

**Random Forest:**
- Ensemble of decision trees
- Captures non-linear relationships and feature interactions
- Generally robust and performs well on many tasks

**XGBoost:**
- Gradient boosting algorithm with tree-based learners
- Often achieves state-of-the-art performance on tabular data
- Handles missing values and complex feature interactions well
""")

# Retrain warning
st.header("Retrain Models")

st.warning("""
**Note:** Retraining models will overwrite existing model artifacts.
Make sure you want to do this before clicking the Retrain button.
""")
