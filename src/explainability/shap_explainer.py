"""SHAP explainability for churn predictions"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Explainability features will be limited.")

from src.models.training import load_model, get_best_model_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_explainer(pipeline, X_background: pd.DataFrame):
    """Create a SHAP explainer for the model.

    Args:
        pipeline: Trained pipeline
        X_background: Background dataset for explainer

    Returns:
        SHAP explainer
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    # Get the model from the pipeline
    model = pipeline.named_steps["model"]

    # Get the preprocessed background data
    X_background_processed = pipeline.named_steps["preprocessor"].transform(X_background)

    # Create explainer based on model type
    model_type = type(model).__name__

    if model_type in ["LogisticRegression", "LinearRegression"]:
        explainer = shap.LinearExplainer(model, X_background_processed)
    elif model_type in ["RandomForestClassifier", "XGBClassifier"]:
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to generic explainer
        explainer = shap.Explainer(model, X_background_processed)

    logger.info(f"Created {model_type} explainer")
    return explainer


def get_feature_importance(
    explainer,
    X_background: pd.DataFrame,
    pipeline
) -> pd.DataFrame:
    """Get global feature importance using SHAP values.

    Args:
        explainer: SHAP explainer
        X_background: Background dataset
        pipeline: Trained pipeline

    Returns:
        DataFrame with feature importance
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")

    # Preprocess background data
    X_processed = pipeline.named_steps["preprocessor"].transform(X_background)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)

    # Handle binary classification (use positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Get feature names
    if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(len(mean_shap))]

    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_shap
    }).sort_values("importance", ascending=False)

    return importance_df


def explain_prediction(
    explainer,
    customer_data: pd.DataFrame,
    pipeline
) -> Dict[str, Any]:
    """Explain a single prediction using SHAP.

    Args:
        explainer: SHAP explainer
        customer_data: Single customer's data
        pipeline: Trained pipeline

    Returns:
        Dictionary with explanation details
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")

    # Preprocess customer data
    X_processed = pipeline.named_steps["preprocessor"].transform(customer_data)

    # Get prediction
    proba = pipeline.predict_proba(customer_data)[0, 1]

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = shap_values[0]  # Single sample

    # Get feature names
    if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(len(shap_values))]

    # Get feature values
    feature_values = X_processed[0]

    # Create contribution DataFrame
    contributions = pd.DataFrame({
        "feature": feature_names,
        "value": feature_values,
        "contribution": shap_values
    }).sort_values("contribution", ascending=False)

    # Separate positive and negative contributions
    positive = contributions[contributions["contribution"] > 0].head(5)
    negative = contributions[contributions["contribution"] < 0].tail(5)

    explanation = {
        "prediction_probability": float(proba),
        "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else None,
        "positive_contributions": positive.to_dict("records"),
        "negative_contributions": negative.to_dict("records"),
        "all_contributions": contributions.to_dict("records")
    }

    return explanation


def generate_plain_english_explanation(explanation: Dict[str, Any]) -> str:
    """Generate a plain-English explanation from SHAP results.

    Args:
        explanation: SHAP explanation dictionary

    Returns:
        Plain-English explanation string
    """
    proba = explanation["prediction_probability"]
    proba_pct = proba * 100

    intro = f"This customer has a {proba_pct:.1f}% probability of churning. "

    if proba > 0.7:
        risk_level = "high"
    elif proba > 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    intro += f"This is considered {risk_level} risk.\n\n"

    # Top positive factors
    if explanation["positive_contributions"]:
        intro += "Factors increasing churn risk:\n"
        for contrib in explanation["positive_contributions"][:3]:
            feature = contrib["feature"]
            value = contrib["value"]
            impact = contrib["contribution"]
            intro += f"- {feature} (value: {value:.2f}, impact: +{impact:.3f})\n"
        intro += "\n"

    # Top negative factors
    if explanation["negative_contributions"]:
        intro += "Factors decreasing churn risk:\n"
        for contrib in explanation["negative_contributions"][:3]:
            feature = contrib["feature"]
            value = contrib["value"]
            impact = contrib["contribution"]
            intro += f"- {feature} (value: {value:.2f}, impact: {impact:.3f})\n"

    # Caveats
    intro += "\nNote: These explanations describe how the model uses features to make predictions. "
    intro += "They do not prove causation, and the model may be incorrect. "
    intro += "Use these insights to support, not replace, business judgment."

    return intro


def create_shap_summary_plot(
    explainer,
    X_background: pd.DataFrame,
    pipeline
) -> Any:
    """Create a SHAP summary plot.

    Args:
        explainer: SHAP explainer
        X_background: Background dataset
        pipeline: Trained pipeline

    Returns:
        SHAP plot object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")

    X_processed = pipeline.named_steps["preprocessor"].transform(X_background)
    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap.summary_plot(shap_values, X_processed, plot_type="bar")


if __name__ == "__main__":
    # Test SHAP functionality
    pass
