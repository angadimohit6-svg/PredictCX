"""Model inference for churn predictions"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.models.training import load_model, get_best_model_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_churn_probability(
    customer_data: pd.DataFrame,
    model_name: Optional[str] = None
) -> np.ndarray:
    """Predict churn probability for customer(s).

    Args:
        customer_data: DataFrame with customer features
        model_name: Name of model to use (default: best model)

    Returns:
        Array of churn probabilities
    """
    if model_name is None:
        model_name = get_best_model_name()

    pipeline, metadata = load_model(model_name)

    # Ensure customer_data has the required columns
    # This will be handled by the pipeline's preprocessing

    # Get probability predictions
    proba = pipeline.predict_proba(customer_data)[:, 1]

    logger.info(f"Generated {len(proba)} predictions using {model_name}")

    return proba


def predict_with_risk_band(
    customer_data: pd.DataFrame,
    model_name: Optional[str] = None,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4
) -> pd.DataFrame:
    """Predict churn probability and assign risk bands.

    Args:
        customer_data: DataFrame with customer features
        model_name: Name of model to use
        high_threshold: Threshold for high-risk classification
        medium_threshold: Threshold for medium-risk classification

    Returns:
        DataFrame with probabilities and risk bands
    """
    proba = predict_churn_probability(customer_data, model_name)

    # Assign risk bands
    risk_bands = []
    for p in proba:
        if p >= high_threshold:
            risk_bands.append("High")
        elif p >= medium_threshold:
            risk_bands.append("Medium")
        else:
            risk_bands.append("Low")

    result_df = customer_data.copy()
    result_df["churn_probability"] = proba
    result_df["risk_band"] = risk_bands

    return result_df


def predict_single_customer(
    customer_features: Dict[str, Any],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Predict churn for a single customer.

    Args:
        customer_features: Dictionary of customer features
        model_name: Name of model to use

    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_features])

    # Get prediction
    proba = predict_churn_probability(df, model_name)[0]

    # Determine risk band
    if proba >= 0.7:
        risk_band = "High"
    elif proba >= 0.4:
        risk_band = "Medium"
    else:
        risk_band = "Low"

    result = {
        "churn_probability": float(proba),
        "risk_band": risk_band,
        "model_used": model_name if model_name else get_best_model_name()
    }

    return result


def batch_predict(
    df: pd.DataFrame,
    model_name: Optional[str] = None
) -> pd.DataFrame:
    """Add churn predictions to a DataFrame of customers.

    Args:
        df: DataFrame with customer features
        model_name: Name of model to use

    Returns:
        DataFrame with added prediction columns
    """
    proba = predict_churn_probability(df, model_name)

    result_df = df.copy()
    result_df["churn_probability"] = proba
    result_df["risk_band"] = pd.cut(
        proba,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    return result_df


if __name__ == "__main__":
    # Test inference
    pass
