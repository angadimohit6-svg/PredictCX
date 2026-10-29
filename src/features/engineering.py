"""Feature engineering for churn prediction"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Columns to exclude from features
ID_COLUMNS = ["customerID"]
TARGET_COLUMN = "Churn"

# Categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Numerical columns for scaling
NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def create_feature_columns(df: pd.DataFrame) -> List[str]:
    """Create the list of feature columns excluding IDs and target.

    Args:
        df: Input DataFrame

    Returns:
        List of feature column names
    """
    feature_cols = [
        col for col in df.columns
        if col not in ID_COLUMNS and col != TARGET_COLUMN
    ]
    return feature_cols


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Encode the target variable (Churn) to binary.

    Args:
        df: Input DataFrame
        target_col: Name of the target column

    Returns:
        DataFrame with encoded target column
    """
    df = df.copy()
    df["Churn_Encoded"] = df[target_col].map({"Yes": 1, "No": 0})
    return df


def create_preprocessing_pipeline(
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """Create a preprocessing pipeline for feature engineering.

    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLUMNS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLUMNS

    # Filter to only include columns that exist in the data
    categorical_cols = [col for col in categorical_cols if col in categorical_cols]
    numerical_cols = [col for col in numerical_cols if col in numerical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )

    logger.info(f"Preprocessing pipeline created for {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
    return preprocessor


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Create tenure bands
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72", "72+"]
    )

    # Create charge ratio (MonthlyCharges / tenure, handling division by zero)
    df["ChargeRatio"] = np.where(
        df["tenure"] > 0,
        df["MonthlyCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Count additional services
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    service_cols = [col for col in service_cols if col in df.columns]

    if service_cols:
        df["ServiceCount"] = df[service_cols].apply(
            lambda row: sum(row == "Yes"), axis=1
        )

    logger.info("Feature engineering complete")
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN
) -> tuple:
    """Prepare features and target for model training.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Name of the target column

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if feature_cols is None:
        feature_cols = create_feature_columns(df)

    # Filter to only include columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].map({"Yes": 1, "No": 0})

    logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
    return X, y
