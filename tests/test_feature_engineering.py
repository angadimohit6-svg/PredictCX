"""Tests for feature engineering module"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.engineering import (
    create_feature_columns,
    encode_target,
    create_preprocessing_pipeline,
    engineer_features,
    prepare_features_and_target
)


def test_create_feature_columns():
    """Test feature column creation"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "Churn": ["Yes", "No"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 80.0]
    })

    feature_cols = create_feature_columns(df)

    assert "customerID" not in feature_cols
    assert "Churn" not in feature_cols
    assert "tenure" in feature_cols
    assert "MonthlyCharges" in feature_cols


def test_encode_target():
    """Test target encoding"""
    df = pd.DataFrame({
        "Churn": ["Yes", "No", "Yes", "No"]
    })

    df_encoded = encode_target(df)

    assert "Churn_Encoded" in df_encoded.columns
    assert df_encoded["Churn_Encoded"].tolist() == [1, 0, 1, 0]


def test_create_preprocessing_pipeline():
    """Test preprocessing pipeline creation"""
    pipeline = create_preprocessing_pipeline()

    assert pipeline is not None
    assert len(pipeline.transformers) == 2
    # Check that transformers have the expected names
    transformer_names = [t[0] for t in pipeline.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_engineer_features():
    """Test feature engineering"""
    df = pd.DataFrame({
        "tenure": [12, 24, 6],
        "MonthlyCharges": [50.0, 80.0, 45.0],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "No"],
        "DeviceProtection": ["Yes", "Yes", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes"],
        "StreamingMovies": ["No", "Yes", "No"]
    })

    df_engineered = engineer_features(df)

    assert "TenureBand" in df_engineered.columns
    assert "ChargeRatio" in df_engineered.columns
    assert "ServiceCount" in df_engineered.columns


def test_prepare_features_and_target():
    """Test features and target preparation"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "Churn": ["Yes", "No"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 80.0]
    })

    X, y = prepare_features_and_target(df)

    assert X.shape[1] == 2  # tenure and MonthlyCharges
    assert len(y) == 2
    assert y.tolist() == [1, 0]
