"""Tests for data ingestion module"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ingestion import validate_dataset, clean_dataset


def test_validate_dataset():
    """Test dataset validation"""
    # Create a valid sample dataset
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "Churn": ["Yes", "No", "Yes"],
        "tenure": [12, 24, 6],
        "MonthlyCharges": [50.0, 80.0, 45.0],
        "TotalCharges": [600.0, 1920.0, 270.0],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "InternetService": ["DSL", "Fiber optic", "DSL"]
    })

    is_valid, report = validate_dataset(df)

    assert is_valid == True
    assert report["row_count"] == 3
    assert report["missing_values"]["customerID"] == 0


def test_validate_dataset_missing_columns():
    """Test validation with missing required columns"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "Churn": ["Yes", "No"]
    })

    is_valid, report = validate_dataset(df)

    assert is_valid == False
    assert len(report["issues"]) > 0


def test_clean_dataset():
    """Test dataset cleaning"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "Churn": ["Yes", "No", "Yes", "No"],
        "tenure": [12, 24, 6, 36],
        "MonthlyCharges": [50.0, 80.0, 45.0, 90.0],
        "TotalCharges": [600.0, " ", 270.0, 3240.0],  # One space (non-numeric)
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Credit card"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"]
    })

    df_clean, report = clean_dataset(df)

    assert report["original_rows"] == 4
    assert report["rows_removed"] >= 1  # At least the row with missing TotalCharges
    assert "Converted TotalCharges to numeric" in report["changes"]


def test_clean_dataset_no_duplicates():
    """Test cleaning without duplicates"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "Churn": ["Yes", "No", "Yes"],
        "tenure": [12, 24, 6],
        "MonthlyCharges": [50.0, 80.0, 45.0],
        "TotalCharges": [600.0, 1920.0, 270.0],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "InternetService": ["DSL", "Fiber optic", "DSL"]
    })

    df_clean, report = clean_dataset(df)

    assert report["rows_removed"] == 0
