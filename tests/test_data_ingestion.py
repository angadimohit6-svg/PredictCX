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
from src.data.app_data import ensure_customer_data


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


def test_ensure_customer_data_downloads_cleans_and_initializes_database(
    tmp_path, monkeypatch
):
    """A fresh deployment can prepare its public source dataset on first run."""
    raw_df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes"],
        "PaymentMethod": ["Mailed check", "Electronic check"],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": ["600.0", " "],
        "Churn": ["No", "Yes"],
    })

    def write_dataset(output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(output_path, index=False)
        return output_path

    monkeypatch.setattr("src.data.app_data.download_dataset", write_dataset)

    customers = ensure_customer_data(tmp_path)

    assert len(customers) == 1
    assert customers.iloc[0]["customerID"] == "C001"
    assert (tmp_path / "data/processed/telco_customer_churn_clean.csv").exists()
    assert (tmp_path / "data/processed/customer_churn.db").exists()


def test_ensure_customer_data_reuses_existing_cleaned_dataset(tmp_path, monkeypatch):
    """Existing cleaned data avoids an unnecessary network download."""
    processed_dir = tmp_path / "data/processed"
    processed_dir.mkdir(parents=True)
    cleaned_df = pd.DataFrame({
        "customerID": ["C001"],
        "Churn": ["No"],
        "tenure": [12],
        "MonthlyCharges": [50.0],
        "TotalCharges": [600.0],
        "Contract": ["One year"],
        "PaymentMethod": ["Mailed check"],
        "InternetService": ["DSL"],
    })
    cleaned_df.to_csv(
        processed_dir / "telco_customer_churn_clean.csv", index=False
    )

    def unexpected_download(**kwargs):
        raise AssertionError("Should use the existing cleaned dataset")

    monkeypatch.setattr("src.data.app_data.download_dataset", unexpected_download)

    customers = ensure_customer_data(tmp_path)

    assert len(customers) == 1
    assert (processed_dir / "customer_churn.db").exists()
