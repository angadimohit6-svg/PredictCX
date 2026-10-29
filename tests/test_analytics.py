"""Tests for analytics module"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.metrics import (
    calculate_kpis,
    calculate_churn_by_segment,
    calculate_retention_scenario
)


def test_calculate_kpis():
    """Test KPI calculation"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "Churn": ["Yes", "No", "Yes", "No"],
        "tenure": [12, 24, 6, 36],
        "MonthlyCharges": [50.0, 80.0, 45.0, 90.0],
        "TotalCharges": [600.0, 1920.0, 270.0, 3240.0]
    })

    kpis = calculate_kpis(df)

    assert kpis["total_customers"] == 4
    assert kpis["churned_customers"] == 2
    assert kpis["retained_customers"] == 2
    assert kpis["churn_rate"] == 50.0
    assert kpis["avg_tenure"] == 19.5
    assert kpis["avg_monthly_charges"] == 66.25


def test_calculate_churn_by_segment():
    """Test churn calculation by segment"""
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "Churn": ["Yes", "No", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"]
    })

    segment_churn = calculate_churn_by_segment(df, "Contract")

    assert "Month-to-month" in segment_churn.index
    assert "total_customers" in segment_churn.columns
    assert "churn_rate" in segment_churn.columns


def test_calculate_retention_scenario():
    """Test retention scenario calculation"""
    scenario = calculate_retention_scenario(
        eligible_customers=1000,
        campaign_reach=0.8,
        incremental_retention_rate=0.15,
        contact_cost_per_customer=10.0,
        contribution_margin_per_retained=500.0
    )

    assert scenario["customers_targeted"] == 1000
    assert scenario["customers_reached"] == 800
    assert scenario["incremental_retained"] == 120
    assert scenario["campaign_cost"] == 8000.0
    assert scenario["estimated_value"] == 60000.0
    assert scenario["net_value"] == 52000.0
    assert scenario["break_even_rate"] == 0.02  # 10/500
