"""Tests for database operations"""
import pytest
import pandas as pd
import sqlite3
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_operations import (
    initialize_database,
    get_total_customers,
    get_churn_rate,
    execute_query
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create test data
    df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "Churn": ["Yes", "No", "Yes", "No"],
        "tenure": [12, 24, 6, 36],
        "MonthlyCharges": [50.0, 80.0, 45.0, 90.0],
        "TotalCharges": [600.0, 1920.0, 270.0, 3240.0],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Credit card"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"]
    })

    # Initialize database
    conn = initialize_database(df, db_path)
    conn.close()

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


def test_initialize_database(temp_db):
    """Test database initialization"""
    assert temp_db.exists()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
    result = cursor.fetchone()
    assert result is not None

    conn.close()


def test_get_total_customers(temp_db):
    """Test getting total customer count"""
    total = get_total_customers(temp_db)
    assert total == 4


def test_get_churn_rate(temp_db):
    """Test getting churn rate"""
    churn_rate = get_churn_rate(temp_db)
    assert churn_rate == 50.0  # 2 out of 4


def test_execute_query(temp_db):
    """Test executing custom query"""
    query = "SELECT COUNT(*) as count FROM customers WHERE Churn = 'Yes'"
    result = execute_query(query, db_path=temp_db)

    assert len(result) == 1
    assert result["count"].iloc[0] == 2
