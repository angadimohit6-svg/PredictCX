"""SQLite database operations for customer analytics"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import sqlite3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Get the path to the SQLite database.

    Returns:
        Path to the database file
    """
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "processed" / "customer_churn.db"
    return db_path


def initialize_database(df: pd.DataFrame, db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the SQLite database with customer data.

    Args:
        df: DataFrame with customer data
        db_path: Path to the database file

    Returns:
        SQLite connection
    """
    if db_path is None:
        db_path = get_db_path()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Write DataFrame to database
    df.to_sql("customers", conn, if_exists="replace", index=False)

    # Create indexes for common queries
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_contract ON customers(Contract)",
        "CREATE INDEX IF NOT EXISTS idx_payment ON customers(PaymentMethod)",
        "CREATE INDEX IF NOT EXISTS idx_internet ON customers(InternetService)",
        "CREATE INDEX IF NOT EXISTS idx_churn ON customers(Churn)",
        "CREATE INDEX IF NOT EXISTS idx_tenure ON customers(tenure)"
    ]

    for index_sql in indexes:
        conn.execute(index_sql)

    conn.commit()
    logger.info(f"Database initialized at {db_path} with {len(df)} records")

    return conn


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection.

    Args:
        db_path: Path to the database file

    Returns:
        SQLite connection
    """
    if db_path is None:
        db_path = get_db_path()

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            f"Please initialize it first using initialize_database()."
        )

    conn = sqlite3.connect(db_path)
    return conn


def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """Execute a SQL query and return results as DataFrame.

    Args:
        query: SQL query string
        params: Dictionary of parameters for parameterized query
        db_path: Path to the database file

    Returns:
        DataFrame with query results
    """
    conn = get_connection(db_path)

    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        logger.info(f"Query executed successfully, returned {len(df)} rows")
        return df
    finally:
        conn.close()


def get_total_customers(db_path: Optional[Path] = None) -> int:
    """Get total customer count.

    Args:
        db_path: Path to the database file

    Returns:
        Total customer count
    """
    query = "SELECT COUNT(*) as count FROM customers"
    result = execute_query(query, db_path=db_path)
    return result["count"].iloc[0]


def get_churn_rate(db_path: Optional[Path] = None) -> float:
    """Get overall churn rate.

    Args:
        db_path: Path to the database file

    Returns:
        Churn rate as percentage
    """
    query = """
        SELECT
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate
        FROM customers
    """
    result = execute_query(query, db_path=db_path)
    return result["churn_rate"].iloc[0]


def get_churn_by_contract(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get churn statistics by contract type.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with churn by contract
    """
    query = """
        SELECT
            Contract,
            COUNT(*) as total_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate DESC
    """
    return execute_query(query, db_path=db_path)


def get_churn_by_payment_method(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get churn statistics by payment method.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with churn by payment method
    """
    query = """
        SELECT
            PaymentMethod,
            COUNT(*) as total_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate
        FROM customers
        GROUP BY PaymentMethod
        ORDER BY churn_rate DESC
    """
    return execute_query(query, db_path=db_path)


def get_churn_by_internet_service(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get churn statistics by internet service type.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with churn by internet service
    """
    query = """
        SELECT
            InternetService,
            COUNT(*) as total_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_rate
        FROM customers
        GROUP BY InternetService
        ORDER BY churn_rate DESC
    """
    return execute_query(query, db_path=db_path)


def get_tenure_summary(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get tenure summary statistics.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with tenure summary
    """
    query = """
        SELECT
            AVG(tenure) as avg_tenure,
            MIN(tenure) as min_tenure,
            MAX(tenure) as max_tenure,
            AVG(CASE WHEN Churn = 'Yes' THEN tenure END) as avg_tenure_churned,
            AVG(CASE WHEN Churn = 'No' THEN tenure END) as avg_tenure_retained
        FROM customers
    """
    return execute_query(query, db_path=db_path)


def get_monthly_charges_summary(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get monthly charges summary statistics.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with monthly charges summary
    """
    query = """
        SELECT
            AVG(MonthlyCharges) as avg_monthly_charges,
            MIN(MonthlyCharges) as min_monthly_charges,
            MAX(MonthlyCharges) as max_monthly_charges,
            AVG(CASE WHEN Churn = 'Yes' THEN MonthlyCharges END) as avg_charges_churned,
            AVG(CASE WHEN Churn = 'No' THEN MonthlyCharges END) as avg_charges_retained
        FROM customers
    """
    return execute_query(query, db_path=db_path)


def get_customer_segment_counts(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get customer counts by segment (contract + internet service).

    Args:
        db_path: Path to the database file

    Returns:
        DataFrame with segment counts
    """
    query = """
        SELECT
            Contract,
            InternetService,
            COUNT(*) as customer_count,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count
        FROM customers
        GROUP BY Contract, InternetService
        ORDER BY customer_count DESC
    """
    return execute_query(query, db_path=db_path)


def search_customers(
    customer_id: Optional[str] = None,
    contract: Optional[str] = None,
    payment_method: Optional[str] = None,
    internet_service: Optional[str] = None,
    min_tenure: Optional[int] = None,
    max_tenure: Optional[int] = None,
    min_monthly_charges: Optional[float] = None,
    max_monthly_charges: Optional[float] = None,
    churn_status: Optional[str] = None,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """Search customers with optional filters.

    Args:
        customer_id: Customer ID to search for
        contract: Filter by contract type
        payment_method: Filter by payment method
        internet_service: Filter by internet service
        min_tenure: Minimum tenure
        max_tenure: Maximum tenure
        min_monthly_charges: Minimum monthly charges
        max_monthly_charges: Maximum monthly charges
        churn_status: Filter by churn status ('Yes' or 'No')
        db_path: Path to the database file

    Returns:
        DataFrame with filtered customers
    """
    query = "SELECT * FROM customers WHERE 1=1"
    params = {}

    if customer_id:
        query += " AND customerID LIKE :customer_id"
        params["customer_id"] = f"%{customer_id}%"

    if contract:
        query += " AND Contract = :contract"
        params["contract"] = contract

    if payment_method:
        query += " AND PaymentMethod = :payment_method"
        params["payment_method"] = payment_method

    if internet_service:
        query += " AND InternetService = :internet_service"
        params["internet_service"] = internet_service

    if min_tenure is not None:
        query += " AND tenure >= :min_tenure"
        params["min_tenure"] = min_tenure

    if max_tenure is not None:
        query += " AND tenure <= :max_tenure"
        params["max_tenure"] = max_tenure

    if min_monthly_charges is not None:
        query += " AND MonthlyCharges >= :min_monthly_charges"
        params["min_monthly_charges"] = min_monthly_charges

    if max_monthly_charges is not None:
        query += " AND MonthlyCharges <= :max_monthly_charges"
        params["max_monthly_charges"] = max_monthly_charges

    if churn_status:
        query += " AND Churn = :churn_status"
        params["churn_status"] = churn_status

    return execute_query(query, params=params, db_path=db_path)


if __name__ == "__main__":
    # Test database operations
    pass
