"""Data ingestion and validation for IBM Telco Customer Churn dataset"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATASET_ATTRIBUTION = "IBM Telco Customer Churn Dataset (https://github.com/IBM/telco-customer-churn-on-icp4d)"


def download_dataset(
    url: str = DATASET_URL,
    output_path: Optional[Path] = None
) -> Path:
    """Download the dataset from the provided URL.

    Args:
        url: URL to download the dataset from
        output_path: Path to save the downloaded file

    Returns:
        Path to the downloaded file
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "data" / "raw" / "telco_customer_churn.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Dataset saved to {output_path}")
    return output_path


def load_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the raw dataset
    """
    if file_path is None:
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "data" / "raw" / "telco_customer_churn.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Please download it first using download_dataset()."
        )

    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, dict]:
    """Validate the dataset structure and content.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "issues": []
    }

    # Check for required columns
    required_columns = [
        "customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges",
        "Contract", "PaymentMethod", "InternetService"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        report["issues"].append(f"Missing required columns: {missing_columns}")

    # Check for TotalCharges non-numeric values
    if "TotalCharges" in df.columns:
        non_numeric = df["TotalCharges"].apply(
            lambda x: not str(x).replace(".", "", 1).isdigit() if pd.notna(x) else False
        ).sum()
        if non_numeric > 0:
            report["issues"].append(
                f"Found {non_numeric} non-numeric values in TotalCharges column"
            )

    is_valid = len(report["issues"]) == 0
    return is_valid, report


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean the dataset according to documented rules.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (cleaned_df, cleaning_report)
    """
    report = {
        "original_rows": len(df),
        "rows_removed": 0,
        "changes": []
    }

    df_clean = df.copy()

    # Handle TotalCharges: replace empty strings with NaN, then convert to numeric
    if "TotalCharges" in df_clean.columns:
        report["changes"].append("Converted TotalCharges to numeric")
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )

    # Remove rows with missing critical values
    critical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    before_clean = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_columns)
    removed = before_clean - len(df_clean)
    if removed > 0:
        report["rows_removed"] += removed
        report["changes"].append(
            f"Removed {removed} rows with missing values in critical columns"
        )

    # Remove duplicate rows (excluding customerID)
    if "customerID" in df_clean.columns:
        customer_cols = [col for col in df_clean.columns if col != "customerID"]
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=customer_cols)
        removed_dedup = before_dedup - len(df_clean)
        if removed_dedup > 0:
            report["rows_removed"] += removed_dedup
            report["changes"].append(
                f"Removed {removed_dedup} duplicate rows (excluding customerID)"
            )

    report["final_rows"] = len(df_clean)
    logger.info(f"Cleaning complete: {report}")

    return df_clean, report


def save_cleaned_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save the cleaned dataset to a CSV file.

    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned file

    Returns:
        Path to the saved file
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to {output_path}")

    return output_path


def run_ingestion_pipeline() -> Tuple[pd.DataFrame, dict]:
    """Run the complete ingestion pipeline.

    Returns:
        Tuple of (cleaned_df, pipeline_report)
    """
    pipeline_report = {
        "dataset_attribution": DATASET_ATTRIBUTION,
        "steps": []
    }

    # Download dataset
    try:
        raw_path = download_dataset()
        pipeline_report["steps"].append("Dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    # Load dataset
    df = load_dataset(raw_path)
    pipeline_report["steps"].append(f"Dataset loaded: {len(df)} rows")

    # Validate dataset
    is_valid, validation_report = validate_dataset(df)
    pipeline_report["validation"] = validation_report

    if not is_valid:
        logger.warning(f"Dataset validation issues: {validation_report['issues']}")
        pipeline_report["steps"].append(f"Validation warnings: {validation_report['issues']}")
    else:
        pipeline_report["steps"].append("Dataset validation passed")

    # Clean dataset
    df_clean, cleaning_report = clean_dataset(df)
    pipeline_report["cleaning"] = cleaning_report
    pipeline_report["steps"].append(f"Dataset cleaned: {cleaning_report['final_rows']} rows")

    # Save cleaned dataset
    cleaned_path = save_cleaned_dataset(df_clean)
    pipeline_report["steps"].append(f"Cleaned dataset saved to {cleaned_path}")

    return df_clean, pipeline_report


if __name__ == "__main__":
    df, report = run_ingestion_pipeline()
    print("\n=== Ingestion Pipeline Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")
