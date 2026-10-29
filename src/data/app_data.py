"""Prepare and cache customer data for Streamlit pages."""
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.database.db_operations import initialize_database
from src.data.ingestion import (
    clean_dataset,
    download_dataset,
    load_dataset,
    save_cleaned_dataset,
    validate_dataset,
)


class DatasetSetupError(RuntimeError):
    """Raised when the app cannot prepare a valid customer dataset."""


def ensure_customer_data(project_root: Path) -> pd.DataFrame:
    """Load the cleaned dataset, creating it and the SQLite database if needed."""
    data_dir = project_root / "data"
    raw_path = data_dir / "raw" / "telco_customer_churn.csv"
    cleaned_path = data_dir / "processed" / "telco_customer_churn_clean.csv"
    database_path = data_dir / "processed" / "customer_churn.db"

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        is_valid, report = validate_dataset(df)
        if not is_valid:
            raise DatasetSetupError(
                f"The cleaned customer dataset is invalid: {report['issues']}"
            )
    else:
        if raw_path.exists():
            df_raw = load_dataset(raw_path)
        else:
            try:
                download_dataset(output_path=raw_path)
            except requests.RequestException as exc:
                raise DatasetSetupError(
                    "The customer dataset is missing and could not be downloaded. "
                    "Check that the deployment can access the internet, then restart "
                    "the app. Dataset source: IBM Telco Customer Churn."
                ) from exc
            df_raw = load_dataset(raw_path)

        is_valid, report = validate_dataset(df_raw)
        missing_columns = [
            issue for issue in report["issues"]
            if issue.startswith("Missing required columns:")
        ]
        if missing_columns:
            raise DatasetSetupError(
                f"The downloaded customer dataset has an unexpected schema: "
                f"{missing_columns}"
            )

        df, _ = clean_dataset(df_raw)
        if df.empty:
            raise DatasetSetupError(
                "The customer dataset contains no usable records after cleaning."
            )
        save_cleaned_dataset(df, output_path=cleaned_path)

    if not database_path.exists():
        initialize_database(df, db_path=database_path).close()

    return df


@st.cache_data(show_spinner="Preparing customer dataset...")
def load_customer_data(project_root: str) -> pd.DataFrame:
    """Cached Streamlit entry point for the customer dataset."""
    return ensure_customer_data(Path(project_root))


def load_customer_data_or_stop(project_root: Path) -> pd.DataFrame:
    """Load customer data and show an actionable message if setup fails."""
    try:
        return load_customer_data(str(project_root))
    except DatasetSetupError as exc:
        st.error(str(exc))
        st.stop()
