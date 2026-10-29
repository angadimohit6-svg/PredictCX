"""Script to initialize the SQLite database with cleaned data"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.database.db_operations import initialize_database, get_churn_rate, get_total_customers

# Load cleaned dataset
cleaned_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"

df = pd.read_csv(cleaned_path)
print(f"Loaded {len(df)} cleaned records")

# Initialize database
conn = initialize_database(df)
print("Database initialized successfully")

# Test queries
total = get_total_customers()
churn_rate = get_churn_rate()

print(f"\nTotal customers: {total}")
print(f"Overall churn rate: {churn_rate:.2f}%")

conn.close()
print("\nDatabase setup complete")
