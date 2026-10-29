"""Script to train and evaluate ML models"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.training import train_and_evaluate_all

# Load cleaned dataset
cleaned_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"
df = pd.read_csv(cleaned_path)

print(f"Loaded {len(df)} cleaned records")
print("\nTraining models...")

# Train and evaluate all models
results = train_and_evaluate_all(df, save_artifacts=True)

print("\n=== Training Results ===")
print(f"Best model: {results['best_model']}")

for model_name, model_results in results["models"].items():
    print(f"\n{model_name}:")
    print(f"  CV ROC-AUC: {model_results['cv_metrics']['cv_mean']:.4f} (+/- {model_results['cv_metrics']['cv_std']:.4f})")
    print(f"  Test ROC-AUC: {model_results['test_metrics']['roc_auc']:.4f}")
    print(f"  Test F1: {model_results['test_metrics']['f1']:.4f}")
    print(f"  Test Precision: {model_results['test_metrics']['precision']:.4f}")
    print(f"  Test Recall: {model_results['test_metrics']['recall']:.4f}")

print("\nModel training complete!")
