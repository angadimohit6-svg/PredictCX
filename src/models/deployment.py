"""First-run model setup for Streamlit deployments."""
from pathlib import Path
from typing import Any

import streamlit as st

from src.data.app_data import load_customer_data
from src.models.training import train_and_evaluate_all


def _ensure_model_artifacts(project_root: Path) -> str:
    models_path = project_root / "models"
    results_path = models_path / "training_results.json"

    if results_path.exists():
        import json

        with results_path.open() as results_file:
            results: dict[str, Any] = json.load(results_file)
        best_model = results.get("best_model")
        if best_model and all(
            (models_path / f"{best_model}_{suffix}").exists()
            for suffix in ("pipeline.joblib", "metadata.json")
        ):
            return best_model

    customer_data = load_customer_data(str(project_root))
    results = train_and_evaluate_all(customer_data, save_artifacts=True)
    return results["best_model"]


@st.cache_resource(show_spinner="Training churn models for this deployment...")
def ensure_model_artifacts(project_root: str, dataset_version: int) -> str:
    """Train and persist models once when deployment artifacts are absent."""
    del dataset_version
    return _ensure_model_artifacts(Path(project_root))


def ensure_model_artifacts_or_stop(project_root: Path) -> str:
    """Ensure models are ready and display a clear error if training fails."""
    cleaned_path = (
        project_root / "data" / "processed" / "telco_customer_churn_clean.csv"
    )
    dataset_version = cleaned_path.stat().st_mtime_ns
    try:
        return ensure_model_artifacts(str(project_root), dataset_version)
    except Exception as exc:
        st.error(
            "The churn models could not be prepared for this deployment. "
            f"Open Model Lab to retry training. Details: {exc}"
        )
        st.stop()
