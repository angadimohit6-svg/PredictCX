"""Model training and evaluation for churn prediction"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. XGBoost models will be skipped.")

from src.features.engineering import create_preprocessing_pipeline, prepare_features_and_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def get_model_artifacts_path() -> Path:
    """Get the path to save model artifacts.

    Returns:
        Path to the models directory
    """
    project_root = Path(__file__).parent.parent.parent
    models_path = project_root / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def create_models() -> Dict[str, Any]:
    """Create a dictionary of models to train.

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
        )
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            scale_pos_weight=1  # Will be adjusted based on class imbalance
        )

    return models


def build_pipeline(model_name: str, model: Any) -> Pipeline:
    """Build a complete pipeline with preprocessing and model.

    Args:
        model_name: Name of the model
        model: Model instance

    Returns:
        sklearn Pipeline
    """
    preprocessor = create_preprocessing_pipeline()

    if model_name == "xgboost" and XGBOOST_AVAILABLE:
        # XGBoost can handle categorical features directly
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate a trained model on test data.

    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    # Get probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Get predictions at threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    metrics["pr_auc"] = auc(recall, precision)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(f"Model evaluation: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")

    return metrics


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Dict[str, float]:
    """Perform cross-validation on training data.

    Args:
        pipeline: Pipeline to evaluate
        X_train: Training features
        y_train: Training target
        cv: Number of CV folds

    Returns:
        Dictionary of CV metrics
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_metrics = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist()
    }

    logger.info(f"Cross-validation: mean={cv_metrics['cv_mean']:.4f}, std={cv_metrics['cv_std']:.4f}")

    return cv_metrics


def train_and_evaluate_all(
    df: pd.DataFrame,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """Train and evaluate all models.

    Args:
        df: Cleaned DataFrame with features and target
        save_artifacts: Whether to save model artifacts

    Returns:
        Dictionary with all model results
    """
    results = {
        "models": {},
        "best_model": None,
        "training_config": {
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "cv_folds": 5
        }
    }

    # Prepare features and target
    X, y = prepare_features_and_target(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create models
    models = create_models()

    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")

        # Build pipeline
        pipeline = build_pipeline(model_name, model)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Cross-validation
        cv_metrics = cross_validate_model(pipeline, X_train, y_train)

        # Evaluate on test set
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        # Store results
        results["models"][model_name] = {
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "pipeline": pipeline
        }

        # Save artifacts if requested
        if save_artifacts:
            save_model_artifacts(model_name, pipeline, test_metrics, cv_metrics)

    # Select best model based on test ROC-AUC
    best_model_name = max(
        results["models"].keys(),
        key=lambda k: results["models"][k]["test_metrics"]["roc_auc"]
    )
    results["best_model"] = best_model_name
    logger.info(f"Best model: {best_model_name}")

    # Save results summary
    if save_artifacts:
        save_training_results(results)

    return results


def save_model_artifacts(
    model_name: str,
    pipeline: Pipeline,
    test_metrics: Dict[str, Any],
    cv_metrics: Dict[str, float]
) -> Path:
    """Save model artifacts to disk.

    Args:
        model_name: Name of the model
        pipeline: Trained pipeline
        test_metrics: Test metrics
        cv_metrics: Cross-validation metrics

    Returns:
        Path to saved artifacts
    """
    models_path = get_model_artifacts_path()
    model_path = models_path / f"{model_name}_pipeline.joblib"

    # Save pipeline
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        "cv_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in cv_metrics.items()},
        "feature_names": pipeline.named_steps["preprocessor"].get_feature_names_out().tolist() if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out") else []
    }

    metadata_path = models_path / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model artifacts saved for {model_name}")

    return model_path


def save_training_results(results: Dict[str, Any]) -> Path:
    """Save training results summary.

    Args:
        results: Training results dictionary

    Returns:
        Path to saved results
    """
    models_path = get_model_artifacts_path()

    # Create a summary without the pipeline objects
    summary = {
        "best_model": results["best_model"],
        "training_config": results["training_config"],
        "models": {
            name: {
                "cv_metrics": model["cv_metrics"],
                "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in model["test_metrics"].items()}
            }
            for name, model in results["models"].items()
        }
    }

    summary_path = models_path / "training_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training results saved to {summary_path}")

    return summary_path


def load_model(model_name: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load a trained model and its metadata.

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (pipeline, metadata)
    """
    models_path = get_model_artifacts_path()
    model_path = models_path / f"{model_name}_pipeline.joblib"
    metadata_path = models_path / f"{model_name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Model loaded: {model_name}")

    return pipeline, metadata


def get_best_model_name() -> str:
    """Get the name of the best model from training results.

    Returns:
        Name of the best model
    """
    models_path = get_model_artifacts_path()
    results_path = models_path / "training_results.json"

    if not results_path.exists():
        raise FileNotFoundError("Training results not found. Please train models first.")

    with open(results_path, "r") as f:
        results = json.load(f)

    return results["best_model"]


if __name__ == "__main__":
    # Test model training
    pass
