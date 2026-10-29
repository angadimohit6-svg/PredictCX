"""Tests for preparing model artifacts on fresh deployments."""
import json

from src.models import deployment


def test_ensure_model_artifacts_trains_when_models_are_missing(
    tmp_path, monkeypatch
):
    models_path = tmp_path / "models"
    customer_data = object()
    monkeypatch.setattr(
        deployment,
        "load_customer_data",
        lambda project_root: customer_data,
    )
    training_calls = []

    def fake_train(data, save_artifacts):
        assert data is customer_data
        assert save_artifacts is True
        training_calls.append(True)
        models_path.mkdir(parents=True, exist_ok=True)
        (models_path / "logistic_regression_pipeline.joblib").touch()
        (models_path / "logistic_regression_metadata.json").touch()
        (models_path / "training_results.json").write_text(json.dumps({
            "best_model": "logistic_regression"
        }))
        return {"best_model": "logistic_regression"}

    monkeypatch.setattr(deployment, "train_and_evaluate_all", fake_train)

    model_name = deployment._ensure_model_artifacts(tmp_path)

    assert model_name == "logistic_regression"
    assert training_calls == [True]


def test_ensure_model_artifacts_reuses_complete_artifacts(tmp_path, monkeypatch):
    models_path = tmp_path / "models"
    models_path.mkdir()
    (models_path / "training_results.json").write_text(json.dumps({
        "best_model": "logistic_regression"
    }))
    (models_path / "logistic_regression_pipeline.joblib").touch()
    (models_path / "logistic_regression_metadata.json").touch()

    def unexpected_train(*args, **kwargs):
        raise AssertionError("Complete model artifacts should be reused")

    monkeypatch.setattr(deployment, "train_and_evaluate_all", unexpected_train)

    assert deployment._ensure_model_artifacts(tmp_path) == "logistic_regression"
