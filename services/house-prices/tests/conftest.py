import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from training.config import TrainingConfig
from training.train import train


@pytest.fixture(scope="session")
def trained_artifacts():
    """Train a small model for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / "artifacts"
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")

        config = TrainingConfig(
            artifacts_dir=artifacts_dir,
            model_params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "min_samples_leaf": 10,
            },
        )
        train(config)
        yield artifacts_dir


@pytest.fixture(scope="session")
def ab_artifacts(trained_artifacts):
    """Set up A/B experiment with two variants from the same trained model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ab_dir = Path(tmpdir) / "artifacts"
        ab_dir.mkdir()

        for name in ("control", "challenger"):
            variant_dir = ab_dir / name
            variant_dir.mkdir()
            shutil.copy(
                trained_artifacts / "model.joblib", variant_dir / "model.joblib"
            )
            shutil.copy(
                trained_artifacts / "metadata.json", variant_dir / "metadata.json"
            )

        experiment_config = {
            "name": "test-experiment",
            "variants": [
                {"name": "control", "weight": 0.7},
                {"name": "challenger", "weight": 0.3},
            ],
        }
        with open(ab_dir / "experiment.json", "w") as f:
            json.dump(experiment_config, f)

        yield ab_dir


@pytest.fixture(scope="session")
def client(trained_artifacts):
    """Create test client with single model (fallback mode)."""
    import serving.app as app_module

    original = app_module.ARTIFACTS_DIR
    app_module.ARTIFACTS_DIR = trained_artifacts
    from serving.app import app

    with TestClient(app) as c:
        yield c

    app_module.ARTIFACTS_DIR = original
