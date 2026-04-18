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
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")
        config = TrainingConfig(
            artifacts_dir=Path(tmpdir) / "artifacts",
            model_params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "min_samples_leaf": 10,
            },
        )
        train(config)
        yield Path(tmpdir) / "artifacts"


@pytest.fixture(scope="session")
def client(trained_artifacts):
    """Create test client with trained model."""
    import serving.app as app_module

    original = app_module.ARTIFACTS_DIR
    app_module.ARTIFACTS_DIR = trained_artifacts
    from serving.app import app

    with TestClient(app) as c:
        yield c

    app_module.ARTIFACTS_DIR = original
