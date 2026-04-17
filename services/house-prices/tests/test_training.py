import tempfile
from pathlib import Path

import mlflow

from training.config import TrainingConfig
from training.data import get_feature_stats, load_dataset
from training.train import build_pipeline, evaluate, train


def test_load_dataset():
    config = TrainingConfig()
    X_train, X_test, y_train, y_test, feature_names = load_dataset(config)
    assert X_train.shape[1] == 8
    assert len(feature_names) == 8
    assert X_train.shape[0] > X_test.shape[0]


def test_feature_stats():
    config = TrainingConfig()
    X_train, _, _, _, feature_names = load_dataset(config)
    stats = get_feature_stats(X_train, list(feature_names))
    assert len(stats) == 8
    for name in feature_names:
        assert "mean" in stats[name]
        assert "std" in stats[name]


def test_build_pipeline():
    config = TrainingConfig()
    pipeline = build_pipeline(config)
    assert len(pipeline.steps) == 2


def test_evaluate():
    config = TrainingConfig()
    X_train, X_test, y_train, _, _ = load_dataset(config)
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_train, y_train)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.5


def test_train_e2e():
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
        metadata = train(config)
        assert (Path(tmpdir) / "artifacts" / "model.joblib").exists()
        assert (Path(tmpdir) / "artifacts" / "metadata.json").exists()
        assert metadata["test_metrics"]["r2"] > 0.0
