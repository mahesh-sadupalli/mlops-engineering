"""Training pipeline for house price prediction model."""

import json
import time

import mlflow
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .data import get_feature_stats, load_dataset

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """Build sklearn pipeline with preprocessing and model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(**config.model_params)),
        ]
    )


def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute regression metrics."""
    preds = pipeline.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def train(config: TrainingConfig | None = None) -> dict:
    """Run the full training pipeline."""
    config = config or TrainingConfig()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(config)
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    pipeline = build_pipeline(config)

    mlflow.set_experiment("house-prices")
    with mlflow.start_run():
        mlflow.log_params(config.model_params)
        mlflow.log_param("model_type", config.model_type)

        print("Training...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        train_metrics = evaluate(pipeline, X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_metric("train_time_seconds", train_time)
        for split, metrics in [("train", train_metrics), ("test", test_metrics)]:
            for name, value in metrics.items():
                mlflow.log_metric(f"{split}_{name}", value)

        print(
            f"Train RMSE: {train_metrics['rmse']:.4f} | R²: {train_metrics['r2']:.4f}"
        )
        print(f"Test  RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f}")

        # Save model
        joblib.dump(pipeline, config.model_path)
        mlflow.log_artifact(str(config.model_path))

        # Save metadata
        metadata = {
            "feature_names": list(feature_names),
            "model_type": config.model_type,
            "model_params": config.model_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "train_time_seconds": round(train_time, 2),
            "feature_stats": get_feature_stats(X_train, list(feature_names)),
        }
        with open(config.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(config.metadata_path))

    print(f"Artifacts saved to {config.artifacts_dir}/")
    return metadata


if __name__ == "__main__":
    train()
