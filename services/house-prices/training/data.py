import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import TrainingConfig


def load_dataset(config: TrainingConfig):
    """Load California Housing dataset and split into train/test."""
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    return X_train, X_test, y_train, y_test, data.feature_names


def get_feature_stats(X: np.ndarray, feature_names: list[str]) -> dict:
    """Compute per-feature statistics for monitoring drift."""
    stats = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
        }
    return stats
