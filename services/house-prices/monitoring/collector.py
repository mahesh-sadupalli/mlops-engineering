"""Collects incoming prediction features for drift analysis."""

import threading
import time
from collections import deque

import numpy as np


class FeatureCollector:
    """Thread-safe ring buffer that stores recent prediction features."""

    def __init__(self, feature_names: list[str], window_size: int = 1000):
        self.feature_names = feature_names
        self.window_size = window_size
        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total_predictions = 0
        self._start_time = time.time()

    def record(self, features: list[float]) -> None:
        with self._lock:
            self._buffer.append(features)
            self._total_predictions += 1

    def record_batch(self, instances: list[list[float]]) -> None:
        with self._lock:
            for row in instances:
                self._buffer.append(row)
            self._total_predictions += len(instances)

    def get_current_stats(self) -> dict[str, dict[str, float]] | None:
        """Compute feature statistics from the current buffer."""
        with self._lock:
            if len(self._buffer) < 10:
                return None
            data = np.array(list(self._buffer))

        stats = {}
        for i, name in enumerate(self.feature_names):
            col = data[:, i]
            stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "median": float(np.median(col)),
            }
        return stats

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
