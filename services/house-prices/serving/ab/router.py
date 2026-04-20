"""Routes prediction requests to model variants based on traffic weights."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from .experiment import Experiment, Variant


@dataclass
class VariantMetrics:
    """Tracks prediction metrics for a single variant."""

    count: int = 0
    total_latency_ms: float = 0.0
    prediction_sum: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count > 0 else 0.0

    @property
    def avg_prediction(self) -> float:
        return self.prediction_sum / self.count if self.count > 0 else 0.0


class ABRouter:
    """Routes requests to variants based on weighted random selection."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self._metrics: dict[str, VariantMetrics] = {
            v.name: VariantMetrics() for v in experiment.variants
        }
        self._lock = threading.Lock()

    def select_variant(self, forced_variant: str | None = None) -> Variant:
        """Select a variant for a prediction request.

        Args:
            forced_variant: If set, forces selection of a specific variant
                           (useful for testing or sticky sessions).
        """
        if forced_variant:
            variant = self.experiment.get_variant(forced_variant)
            if variant:
                return variant

        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for v in self.experiment.variants:
            cumulative += v.weight
            if r < cumulative:
                return v

        return self.experiment.variants[-1]

    def record_prediction(
        self, variant_name: str, latency_ms: float, prediction: float
    ) -> None:
        with self._lock:
            m = self._metrics[variant_name]
            m.count += 1
            m.total_latency_ms += latency_ms
            m.prediction_sum += prediction

    def get_metrics(self) -> dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "count": m.count,
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                    "avg_prediction": round(m.avg_prediction, 4),
                }
                for name, m in self._metrics.items()
            }

    def get_summary(self) -> dict:
        metrics = self.get_metrics()
        total = sum(m["count"] for m in metrics.values())
        return {
            "experiment_name": self.experiment.name,
            "variants": [
                {
                    "name": v.name,
                    "weight": v.weight,
                    "model_type": v.metadata.get("model_type", "unknown"),
                    "test_r2": v.metadata.get("test_metrics", {}).get("r2"),
                    **metrics.get(v.name, {}),
                }
                for v in self.experiment.variants
            ],
            "total_predictions": total,
        }
