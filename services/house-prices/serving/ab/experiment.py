"""A/B experiment configuration and variant management."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


@dataclass
class Variant:
    """A single model variant in an A/B experiment."""

    name: str
    model_path: Path
    metadata_path: Path
    weight: float  # Traffic percentage (0.0 - 1.0)
    model: object = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict, repr=False)

    def load(self) -> None:
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)


@dataclass
class Experiment:
    """An A/B experiment with two or more model variants."""

    name: str
    variants: list[Variant]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load_all(self) -> None:
        for v in self.variants:
            v.load()

    def get_variant(self, name: str) -> Variant | None:
        for v in self.variants:
            if v.name == name:
                return v
        return None

    @property
    def weights(self) -> list[float]:
        return [v.weight for v in self.variants]

    @property
    def variant_names(self) -> list[str]:
        return [v.name for v in self.variants]

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update traffic split. Weights must sum to 1.0."""
        with self._lock:
            total = sum(new_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            for v in self.variants:
                if v.name in new_weights:
                    v.weight = new_weights[v.name]


def load_experiment(artifacts_dir: Path) -> Experiment:
    """Load an A/B experiment from the artifacts directory.

    Expects:
      artifacts_dir/
        experiment.json          # Experiment config
        <variant_name>/
          model.joblib
          metadata.json
    Falls back to single-model mode if no experiment.json exists.
    """
    config_path = artifacts_dir / "experiment.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        variants = []
        for v_config in config["variants"]:
            variant_dir = artifacts_dir / v_config["name"]
            variants.append(
                Variant(
                    name=v_config["name"],
                    model_path=variant_dir / "model.joblib",
                    metadata_path=variant_dir / "metadata.json",
                    weight=v_config["weight"],
                )
            )
        experiment = Experiment(name=config["name"], variants=variants)
    else:
        # Single model fallback: treat existing model as "control"
        variant = Variant(
            name="control",
            model_path=artifacts_dir / "model.joblib",
            metadata_path=artifacts_dir / "metadata.json",
            weight=1.0,
        )
        experiment = Experiment(name="default", variants=[variant])

    experiment.load_all()
    return experiment
