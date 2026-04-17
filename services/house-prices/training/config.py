from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # Data
    test_size: float = 0.2
    random_state: int = 42

    # Model
    model_type: str = "gradient_boosting"
    model_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "min_samples_leaf": 10,
        }
    )

    # Paths
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "model.joblib"
    metadata_filename: str = "metadata.json"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename
