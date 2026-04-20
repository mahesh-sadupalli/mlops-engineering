from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_names: list[str]
    test_metrics: dict[str, float]
    train_samples: int


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Feature values in order: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]}
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted house price in $100k units")
    variant: str = Field(
        "control", description="Model variant that served this prediction"
    )


class BatchPredictionRequest(BaseModel):
    instances: list[list[float]]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    variant: str = Field(
        "control", description="Model variant that served this prediction"
    )


class ABVariantInfo(BaseModel):
    name: str
    weight: float
    model_type: str
    test_r2: float | None = None
    count: int = 0
    avg_latency_ms: float = 0.0
    avg_prediction: float = 0.0


class ABExperimentResponse(BaseModel):
    experiment_name: str
    variants: list[ABVariantInfo]
    total_predictions: int


class DriftFeatureDetail(BaseModel):
    mean_shift: float
    std_ratio: float
    range_breach: bool
    status: str
    training_mean: float
    current_mean: float


class DriftResponse(BaseModel):
    overall_status: str
    drifted_features: int
    total_features: int
    features: dict[str, DriftFeatureDetail]


class MonitoringResponse(BaseModel):
    total_predictions: int
    buffer_samples: int
    uptime_seconds: float
    drift: DriftResponse | None = None
