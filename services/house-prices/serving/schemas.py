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


class BatchPredictionRequest(BaseModel):
    instances: list[list[float]]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
