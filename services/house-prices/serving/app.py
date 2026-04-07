"""FastAPI model serving for house price prediction."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ARTIFACTS_DIR / "model.joblib"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run training first.")

    _state["model"] = joblib.load(model_path)
    with open(metadata_path) as f:
        _state["metadata"] = json.load(f)

    print(f"Model loaded from {model_path}")
    yield
    _state.clear()


app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded="model" in _state,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    meta = _state["metadata"]
    return ModelInfoResponse(
        model_type=meta["model_type"],
        feature_names=meta["feature_names"],
        test_metrics=meta["test_metrics"],
        train_samples=meta["train_samples"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    meta = _state["metadata"]
    expected = len(meta["feature_names"])
    if len(req.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(req.features)}",
        )

    X = np.array(req.features).reshape(1, -1)
    prediction = float(_state["model"].predict(X)[0])
    return PredictionResponse(prediction=prediction)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    meta = _state["metadata"]
    expected = len(meta["feature_names"])

    for i, row in enumerate(req.instances):
        if len(row) != expected:
            raise HTTPException(
                status_code=422,
                detail=f"Instance {i}: expected {expected} features, got {len(row)}",
            )

    X = np.array(req.instances)
    predictions = _state["model"].predict(X).tolist()
    return BatchPredictionResponse(predictions=predictions)
