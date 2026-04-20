"""FastAPI model serving for house price prediction with A/B testing."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException

from monitoring.collector import FeatureCollector
from monitoring.drift import compute_drift

from .ab.experiment import load_experiment
from .ab.router import ABRouter
from .schemas import (
    ABExperimentResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringResponse,
    PredictionRequest,
    PredictionResponse,
)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    experiment = load_experiment(ARTIFACTS_DIR)

    # Use the first variant's metadata for shared config (feature names, etc.)
    primary = experiment.variants[0]

    _state["experiment"] = experiment
    _state["router"] = ABRouter(experiment)
    _state["model"] = primary.model
    _state["metadata"] = primary.metadata
    _state["collector"] = FeatureCollector(
        feature_names=primary.metadata["feature_names"],
        window_size=1000,
    )

    variant_names = ", ".join(f"{v.name} ({v.weight:.0%})" for v in experiment.variants)
    print(f"Experiment '{experiment.name}' loaded: {variant_names}")
    yield
    _state.clear()


app = FastAPI(
    title="House Price Prediction API",
    version="2.0.0",
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
def predict(
    req: PredictionRequest,
    x_variant: Optional[str] = Header(None, description="Force a specific variant"),
):
    meta = _state["metadata"]
    expected = len(meta["feature_names"])
    if len(req.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(req.features)}",
        )

    router: ABRouter = _state["router"]
    variant = router.select_variant(forced_variant=x_variant)

    start = time.perf_counter()
    X = np.array(req.features).reshape(1, -1)
    prediction = float(variant.model.predict(X)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    router.record_prediction(variant.name, latency_ms, prediction)
    _state["collector"].record(req.features)
    return PredictionResponse(prediction=prediction, variant=variant.name)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    req: BatchPredictionRequest,
    x_variant: Optional[str] = Header(None, description="Force a specific variant"),
):
    meta = _state["metadata"]
    expected = len(meta["feature_names"])

    for i, row in enumerate(req.instances):
        if len(row) != expected:
            raise HTTPException(
                status_code=422,
                detail=f"Instance {i}: expected {expected} features, got {len(row)}",
            )

    router: ABRouter = _state["router"]
    variant = router.select_variant(forced_variant=x_variant)

    start = time.perf_counter()
    X = np.array(req.instances)
    predictions = variant.model.predict(X).tolist()
    latency_ms = (time.perf_counter() - start) * 1000

    avg_pred = sum(predictions) / len(predictions) if predictions else 0.0
    router.record_prediction(variant.name, latency_ms, avg_pred)
    _state["collector"].record_batch(req.instances)
    return BatchPredictionResponse(predictions=predictions, variant=variant.name)


@app.get("/ab/experiment", response_model=ABExperimentResponse)
def ab_experiment():
    """Get current A/B experiment status and per-variant metrics."""
    router: ABRouter = _state["router"]
    summary = router.get_summary()
    return ABExperimentResponse(**summary)


@app.put("/ab/weights")
def ab_update_weights(weights: dict[str, float]):
    """Update traffic split between variants. Weights must sum to 1.0."""
    router: ABRouter = _state["router"]
    try:
        router.experiment.update_weights(weights)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "updated", "weights": weights}


@app.get("/monitoring/drift", response_model=MonitoringResponse)
def monitoring_drift():
    collector = _state["collector"]
    meta = _state["metadata"]

    current_stats = collector.get_current_stats()
    drift = None
    if current_stats and "feature_stats" in meta:
        drift_result = compute_drift(meta["feature_stats"], current_stats)
        drift = DriftResponse(**drift_result)

    return MonitoringResponse(
        total_predictions=collector.total_predictions,
        buffer_samples=collector.sample_count,
        uptime_seconds=round(collector.uptime_seconds, 1),
        drift=drift,
    )
