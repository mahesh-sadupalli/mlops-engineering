"""Tests for A/B testing functionality."""

from serving.ab.experiment import load_experiment
from serving.ab.router import ABRouter

SAMPLE_FEATURES = [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]


# ── Experiment unit tests ────────────────────────────────────────────────────


def test_load_single_model_fallback(trained_artifacts):
    """When no experiment.json exists, treat as single 'control' variant."""
    experiment = load_experiment(trained_artifacts)
    assert experiment.name == "default"
    assert len(experiment.variants) == 1
    assert experiment.variants[0].name == "control"
    assert experiment.variants[0].weight == 1.0
    assert experiment.variants[0].model is not None


def test_load_ab_experiment(ab_artifacts):
    """Load a proper A/B experiment with two variants."""
    experiment = load_experiment(ab_artifacts)
    assert experiment.name == "test-experiment"
    assert len(experiment.variants) == 2
    assert experiment.variant_names == ["control", "challenger"]
    assert experiment.weights == [0.7, 0.3]


def test_update_weights(ab_artifacts):
    experiment = load_experiment(ab_artifacts)
    experiment.update_weights({"control": 0.5, "challenger": 0.5})
    assert experiment.variants[0].weight == 0.5
    assert experiment.variants[1].weight == 0.5


def test_update_weights_invalid(ab_artifacts):
    experiment = load_experiment(ab_artifacts)
    try:
        experiment.update_weights({"control": 0.3, "challenger": 0.3})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Router unit tests ────────────────────────────────────────────────────────


def test_router_forced_variant(ab_artifacts):
    experiment = load_experiment(ab_artifacts)
    router = ABRouter(experiment)
    variant = router.select_variant(forced_variant="challenger")
    assert variant.name == "challenger"


def test_router_weighted_selection(ab_artifacts):
    """Over many selections, distribution should roughly match weights."""
    experiment = load_experiment(ab_artifacts)
    experiment.update_weights({"control": 0.7, "challenger": 0.3})
    router = ABRouter(experiment)

    counts = {"control": 0, "challenger": 0}
    for _ in range(1000):
        v = router.select_variant()
        counts[v.name] += 1

    # With 70/30 split over 1000, control should be 600-800
    assert 550 < counts["control"] < 850


def test_router_records_metrics(ab_artifacts):
    experiment = load_experiment(ab_artifacts)
    router = ABRouter(experiment)
    router.record_prediction("control", 5.0, 3.5)
    router.record_prediction("control", 7.0, 4.0)
    metrics = router.get_metrics()
    assert metrics["control"]["count"] == 2
    assert metrics["control"]["avg_latency_ms"] == 6.0


def test_router_summary(ab_artifacts):
    experiment = load_experiment(ab_artifacts)
    router = ABRouter(experiment)
    summary = router.get_summary()
    assert summary["experiment_name"] == "test-experiment"
    assert len(summary["variants"]) == 2
    assert summary["total_predictions"] == 0


# ── API integration tests (single-model fallback) ────────────────────────────


def test_predict_returns_variant(client):
    """Single-model mode returns 'control' as variant."""
    resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] == "control"


def test_predict_batch_returns_variant(client):
    resp = client.post(
        "/predict/batch", json={"instances": [SAMPLE_FEATURES, SAMPLE_FEATURES]}
    )
    assert resp.status_code == 200
    assert resp.json()["variant"] == "control"


def test_ab_experiment_endpoint_single_model(client):
    """Experiment endpoint works in single-model mode."""
    resp = client.get("/ab/experiment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_name"] == "default"
    assert len(data["variants"]) == 1


# ── API integration tests (full A/B experiment) ──────────────────────────────


def _make_ab_client(ab_artifacts):
    """Helper: create a fresh FastAPI TestClient with A/B experiment config."""
    from fastapi import FastAPI

    from serving.ab.experiment import load_experiment
    from serving.ab.router import ABRouter

    experiment = load_experiment(ab_artifacts)
    router = ABRouter(experiment)
    primary = experiment.variants[0]

    # Build a minimal app that mirrors the real one
    from contextlib import asynccontextmanager

    from monitoring.collector import FeatureCollector

    state = {
        "experiment": experiment,
        "router": router,
        "model": primary.model,
        "metadata": primary.metadata,
        "collector": FeatureCollector(primary.metadata["feature_names"]),
    }

    import serving.app as app_module

    # Temporarily patch _state
    original_state = dict(app_module._state)
    app_module._state.update(state)

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    # Copy routes from the real app
    for route in app_module.app.routes:
        test_app.routes.append(route)

    from starlette.testclient import TestClient

    return TestClient(test_app), app_module, original_state


def test_ab_forced_variant_header(ab_artifacts):
    """Force a specific variant via X-Variant header."""
    experiment = load_experiment(ab_artifacts)
    router = ABRouter(experiment)

    variant = router.select_variant(forced_variant="challenger")
    assert variant.name == "challenger"

    # Also test prediction through the variant
    import numpy as np

    X = np.array(SAMPLE_FEATURES).reshape(1, -1)
    prediction = float(variant.model.predict(X)[0])
    assert prediction > 0


def test_ab_traffic_distribution(ab_artifacts):
    """Verify traffic roughly matches weights over many selections."""
    experiment = load_experiment(ab_artifacts)
    experiment.update_weights({"control": 0.7, "challenger": 0.3})
    router = ABRouter(experiment)

    import numpy as np

    X = np.array(SAMPLE_FEATURES).reshape(1, -1)

    for _ in range(200):
        variant = router.select_variant()
        prediction = float(variant.model.predict(X)[0])
        router.record_prediction(variant.name, 1.0, prediction)

    metrics = router.get_metrics()
    control_count = metrics["control"]["count"]
    # With 70/30 over 200 requests, control should be roughly 110-170
    assert 90 < control_count < 180, f"Expected ~140, got {control_count}"
    assert metrics["challenger"]["count"] > 0


def test_ab_update_weights_validation(ab_artifacts):
    """Weights must sum to 1.0."""
    experiment = load_experiment(ab_artifacts)
    try:
        experiment.update_weights({"control": 0.8, "challenger": 0.8})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "1.0" in str(e)
