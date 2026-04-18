from monitoring.collector import FeatureCollector
from monitoring.drift import compute_drift


FEATURE_NAMES = ["f1", "f2", "f3"]


def test_collector_record():
    c = FeatureCollector(FEATURE_NAMES, window_size=100)
    c.record([1.0, 2.0, 3.0])
    c.record([4.0, 5.0, 6.0])
    assert c.sample_count == 2
    assert c.total_predictions == 2


def test_collector_batch():
    c = FeatureCollector(FEATURE_NAMES, window_size=100)
    c.record_batch([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert c.sample_count == 2
    assert c.total_predictions == 2


def test_collector_window_overflow():
    c = FeatureCollector(FEATURE_NAMES, window_size=5)
    for i in range(10):
        c.record([float(i)] * 3)
    assert c.sample_count == 5
    assert c.total_predictions == 10


def test_collector_stats_too_few():
    c = FeatureCollector(FEATURE_NAMES, window_size=100)
    c.record([1.0, 2.0, 3.0])
    assert c.get_current_stats() is None


def test_collector_stats():
    c = FeatureCollector(FEATURE_NAMES, window_size=100)
    for i in range(20):
        c.record([float(i), float(i * 2), float(i * 3)])
    stats = c.get_current_stats()
    assert stats is not None
    assert set(stats.keys()) == {"f1", "f2", "f3"}
    assert "mean" in stats["f1"]
    assert "std" in stats["f1"]


def test_drift_no_drift():
    training = {"f1": {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0, "median": 5.0}}
    current = {"f1": {"mean": 5.1, "std": 2.1, "min": 0.5, "max": 9.5, "median": 5.0}}
    result = compute_drift(training, current)
    assert result["overall_status"] == "none"
    assert result["features"]["f1"]["status"] == "none"


def test_drift_high_mean_shift():
    training = {"f1": {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0, "median": 5.0}}
    current = {
        "f1": {"mean": 10.0, "std": 2.0, "min": 5.0, "max": 15.0, "median": 10.0}
    }
    result = compute_drift(training, current)
    assert result["features"]["f1"]["status"] == "high"
    assert result["drifted_features"] == 1


def test_drift_std_change():
    training = {"f1": {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0, "median": 5.0}}
    current = {"f1": {"mean": 5.0, "std": 5.0, "min": -5.0, "max": 15.0, "median": 5.0}}
    result = compute_drift(training, current)
    assert result["features"]["f1"]["status"] == "high"


def test_drift_range_breach():
    training = {"f1": {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0, "median": 5.0}}
    current = {"f1": {"mean": 5.0, "std": 2.0, "min": -5.0, "max": 10.0, "median": 5.0}}
    result = compute_drift(training, current)
    assert result["features"]["f1"]["range_breach"] is True


def test_drift_endpoint(client):
    """Test the monitoring endpoint returns after some predictions."""
    # Make a few predictions first
    features = [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]
    for _ in range(15):
        client.post("/predict", json={"features": features})

    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predictions"] >= 15
    assert data["buffer_samples"] >= 15
    assert data["drift"] is not None
    assert data["drift"]["total_features"] == 8
