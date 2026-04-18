SAMPLE_FEATURES = [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == "gradient_boosting"
    assert len(data["feature_names"]) == 8


def test_predict(client):
    resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0


def test_predict_wrong_feature_count(client):
    resp = client.post("/predict", json={"features": [1.0, 2.0]})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        "/predict/batch", json={"instances": [SAMPLE_FEATURES, SAMPLE_FEATURES]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
