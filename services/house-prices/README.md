# House Price Prediction Service

Production-grade regression service for predicting house prices using the California Housing dataset.

**Python** for training + serving | **Go** for high-throughput inference proxy | **Docker** for deployment

## Quick Start

```bash
# Train
python -m training.train

# Serve
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Docker
docker compose up
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
