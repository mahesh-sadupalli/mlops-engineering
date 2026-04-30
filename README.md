<div align="center">

# MLOps Engineering

**Production-grade ML systems built with Python and Go, deployed with containers and Kubernetes.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![CI](https://github.com/mahesh-sadupalli/mlops-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/mahesh-sadupalli/mlops-engineering/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

<br>

## The Idea

Most ML projects stop at a Jupyter notebook. This one doesn't.

This repository takes a house price prediction model from raw data all the way to a production system — with a Go proxy for high-throughput inference, A/B testing for model variants, drift detection for incoming data, Kubernetes manifests for orchestration, and a CI/CD pipeline that gates every push.

The goal is simple: **show what it actually takes to ship an ML model**, not just train one.

---

## Architecture

```mermaid
flowchart TB
    subgraph ingress ["Internet"]
        C([Client])
    end

    subgraph k8s ["Kubernetes Cluster"]
        direction TB

        subgraph proxy ["Go Proxy · Port 8080"]
            direction LR
            VAL[Request\nValidation] --> POOL[Connection\nPooling]
            POOL --> LOG[Structured\nLogging]
        end

        subgraph api ["FastAPI · Port 8000"]
            direction TB
            PYDANTIC[Pydantic\nValidation] --> ROUTER[A/B Router]
            ROUTER --> MODEL_A[Model A\ncontrol]
            ROUTER --> MODEL_B[Model B\nchallenger]
            PYDANTIC --> DRIFT[Drift\nDetector]
            PYDANTIC --> METRICS[Feature\nCollector]
        end
    end

    subgraph train ["Training Pipeline"]
        direction LR
        DATA[California\nHousing] --> PIPELINE[sklearn\nPipeline]
        PIPELINE --> ARTIFACTS[Artifacts]
        PIPELINE --> MLFLOW[MLflow\nTracking]
    end

    subgraph cicd ["CI / CD"]
        direction LR
        LINT[Lint] --> TEST[Tests]
        TEST --> BUILD[Docker\nBuild]
    end

    C --> proxy
    proxy --> api
    ARTIFACTS --> api
    BUILD -.->|deploy| k8s

    style ingress fill:#e8f4f8,stroke:#b8d4e3,color:#2c3e50
    style k8s fill:#f0f4f8,stroke:#a3b8cc,color:#2c3e50
    style proxy fill:#3498db,stroke:#2980b9,color:#ffffff
    style api fill:#2ecc71,stroke:#27ae60,color:#ffffff
    style train fill:#9b59b6,stroke:#8e44ad,color:#ffffff
    style cicd fill:#e67e22,stroke:#d35400,color:#ffffff
    style C fill:#34495e,stroke:#2c3e50,color:#ffffff
    style VAL fill:#2980b9,stroke:#2471a3,color:#ffffff
    style POOL fill:#2980b9,stroke:#2471a3,color:#ffffff
    style LOG fill:#2980b9,stroke:#2471a3,color:#ffffff
    style PYDANTIC fill:#27ae60,stroke:#1e8449,color:#ffffff
    style ROUTER fill:#27ae60,stroke:#1e8449,color:#ffffff
    style MODEL_A fill:#1e8449,stroke:#196f3d,color:#ffffff
    style MODEL_B fill:#1e8449,stroke:#196f3d,color:#ffffff
    style DRIFT fill:#27ae60,stroke:#1e8449,color:#ffffff
    style METRICS fill:#27ae60,stroke:#1e8449,color:#ffffff
    style DATA fill:#8e44ad,stroke:#7d3c98,color:#ffffff
    style PIPELINE fill:#8e44ad,stroke:#7d3c98,color:#ffffff
    style ARTIFACTS fill:#8e44ad,stroke:#7d3c98,color:#ffffff
    style MLFLOW fill:#8e44ad,stroke:#7d3c98,color:#ffffff
    style LINT fill:#d35400,stroke:#ba4a00,color:#ffffff
    style TEST fill:#d35400,stroke:#ba4a00,color:#ffffff
    style BUILD fill:#d35400,stroke:#ba4a00,color:#ffffff
```

---

## The Model

Predicts **median house value** for California census block groups (in $100k units) using the 1990 U.S. Census dataset from scikit-learn.

**8 input features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

**Gradient Boosting Regressor** (200 trees, depth 5, learning rate 0.1) wrapped in an sklearn Pipeline with StandardScaler preprocessing. Trained on 16,512 samples.

| | RMSE | MAE | R² |
|---|---|---|---|
| **Train** | 0.382 | 0.265 | 0.891 |
| **Test** | 0.466 | 0.311 | 0.835 |

---

## Training Pipeline

Fully reproducible. A `TrainingConfig` dataclass pins every parameter. Every run is logged to MLflow.

1. **Load & split** — California Housing dataset, 80/20 split, computes per-feature statistics for drift monitoring.
2. **Train & track** — sklearn Pipeline (StandardScaler + GBR), logs parameters and metrics to MLflow.
3. **Save & log** — Serializes to `model.joblib`, writes `metadata.json` with feature stats, uploads both as MLflow artifacts.

---

## Serving Layer

### FastAPI

The model API loads artifacts at startup. Supports single-model and multi-model (A/B) modes.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check with model load status |
| `/model/info` | GET | Model type, feature names, metrics |
| `/predict` | POST | Single prediction (returns variant name) |
| `/predict/batch` | POST | Batch prediction |
| `/ab/experiment` | GET | A/B experiment status and per-variant metrics |
| `/ab/weights` | PUT | Update traffic split live |
| `/monitoring/drift` | GET | Feature drift report against training baselines |

### Go Proxy

Python handles ML. Go handles traffic. The proxy sits in front of FastAPI:

- Validates requests before they reach Python (JSON structure, feature count)
- Limits request sizes (1MB single, 10MB batch)
- Pools TCP connections (100 max idle, 90s timeout)
- Logs every request as structured JSON via `slog`
- Tracks total requests, errors, and average latency at `/metrics`

---

## A/B Testing

Route traffic between model variants with weighted random selection. Drop an `experiment.json` in `artifacts/`:

```json
{
  "name": "gbr-200-vs-500-trees",
  "variants": [
    {"name": "control", "weight": 0.8},
    {"name": "challenger", "weight": 0.2}
  ]
}
```

Each variant has its own `model.joblib` and `metadata.json` under `artifacts/<variant_name>/`. Predictions include `"variant": "control"` so you know which model served them. Force a variant with the `X-Variant` header. Update weights live via `PUT /ab/weights` — no restart needed.

Without `experiment.json`, the service runs in single-model mode. Fully backward compatible.

---

## Monitoring & Drift Detection

Every prediction feeds a ring buffer (last 1,000 requests). The drift detector compares live feature distributions against training baselines using:

- **Normalized mean shift** — how many training std devs the mean has moved
- **Std ratio** — how much the data spread has changed
- **Range breach** — values appearing outside training bounds

Each feature is classified as `none`, `moderate`, or `high` drift. Hit `GET /monitoring/drift` for a full per-feature report.

---

## Kubernetes

Production deployment with autoscaling:

```bash
kubectl apply -k deployments/k8s/
```

| Resource | What it does |
|----------|-------------|
| **model-api** | 2-8 pods, scales at 70% CPU, ClusterIP (internal) |
| **go-proxy** | 2-6 pods, scales at 75% CPU, LoadBalancer (external) |
| **HPA** | Autoscales both services independently |

Readiness and liveness probes on `/health`. Go proxy reaches Python via Kubernetes DNS.

---

## CI/CD

GitHub Actions pipeline on every push to `main` and every PR:

```
Lint (ruff) → Python Tests + Go Tests (parallel) → Docker Build
```

Fails fast — a formatting error surfaces in 5 seconds, not 3 minutes.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **ML** | scikit-learn, pandas, numpy |
| **Serving** | FastAPI, Pydantic v2, uvicorn |
| **Proxy** | Go 1.22, net/http, slog |
| **A/B Testing** | Weighted routing, per-variant metrics |
| **Monitoring** | Feature drift detection, ring buffer collector |
| **Tracking** | MLflow |
| **Containers** | Docker (multi-stage), Docker Compose |
| **Orchestration** | Kubernetes, Kustomize, HPA |
| **CI/CD** | GitHub Actions, ruff |
| **Testing** | pytest (34 tests), go test |

---

## Design Decisions

**Python for ML, Go for infrastructure.** Each language does what it's best at. Python handles data, training, and model serving. Go handles request routing, validation, and connection management.

**Validation at every layer.** The Go proxy validates before forwarding. FastAPI validates again with Pydantic. Two layers mean bad input never reaches the model.

**Backward compatible by default.** A/B testing, drift detection, and monitoring are additive. The service works with a single model and no experiment config. Each feature activates when its configuration is present.

**Observability from day one.** Structured logs, `/metrics`, `/health`, `/monitoring/drift`, `/ab/experiment`, and MLflow tracking. When something goes wrong in production, the data is already there.

---

## License

[MIT](LICENSE)
