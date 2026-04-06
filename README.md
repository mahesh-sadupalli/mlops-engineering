<div align="center">

# MLOps Engineering

**Production-grade ML systems built with Python and Go, deployed with containers and Kubernetes.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

*Bridging the gap between ML notebooks and production systems — from experiment tracking to scaled deployment.*

</div>

## What This Repo Is About

This repository implements end-to-end MLOps workflows as production-ready services. Each project demonstrates a different slice of the ML lifecycle:

| Area | What's Covered |
|------|---------------|
| **Model Serving** | FastAPI + Go high-throughput inference endpoints, A/B testing, canary deploys |
| **Training Pipelines** | Reproducible training with MLflow tracking, hyperparameter optimization |
| **Feature Engineering** | Feature stores, transformation pipelines, online/offline serving |
| **Data Ingestion** | Go-based ingestion services for throughput-critical data paths |
| **Monitoring & Drift** | Model performance tracking, data drift detection, automated alerts |
| **Infrastructure** | Dockerized services, Kubernetes manifests, CI/CD with GitHub Actions |

## Tech Stack

<table>
<tr><td><b>ML & Serving</b></td><td>PyTorch, scikit-learn, FastAPI, MLflow</td></tr>
<tr><td><b>Infrastructure</b></td><td>Go, Docker, Kubernetes, Helm</td></tr>
<tr><td><b>Observability</b></td><td>Prometheus, Grafana, structured logging</td></tr>
<tr><td><b>Data</b></td><td>Redis, PostgreSQL, Apache Kafka</td></tr>
<tr><td><b>CI/CD</b></td><td>GitHub Actions, Makefile-driven builds</td></tr>
</table>

## Design Philosophy

> **Python for ML logic. Go for infrastructure. Containers for everything.**

- Every service is containerized and Kubernetes-ready
- Experiments are tracked and reproducible
- Observability is built in from day one, not bolted on later
- Code is tested, linted, and deployable through CI/CD

## License

[MIT](LICENSE)
