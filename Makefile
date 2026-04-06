.PHONY: help build test lint clean docker-build docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Python
install: ## Install Python dependencies
	pip install -r requirements.txt

test: ## Run Python tests
	pytest tests/ -v --cov

lint: ## Lint Python code
	ruff check .
	ruff format --check .

format: ## Format Python code
	ruff format .

# Go
build-go: ## Build Go services
	go build -o bin/ ./cmd/...

test-go: ## Run Go tests
	go test ./... -v -race

# Docker
docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

clean: ## Clean build artifacts
	rm -rf bin/ dist/ build/ *.egg-info __pycache__ .pytest_cache htmlcov
