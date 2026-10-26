"""Tools the agent can call: doc search plus live calls to the house-prices API."""

import json

import httpx
from chromadb.api.models.Collection import Collection

from .store import search


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": list(properties or {}),
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool(
        "search_docs",
        "Semantic search over this project's engineering docs (architecture, "
        "serving, A/B testing, drift detection, Kubernetes, CI/CD). Returns chunks "
        "with source file and section heading. Use for any 'how/why does it work' "
        "question, and cite the sources you use.",
        {"query": {"type": "string", "description": "Natural-language query"}},
    ),
    _tool(
        "get_model_info",
        "Live metadata of the deployed house-price model: type, feature names, "
        "test metrics (RMSE, MAE, R2), training sample count.",
    ),
    _tool(
        "get_drift_report",
        "Live drift report comparing recent prediction inputs against training "
        "baselines: per-feature mean shift, std ratio, range breaches, status.",
    ),
    _tool(
        "get_ab_experiment",
        "Live A/B experiment status: variants, traffic weights, request counts, "
        "average latency and average prediction per variant.",
    ),
    _tool(
        "predict_price",
        "Get a live prediction (median house value in $100k units) from the "
        "deployed model.",
        {
            "features": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Exactly 8 values in order: MedInc, HouseAge, "
                "AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude",
            }
        },
    ),
]


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, collection: Collection, http: httpx.Client):
        self.collection = collection
        self.http = http

    def run(self, name: str, args: dict) -> str:
        """Execute a tool and return its JSON-encoded result."""
        if name == "search_docs":
            return json.dumps(search(self.collection, args["query"]))
        routes = {
            "get_model_info": ("GET", "/model/info", None),
            "get_drift_report": ("GET", "/monitoring/drift", None),
            "get_ab_experiment": ("GET", "/ab/experiment", None),
            "predict_price": ("POST", "/predict", args),
        }
        if name not in routes:
            raise ToolError(f"Unknown tool: {name}")
        method, path, body = routes[name]
        try:
            resp = self.http.request(method, path, json=body)
        except httpx.HTTPError as e:
            raise ToolError(f"house-prices API unreachable: {e}") from e
        if resp.status_code >= 400:
            raise ToolError(f"house-prices API {resp.status_code}: {resp.text}")
        return resp.text
