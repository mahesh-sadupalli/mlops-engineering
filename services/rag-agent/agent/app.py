"""FastAPI service exposing the MLOps copilot agent over Server-Sent Events."""

import json
import logging
from collections import Counter
from contextlib import asynccontextmanager

import anthropic
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .loop import run_agent
from .schemas import ChatRequest, HealthResponse, MetricsResponse
from .store import get_collection
from .tools import Toolbox

logger = logging.getLogger(__name__)

_state: dict = {}


class Metrics:
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.latency_ms = 0
        self.turns = 0
        self.tokens = Counter()
        self.tool_calls = Counter()

    def record(self, event: dict) -> None:
        if event["type"] == "tool":
            self.tool_calls[event["name"]] += 1
        elif event["type"] == "error":
            self.errors += 1
        elif event["type"] == "done":
            self.requests += 1
            self.latency_ms += event["latency_ms"]
            self.turns += event["turns"]
            self.tokens.update(event["usage"])

    def summary(self) -> MetricsResponse:
        n = max(self.requests, 1)
        return MetricsResponse(
            total_requests=self.requests,
            total_errors=self.errors,
            avg_latency_ms=round(self.latency_ms / n, 1),
            avg_turns=round(self.turns / n, 2),
            input_tokens=self.tokens["input_tokens"],
            output_tokens=self.tokens["output_tokens"],
            cache_read_input_tokens=self.tokens["cache_read_input_tokens"],
            tool_calls=dict(self.tool_calls),
        )


def build_state() -> dict:
    collection = get_collection()
    http = httpx.Client(base_url=settings.house_prices_url, timeout=10)
    return {
        "client": anthropic.AsyncAnthropic(),
        "toolbox": Toolbox(collection, http),
        "collection": collection,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.update(build_state(), metrics=Metrics())
    print(f"Agent ready: {_state['collection'].count()} chunks, model {settings.model}")
    yield
    _state["toolbox"].http.close()
    _state.clear()


app = FastAPI(title="MLOps Copilot Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", indexed_chunks=_state["collection"].count())


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return _state["metrics"].summary()


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream agent events as SSE: text, tool, sources, error, done."""
    history = [m.model_dump() for m in req.messages]

    async def events():
        try:
            async for event in run_agent(_state["client"], _state["toolbox"], history):
                _state["metrics"].record(event)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            # Stream boundary: always end with an event the UI can render
            logger.exception("chat failed")
            message = e.message if isinstance(e, anthropic.APIError) else str(e)
            event = {"type": "error", "message": message}
            _state["metrics"].record(event)
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
