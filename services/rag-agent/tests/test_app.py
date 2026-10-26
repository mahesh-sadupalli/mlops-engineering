import json

import pytest
from fastapi.testclient import TestClient

from tests.test_loop import FakeClient, make_toolbox, response, text, tool_use


@pytest.fixture
def api(collection, monkeypatch):
    fake = FakeClient(
        [
            response("tool_use", tool_use("t1", "search_docs", {"query": "drift"})),
            response("end_turn", text("Drift uses mean shift.")),
        ]
    )
    monkeypatch.setattr(
        "agent.app.build_state",
        lambda: {
            "client": fake,
            "toolbox": make_toolbox(collection),
            "collection": collection,
        },
    )
    from agent.app import app

    with TestClient(app) as c:
        yield c


def sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line]


def test_health_reports_index_size(api):
    data = api.get("/health").json()
    assert data == {"status": "healthy", "indexed_chunks": 3}


def test_chat_streams_events(api):
    resp = api.post("/chat", json={"messages": [{"role": "user", "content": "drift?"}]})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp)
    assert [e["type"] for e in events] == ["tool", "sources", "text", "done"]


def test_metrics_aggregate_requests(api):
    api.post("/chat", json={"messages": [{"role": "user", "content": "drift?"}]})
    m = api.get("/metrics").json()
    assert m["total_requests"] == 1
    assert m["avg_turns"] == 2
    assert m["tool_calls"] == {"search_docs": 1}
    assert m["cache_read_input_tokens"] == 100


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "hi"}],
        [{"role": "system", "content": "x"}],
    ],
)
def test_rejects_invalid_history(api, messages):
    assert api.post("/chat", json={"messages": messages}).status_code == 422


def test_unexpected_errors_end_stream_with_error_event(api, monkeypatch):
    def boom(**kwargs):
        raise TypeError("Could not resolve authentication method")

    from agent.app import _state

    monkeypatch.setattr(_state["client"].beta.messages, "stream", boom)
    resp = api.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    events = sse_events(resp)
    assert events[-1]["type"] == "error"
    assert "authentication" in events[-1]["message"]
    assert api.get("/metrics").json()["total_errors"] == 1
