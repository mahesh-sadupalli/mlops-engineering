import asyncio
from types import SimpleNamespace as NS

import httpx

from agent.loop import run_agent
from agent.tools import Toolbox


class FakeStream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def events():
            for block in self.response.content:
                if block.type == "text":
                    yield NS(type="text", text=block.text)

        return events()

    async def get_final_message(self):
        return self.response


class FakeClient:
    """Replays scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(stream=self.stream))

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        return FakeStream(self.responses.pop(0))


def response(stop_reason, *content):
    usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50)
    return NS(stop_reason=stop_reason, content=list(content), usage=usage)


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def collect(client, toolbox, question):
    async def go():
        history = [{"role": "user", "content": question}]
        return [e async for e in run_agent(client, toolbox, history)]

    return asyncio.run(go())


def make_toolbox(collection):
    def handler(request):
        return httpx.Response(503, text="down")

    http = httpx.Client(base_url="http://hp", transport=httpx.MockTransport(handler))
    return Toolbox(collection, http)


def test_answers_with_retrieved_sources(collection):
    client = FakeClient(
        [
            response("tool_use", tool_use("t1", "search_docs", {"query": "HPA"})),
            response("end_turn", text("Scales at 70% CPU [k8s/hpa.md › Autoscaling]")),
        ]
    )
    events = collect(client, make_toolbox(collection), "How does autoscaling work?")
    types = [e["type"] for e in events]

    assert types == ["tool", "sources", "text", "done"]
    assert events[1]["sources"][0]["source"] == "k8s/hpa.md"
    assert events[-1]["turns"] == 2
    assert events[-1]["usage"]["input_tokens"] == 200

    # Tool result is sent back on the second request, paired by id
    second = client.requests[1]["messages"]
    assert second[-1]["content"][0]["tool_use_id"] == "t1"


def test_tool_failure_is_reported_to_model(collection):
    client = FakeClient(
        [
            response("tool_use", tool_use("t1", "get_drift_report", {})),
            response("end_turn", text("The model API is down.")),
        ]
    )
    collect(client, make_toolbox(collection), "Is there drift?")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "503" in result["content"]


def test_refusal_stops_the_loop(collection):
    client = FakeClient([response("refusal")])
    events = collect(client, make_toolbox(collection), "hi")
    assert [e["type"] for e in events] == ["error", "done"]


def test_requests_use_fallbacks_and_cached_system(collection):
    client = FakeClient([response("end_turn", text("hi"))])
    collect(client, make_toolbox(collection), "hi")
    req = client.requests[0]
    assert req["fallbacks"] == "default"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
