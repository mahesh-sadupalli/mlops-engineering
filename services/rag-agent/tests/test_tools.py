import json

import httpx
import pytest

from agent.tools import TOOLS, Toolbox, ToolError


def fake_house_prices(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/model/info":
        return httpx.Response(200, json={"model_type": "gradient_boosting"})
    if request.url.path == "/predict":
        features = json.loads(request.content)["features"]
        if len(features) != 8:
            return httpx.Response(422, json={"detail": "Expected 8 features"})
        return httpx.Response(200, json={"prediction": 4.2, "variant": "control"})
    return httpx.Response(404)


@pytest.fixture
def toolbox(collection):
    http = httpx.Client(
        base_url="http://house-prices", transport=httpx.MockTransport(fake_house_prices)
    )
    return Toolbox(collection, http)


def test_tool_schemas_are_strict():
    for tool in TOOLS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False


def test_search_docs(toolbox):
    hits = json.loads(toolbox.run("search_docs", {"query": "drift mean shift"}))
    assert hits[0]["source"] == "README.md"


def test_model_info_calls_api(toolbox):
    info = json.loads(toolbox.run("get_model_info", {}))
    assert info["model_type"] == "gradient_boosting"


def test_predict_forwards_features(toolbox):
    result = json.loads(toolbox.run("predict_price", {"features": [1.0] * 8}))
    assert result["prediction"] == 4.2


def test_api_errors_become_tool_errors(toolbox):
    with pytest.raises(ToolError, match="422"):
        toolbox.run("predict_price", {"features": [1.0, 2.0]})


def test_unknown_tool(toolbox):
    with pytest.raises(ToolError, match="Unknown tool"):
        toolbox.run("rm_rf", {})
