"""Agentic loop: stream Claude's answer, run tools it asks for, repeat."""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import anthropic

from .config import settings
from .tools import TOOLS, Toolbox, ToolError

SYSTEM = """You are the MLOps copilot for a production house-price prediction \
platform (FastAPI model API, Go proxy, A/B testing, drift detection, Kubernetes, \
GitHub Actions CI).

Answer engineering questions about how the platform is built and how it is \
behaving right now:
- For design or "how does X work" questions, call search_docs and ground your \
answer in what it returns. Cite sources inline as [file › section].
- For questions about the live system (model metrics, drift, A/B traffic, a \
prediction), call the matching live tool rather than guessing.
- If the docs and tools don't contain the answer, say so plainly.

Keep answers concise and technical. Use markdown; code blocks for commands."""


async def run_agent(
    client: anthropic.AsyncAnthropic,
    toolbox: Toolbox,
    history: list[dict],
) -> AsyncIterator[dict]:
    """Yield UI events: text deltas, tool calls, retrieved sources, and a final
    'done' event with token usage. `history` is plain {role, content} turns."""
    messages = list(history)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    start = time.perf_counter()

    for turn in range(1, settings.max_turns + 1):
        async with client.beta.messages.stream(
            model=settings.model,
            max_tokens=16000,
            system=[
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            tools=TOOLS,
            messages=messages,
            output_config={"effort": settings.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield {"type": "text", "text": event.text}
            response = await stream.get_final_message()

        for key in usage:
            usage[key] += getattr(response.usage, key, 0) or 0

        if response.stop_reason == "refusal":
            yield {"type": "error", "message": "The model declined this request."}
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason == "max_tokens":
            break

        for block in tool_uses:
            yield {"type": "tool", "name": block.name, "input": block.input}
        results = await asyncio.gather(*(_run_tool(toolbox, b) for b in tool_uses))
        for block, result in zip(tool_uses, results):
            if block.name == "search_docs" and not result.get("is_error"):
                yield {"type": "sources", "sources": json.loads(result["content"])}

        # All results go back in one user turn so Claude keeps calling in parallel
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    yield {
        "type": "done",
        "turns": turn,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "usage": usage,
    }


async def _run_tool(toolbox: Toolbox, block) -> dict:
    result = {"type": "tool_result", "tool_use_id": block.id}
    try:
        result["content"] = await asyncio.to_thread(
            toolbox.run, block.name, block.input
        )
    except ToolError as e:
        result.update(content=str(e), is_error=True)
    return result
