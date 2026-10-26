from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=40)

    @field_validator("messages")
    @classmethod
    def last_is_user(cls, v: list[Message]) -> list[Message]:
        if v[-1].role != "user":
            raise ValueError("last message must be from the user")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"messages": [{"role": "user", "content": "Is the model drifting?"}]}
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    avg_turns: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    tool_calls: dict[str, int]
