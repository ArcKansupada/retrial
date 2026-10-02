"""Adapters: use retrial with any model, from any provider.

Each adapter translates one provider's wire format to and from the canonical shape below, so
forking a `tool_call` step works whoever ran the call.

    messages       {"role": "user"|"assistant", "content": str | [block, ...]}
    tool call      {"type": "tool_use", "id":..., "name":..., "input": {...}}
    tool result    {"type": "tool_result", "tool_use_id":..., "content": str}
    response       .content (blocks), .stop_reason, .usage, .model

Tools are declared once in canonical form (`name`, `description`, `input_schema`).

    from retrial.providers import openai_adapter

    call_model = openai_adapter(model="gpt-5")
    call_model = openai_adapter(model="llama3.1", base_url="http://localhost:11434/v1")

SDKs are optional extras (`retrial[openai]`, `retrial[gemini]`), imported lazily.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from ..types import JSON

__all__ = [
    "Adapter",
    "ModelResponse",
    "gemini_adapter",
    "openai_adapter",
    "tool_result",
    "tool_uses",
]


class ModelResponse:
    """One model call's result, in retrial's shape rather than a provider's."""

    __slots__ = ("content", "stop_reason", "usage", "model", "raw")

    def __init__(
        self,
        content: list[JSON],
        stop_reason: str | None = None,
        usage: dict[str, Any] | None = None,
        model: str | None = None,
        raw: Any = None,
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or {}
        self.model = model
        self.raw = raw

    def to_dict(self) -> dict[str, Any]:
        """What lands in the trace. Leaves `raw` out."""
        return {
            "content": self.content,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "model": self.model,
        }

    @property
    def text(self) -> str | None:
        """The response's text, joined across blocks. None if it said nothing."""
        parts = [
            block.get("text")
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        found = [p for p in parts if isinstance(p, str) and p]
        return "\n".join(found) if found else None

    def __repr__(self) -> str:
        kinds = [b.get("type", "?") for b in self.content if isinstance(b, dict)]
        return (
            f"ModelResponse(model={self.model!r}, stop_reason="
            f"{self.stop_reason!r}, content=[{', '.join(map(str, kinds))}])"
        )


@runtime_checkable
class Adapter(Protocol):
    """What retrial needs from a provider.

    Callable, so it can be passed as `call_model`. The translators are pure.
    """

    def __call__(self, messages: list[JSON], tools: list[JSON]) -> ModelResponse:
        """Call the model. Canonical messages in, canonical response out."""
        ...

    def to_request(self, messages: list[JSON], tools: list[JSON]) -> dict[str, Any]:
        """Canonical messages and tools -> this provider's request payload."""
        ...

    def from_response(self, response: Any) -> ModelResponse:
        """This provider's response object -> a canonical `ModelResponse`."""
        ...


def tool_result(tool_use_id: str, content: Any) -> dict[str, Any]:
    """Build the canonical tool-result block a fork splices on."""
    if not isinstance(tool_use_id, str) or not tool_use_id:
        raise ValueError(
            f"tool_use_id must be a non-empty string, got {tool_use_id!r}; "
            "a fork matches recorded results by this id and cannot splice "
            "a result it cannot address"
        )
    if not isinstance(content, str):
        # Tool results have to survive the round-trip through storage as text.
        content = json.dumps(content)
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def tool_uses(response: Any) -> list[dict[str, Any]]:
    """The tool calls in a response, whether a live object or a replayed dict."""
    content = (
        response.get("content")
        if isinstance(response, dict)
        else getattr(response, "content", None)
    )
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def openai_adapter(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export, so `retrial.providers` imports without the OpenAI SDK."""
    from .openai import openai_adapter as factory

    return factory(*args, **kwargs)


def gemini_adapter(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export, so `retrial.providers` imports without google-genai."""
    from .gemini import gemini_adapter as factory

    return factory(*args, **kwargs)
