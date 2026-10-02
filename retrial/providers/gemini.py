"""Gemini, through the google-genai SDK.

Turns are `contents` with `parts`, the assistant is `model`, and tool calls carry no id (see
`_call_id`). Needs `retrial[gemini]` and GOOGLE_API_KEY or GEMINI_API_KEY, or a `client=`.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import JSON
from . import ModelResponse

__all__ = ["GeminiAdapter", "gemini_adapter"]

#: Unrecognized reasons pass through untranslated, as in the OpenAI adapter.
_STOP_REASONS = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "safety",
    "RECITATION": "recitation",
}


class GeminiAdapter:
    """Canonical messages in, canonical `ModelResponse` out."""

    def __init__(
        self,
        model: str,
        system: str | None = None,
        client: Any = None,
        api_key: str | None = None,
        max_tokens: int | None = 4096,
        **params: Any,
    ) -> None:
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.params = params
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        """Built on first use, never at import - see `OpenAIAdapter.client`."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "the Gemini adapter needs the google-genai SDK: "
                    "pip install 'retrial[gemini]'. retrial's core never "
                    "imports an SDK; only this adapter does."
                ) from exc
            kwargs = {"api_key": self._api_key} if self._api_key else {}
            self._client = genai.Client(**kwargs)
        return self._client

    def __call__(self, messages: list[JSON], tools: list[JSON] | None = None) -> ModelResponse:
        request = self.to_request(messages, tools or [])
        return self.from_response(self.client.models.generate_content(**request))

    # -- translation, pure and offline-testable -----------------------------

    def to_request(self, messages: list[JSON], tools: list[JSON]) -> dict[str, Any]:
        config: dict[str, Any] = dict(self.params)
        if self.system:
            config["system_instruction"] = self.system
        if self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
        if tools:
            # One tool object holding every declaration, not one per tool.
            config["tools"] = [
                {"function_declarations": [to_gemini_tool(t) for t in tools]}
            ]

        request: dict[str, Any] = {
            "model": self.model,
            "contents": to_gemini_contents(messages),
        }
        if config:
            request["config"] = config
        return request

    def from_response(self, response: Any) -> ModelResponse:
        candidate = _first(_get(response, "candidates")) or {}
        parts = _get(_get(candidate, "content"), "parts") or []

        content: list[JSON] = []
        for index, part in enumerate(parts):
            text = _get(part, "text")
            if isinstance(text, str) and text:
                content.append({"type": "text", "text": text})
                continue
            call = _get(part, "function_call")
            if call is not None:
                name = _get(call, "name")
                content.append(
                    {
                        "type": "tool_use",
                        "id": _call_id(name, index),
                        "name": name,
                        "input": _get(call, "args") or {},
                    }
                )

        finish = _get(candidate, "finish_reason")
        finish = getattr(finish, "name", finish)  # the SDK returns an enum
        stop_reason: str | None
        if any(b.get("type") == "tool_use" for b in content):
            stop_reason = "tool_use"
        else:
            stop_reason = _STOP_REASONS.get(finish, finish) if finish else None

        return ModelResponse(
            content=content,
            stop_reason=stop_reason,
            usage=to_usage(_get(response, "usage_metadata")),
            model=_get(response, "model_version") or self.model,
            raw=response,
        )


def gemini_adapter(model: str, **kwargs: Any) -> GeminiAdapter:
    """Build an adapter. See `GeminiAdapter` for the arguments."""
    return GeminiAdapter(model, **kwargs)


def _call_id(name: Any, index: int) -> str:
    """Give a Gemini tool call the id the canonical shape requires.

    Gemini issues none, so name plus position is used. Two calls to the same tool in one turn
    are told apart by position only.
    """
    return f"{name or 'call'}_{index}"


def to_gemini_contents(messages: list[JSON]) -> list[JSON]:
    """Canonical history -> Gemini `contents`.

    Tool results become `functionResponse` parts, matched to their call by name.
    """
    out: list[JSON] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = "model" if message.get("role") == "assistant" else "user"
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "parts": [{"text": content}]})
            continue
        if not isinstance(content, list):
            continue

        parts: list[JSON] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append({"text": str(block.get("text", ""))})
            elif kind == "tool_use":
                parts.append(
                    {
                        "function_call": {
                            "name": block.get("name"),
                            "args": block.get("input") or {},
                        }
                    }
                )
            elif kind == "tool_result":
                parts.append(
                    {
                        "function_response": {
                            "name": _name_from_id(block.get("tool_use_id")),
                            "response": _as_response(block.get("content")),
                        }
                    }
                )
        if parts:
            out.append({"role": role, "parts": parts})
    return out


def to_gemini_tool(tool: JSON) -> JSON:
    """Canonical tool declaration -> a Gemini function declaration."""
    if not isinstance(tool, dict):
        return tool
    if "function_declarations" in tool:
        return tool
    return {
        "name": tool.get("name"),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema")
        or tool.get("parameters")
        or {"type": "object", "properties": {}},
    }


def to_usage(usage: Any) -> dict[str, Any]:
    """Gemini token counts -> retrial's usage keys.

    Cached tokens are subtracted from the prompt count, as retrial prices them apart.
    """
    if usage is None:
        return {}
    prompt = _int(_get(usage, "prompt_token_count"))
    candidates = _int(_get(usage, "candidates_token_count"))
    cached = _int(_get(usage, "cached_content_token_count"))

    out: dict[str, Any] = {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": candidates,
    }
    if cached:
        out["cache_read_input_tokens"] = cached
    return out


def _name_from_id(tool_use_id: Any) -> str:
    """Recover the tool name from an id `_call_id` built."""
    if not isinstance(tool_use_id, str):
        return ""
    head, _, tail = tool_use_id.rpartition("_")
    return head if head and tail.isdigit() else tool_use_id


def _as_response(content: Any) -> dict[str, Any]:
    """Gemini wants an object back from a tool, not a bare string."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"result": content}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    return {"result": content}


def _get(obj: Any, field: str) -> Any:
    """Read a field off an SDK object (snake_case) or its dict form (camelCase)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if field in obj:
            return obj[field]
        return obj.get(_camel(field))
    found = getattr(obj, field, None)
    return found if found is not None else getattr(obj, _camel(field), None)


def _camel(field: str) -> str:
    head, *rest = field.split("_")
    return head + "".join(word.title() for word in rest)


def _first(seq: Any) -> Any:
    return seq[0] if isinstance(seq, (list, tuple)) and seq else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
