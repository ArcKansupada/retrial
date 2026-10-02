"""OpenAI, and every server that speaks its wire format.

    ollama      http://localhost:11434/v1
    vLLM        http://localhost:8000/v1
    llama.cpp   http://localhost:8080/v1
    LM Studio   http://localhost:1234/v1
    OpenRouter  https://openrouter.ai/api/v1

Local servers ignore the key but the SDK requires one, hence `api_key="unused"`.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import JSON
from . import ModelResponse

__all__ = ["OpenAIAdapter", "openai_adapter"]

#: Unrecognized reasons pass through untranslated.
_STOP_REASONS = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
}


class OpenAIAdapter:
    """Canonical messages in, canonical `ModelResponse` out.

    Extra keyword arguments (temperature, top_p, seed, ...) go on every request.
    """

    def __init__(
        self,
        model: str,
        system: str | None = None,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = 4096,
        **params: Any,
    ) -> None:
        self.model = model
        self.system = system
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.params = params
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        """Built on first use, so importing needs no credentials."""
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "the OpenAI adapter needs the openai SDK: "
                    "pip install 'retrial[openai]'. retrial's core never "
                    "imports an SDK; only this adapter does."
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def __call__(self, messages: list[JSON], tools: list[JSON] | None = None) -> ModelResponse:
        request = self.to_request(messages, tools or [])
        return self.from_response(self.client.chat.completions.create(**request))

    # -- translation, pure and offline-testable -----------------------------

    def to_request(self, messages: list[JSON], tools: list[JSON]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, self.system),
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if tools:
            # Omitted when empty: some compatible servers reject `tools: []`.
            request["tools"] = [to_openai_tool(t) for t in tools]
        request.update(self.params)
        return request

    def from_response(self, response: Any) -> ModelResponse:
        choice = _first(_get(response, "choices")) or {}
        message = _get(choice, "message") or {}

        content: list[JSON] = []
        text = _get(message, "content")
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})
        for call in _get(message, "tool_calls") or []:
            content.append(_to_tool_use(call))

        finish = _get(choice, "finish_reason")
        stop_reason: str | None
        if any(b.get("type") == "tool_use" for b in content):
            # Some compatible servers label a tool turn "stop"; trust the content.
            stop_reason = "tool_use"
        else:
            stop_reason = _STOP_REASONS.get(finish, finish) if finish else None

        return ModelResponse(
            content=content,
            stop_reason=stop_reason,
            usage=to_usage(_get(response, "usage")),
            model=_get(response, "model") or self.model,
            raw=response,
        )


def openai_adapter(model: str, **kwargs: Any) -> OpenAIAdapter:
    """Build an adapter. See `OpenAIAdapter` for the arguments."""
    return OpenAIAdapter(model, **kwargs)


def to_openai_messages(messages: list[JSON], system: str | None = None) -> list[JSON]:
    """Canonical history -> OpenAI's message list.

    Lossy: tool results become separate `role: "tool"` messages.
    """
    out: list[JSON] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "assistant":
            out.append(_assistant_message(content))
            continue

        # A user turn: tool results become their own messages, in order.
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _as_text(block.get("content")),
                    }
                )
            elif kind == "text":
                texts.append(str(block.get("text", "")))
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})

    return out


def _assistant_message(content: list[JSON]) -> JSON:
    texts: list[str] = []
    calls: list[JSON] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        # Reasoning blocks are dropped from the request; the trace still records them.

    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = "\n".join(texts) if texts else None
    if calls:
        message["tool_calls"] = calls
    return message


def to_openai_tool(tool: JSON) -> JSON:
    """Canonical tool declaration -> OpenAI's function schema.

    A tool already in OpenAI's shape passes through.
    """
    if not isinstance(tool, dict):
        return tool
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema")
            or tool.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def to_usage(usage: Any) -> dict[str, Any]:
    """OpenAI token counts -> retrial's usage keys.

    `prompt_tokens` includes cached tokens, which retrial prices apart, so they are subtracted.
    """
    if usage is None:
        return {}
    prompt = _int(_get(usage, "prompt_tokens"))
    completion = _int(_get(usage, "completion_tokens"))
    cached = _int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens"))

    out: dict[str, Any] = {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": completion,
    }
    if cached:
        out["cache_read_input_tokens"] = cached
    return out


def _to_tool_use(call: Any) -> dict[str, Any]:
    function = _get(call, "function") or {}
    return {
        "type": "tool_use",
        "id": _get(call, "id"),
        "name": _get(function, "name"),
        "input": _parse_arguments(_get(function, "arguments")),
    }


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Arguments as a dict.

    Malformed JSON is kept verbatim under `_unparsed_arguments`, not replaced by `{}`.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"_unparsed_arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"_unparsed_arguments": arguments}


def _get(obj: Any, field: str) -> Any:
    """Read a field off an SDK object or the plain dict a replay hands back."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _first(seq: Any) -> Any:
    return seq[0] if isinstance(seq, (list, tuple)) and seq else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content)
