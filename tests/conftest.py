"""Shared fixtures: a scripted fake model and a toy booking agent.

The model is a pure function of the message history and branches on the tool result, so tests
are deterministic and a fork test cannot pass trivially.
"""

import json
from dataclasses import dataclass, field

import pytest

from retrial.storage import Store


@dataclass
class FakeResponse:
    stop_reason: str
    content: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 5})


TOOLS = [{"name": "search_flight"}, {"name": "check_budget"}]


def fake_model(messages, tools=None):
    last = messages[-1]

    if last["role"] == "user" and isinstance(last["content"], str):
        return FakeResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Looking up that flight."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search_flight",
                    "input": {"route": "AUS-SFO"},
                },
            ],
        )

    results = [
        b
        for b in (last["content"] if isinstance(last["content"], list) else [])
        if b.get("type") == "tool_result"
    ]

    if results and results[-1]["tool_use_id"] == "toolu_01":
        price = json.loads(results[-1]["content"])["flight_price"]
        if price <= 500:
            return FakeResponse(
                stop_reason="end_turn",
                content=[{"type": "text", "text": f"Booked for ${price}."}],
            )
        return FakeResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": f"${price} is over budget."},
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "check_budget",
                    "input": {"amount": price},
                },
            ],
        )

    if results and results[-1]["tool_use_id"] == "toolu_02":
        return FakeResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": "Over budget. Need approval first."}],
        )

    raise AssertionError(f"fake_model has no policy for: {last}")


def make_executor(flight_price):
    """The world. `flight_price` is what the tool would really return."""

    def execute_tools(response):
        out = []
        for block in response.content:
            if block.get("type") != "tool_use":
                continue
            if block["name"] == "search_flight":
                payload = {"flight_price": flight_price}
            elif block["name"] == "check_budget":
                payload = {"approved": False}
            else:
                raise AssertionError(f"unknown tool {block['name']}")
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(payload),
                }
            )
        return out

    return execute_tools


def raw_agent(messages, tools, call_model, execute_tools):
    """A raw SDK-shaped loop. Takes `messages` in rather than starting blank."""
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / ".retrial" / "sessions.db"))
    yield s
    s.close()


@pytest.fixture
def opening():
    return [{"role": "user", "content": "Book me AUS to SFO."}]
