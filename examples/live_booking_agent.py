"""The same booking agent, against the real Claude API.

Only `call_model` differs from `booking_agent.py`. It exercises what a scripted model cannot:
real Pydantic responses, real tool_use ids, thinking blocks, and non-determinism.

Needs ANTHROPIC_API_KEY, exported or in a .env file at the repo root.

Run:  python examples/live_booking_agent.py
"""

import json
import os
import pathlib

from retrial import record

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are a flight booking assistant. Use the tools available to you. "
    "Search for the flight first, then book it only if the fare is at or under "
    "the traveller's $600 budget. If it is over budget, do not book — say so "
    "and stop."
)

TOOLS = [
    {
        "name": "search_flight",
        "description": "Look up the current fare for a route.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin airport code"},
                "destination": {"type": "string", "description": "Destination code"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight and return a confirmation code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "fare_usd": {"type": "number"},
            },
            "required": ["origin", "destination", "fare_usd"],
        },
    },
]


def load_env(root=None):
    """Read a .env file into os.environ. Keeps the key out of the transcript."""
    path = pathlib.Path(root or pathlib.Path(__file__).parent.parent) / ".env"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    return True


def make_client():
    load_env()
    import anthropic

    return anthropic.Anthropic()


def make_call_model(client, effort="low"):
    """The one line that differs from the scripted example.

    `effort` is low by default: this checks plumbing, not reasoning.
    """

    def call_model(messages, tools):
        return client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            tools=tools,
            messages=messages,
        )

    return call_model


_lazy_client = None


def call_model(messages, tools):
    """The default model call, used when the CLI invokes this agent.

    A real function that builds its client on first call, so importing needs no credentials.
    """
    global _lazy_client
    if _lazy_client is None:
        _lazy_client = make_client()
    return make_call_model(_lazy_client)(messages, tools)


def execute_tools(response):
    """Real tool blocks: objects on a live call, dicts when replayed."""
    results = []
    for block in response.content:
        if _get(block, "type") != "tool_use":
            continue
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": _get(block, "id"),
                "content": json.dumps(_run(_get(block, "name"), _get(block, "input"))),
            }
        )
    return results


def _get(block, field):
    return block.get(field) if isinstance(block, dict) else getattr(block, field, None)


def _run(name, args):
    if name == "search_flight":
        return {"fare_usd": 450, "carrier": "Southwest", "stops": 0}
    if name == "book_flight":
        if os.environ.get("RETRIAL_DEMO_OUTAGE"):
            return {"error": "airline API timed out"}
        return {"confirmation": "QX7R2M", "fare_usd": args.get("fare_usd")}
    return {"error": f"unknown tool {name}"}


@record(session_name="live-booking")
def run_agent(messages, tools=TOOLS, call_model=call_model, execute_tools=execute_tools):
    """Identical in shape to the scripted example's loop."""
    while True:
        response = call_model(messages, tools)
        # Append the full content list: thinking blocks must be echoed back unchanged.
        messages.append({"role": "assistant", "content": _content(response)})
        if _get(response, "stop_reason") != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


def _content(response):
    return response.content if not isinstance(response, dict) else response["content"]


def final_text(response):
    blocks = _content(response)
    texts = [_get(b, "text") for b in blocks if _get(b, "type") == "text"]
    return texts[-1] if texts else None


if __name__ == "__main__":
    from retrial.cli import echo

    client = make_client()
    result = run_agent(
        [{"role": "user", "content": "Book me a flight from AUS to SFO."}],
        call_model=make_call_model(client),
    )
    # Not `print`: echo() degrades a character a cp1252 console cannot encode.
    echo(final_text(result))
    echo(f"\nRecorded session {run_agent.last_session_id}")
    echo(f"  retrial log {run_agent.last_session_id}")
