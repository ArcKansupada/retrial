"""Turning recorded usage into dollars.

An unknown model prices as None, never a guess. Prices are USD per million tokens and go stale;
`register_prices` adds or overrides them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .types import Step, TrajectoryEntry

#: Both record kinds carry usage, so `trajectory_cost` takes either uncast.
PricedEntry = Step | TrajectoryEntry

# (input, output) USD per 1M tokens. Cached 2026-06.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache reads are ~0.1x input; writes ~1.25x at the default 5-minute TTL.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: For a model that costs nothing per token, such as one on your own hardware.
FREE: tuple[float, float] = (0.0, 0.0)

#: Routing prefixes (LiteLLM, OpenRouter, Bedrock). The price belongs to the model.
_ROUTING_PREFIXES = (
    "anthropic.",
    "us.anthropic.",
    "eu.anthropic.",
    "anthropic/",
    "openai/",
    "google/",
    "gemini/",
    "models/",
)


def register_prices(prices: dict[str, tuple[float, float]]) -> None:
    """Teach retrial what a model costs, in USD per million tokens.

        register_prices({
            "my-finetune": (0.50, 1.50),
            "llama3.1:70b": FREE,
        })

    Call it before the run. Entries override built-ins of the same name.
    """
    for name, price in prices.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"model name must be a non-empty string, got {name!r}")
        if (
            not isinstance(price, tuple)
            or len(price) != 2
            or not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in price)
        ):
            raise ValueError(
                f"price for {name!r} must be a (input, output) tuple of numbers "
                f"in USD per million tokens, got {price!r}"
            )
        if any(p < 0 for p in price):
            raise ValueError(f"price for {name!r} cannot be negative, got {price!r}")
        PRICES[name.strip()] = (float(price[0]), float(price[1]))


def normalize(model: Any) -> str | None:
    """Strip provider prefixes and dated suffixes down to a table key."""
    if not isinstance(model, str):
        return None
    name = model.strip()
    for prefix in _ROUTING_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name in PRICES:
        return name
    # Dated snapshots price as their base model. Longest match wins.
    matches = [key for key in PRICES if name.startswith(key)]
    return max(matches, key=len) if matches else None


def cost_of(serialized_response: Any) -> float | None:
    """Cost in USD for one model call, or None if it cannot be known exactly.

    None means unpriced, not free.
    """
    if not isinstance(serialized_response, dict):
        return None

    model = normalize(serialized_response.get("model"))
    usage = serialized_response.get("usage")
    if model is None or not isinstance(usage, dict):
        return None

    input_price, output_price = PRICES[model]

    def tokens(field: str) -> int:
        value = usage.get(field)
        return value if isinstance(value, int) else 0

    billable = (
        tokens("input_tokens")
        + tokens("cache_read_input_tokens") * CACHE_READ_MULTIPLIER
        + tokens("cache_creation_input_tokens") * CACHE_WRITE_MULTIPLIER
    )
    return (billable * input_price + tokens("output_tokens") * output_price) / 1_000_000


def cost_of_step(entry: PricedEntry) -> float | None:
    """A recorded step's cost, preferring the `cost_usd` stored at record time.

    Falls back to computing it from the response for older traces.
    """
    if entry.get("step_type") != "model_call":
        return None
    stored = entry.get("cost_usd")
    if stored is not None:
        return stored
    return cost_of(entry.get("output"))


def trajectory_cost(entries: Iterable[PricedEntry]) -> tuple[float, int]:
    """(cost, unpriced_calls) over a trajectory's model calls."""
    total = 0.0
    unpriced = 0
    for entry in entries:
        if entry.get("step_type") != "model_call":
            continue
        cost = cost_of_step(entry)
        if cost is None:
            unpriced += 1
        else:
            total += cost
    return total, unpriced


def fmt(cost: float | None) -> str:
    if cost is None:
        return "unpriced"
    if cost < 0.01:
        return f"${cost:.5f}"
    return f"${cost:.4f}"
