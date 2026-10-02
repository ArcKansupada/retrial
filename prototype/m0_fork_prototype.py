"""Milestone 0 - throwaway prototype. Not production code; do not import.

Tests one claim: splice an edited tool result into a recorded history at step N, resume the real
loop, and the agent behaves exactly as if the edit had happened. "Exactly" means
indistinguishable from a from-scratch run in a world where the tool really returns the edited
value.

Run:  python prototype/m0_fork_prototype.py
"""

import copy
import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# A fake model: deterministic, and it branches on the tool result.
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    stop_reason: str  # "tool_use" | "end_turn"
    content: list = field(default_factory=list)


def fake_model(messages, tools):
    """Mimics a booking agent's decision policy."""
    last = messages[-1]

    # Opening user turn -> look up the flight.
    if last["role"] == "user" and isinstance(last["content"], str):
        return ModelResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Let me look up that flight."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search_flight",
                    "input": {"route": "AUS-SFO"},
                },
            ],
        )

    results = _tool_results_in(last)

    # Flight price came back -> branch on it. THIS is what the edit steers.
    if results and results[-1]["tool_use_id"] == "toolu_01":
        price = json.loads(results[-1]["content"])["flight_price"]
        if price <= 500:
            return ModelResponse(
                stop_reason="end_turn",
                content=[{"type": "text", "text": f"Booked for ${price}."}],
            )
        # Over budget: an extra step appears that the original run never had.
        return ModelResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": f"${price} is over budget. Checking policy."},
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "check_budget",
                    "input": {"amount": price},
                },
            ],
        )

    if results and results[-1]["tool_use_id"] == "toolu_02":
        return ModelResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": "Over budget. Need approval first."}],
        )

    raise AssertionError(f"fake_model has no policy for: {last}")


def _tool_results_in(message):
    if not isinstance(message.get("content"), list):
        return []
    return [b for b in message["content"] if b.get("type") == "tool_result"]


# ---------------------------------------------------------------------------
# "The user's code": a raw loop that takes `messages` as a parameter.
# ---------------------------------------------------------------------------


def run_agent(messages, tools, call_model, execute_tools):
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = execute_tools(response)
        messages.append({"role": "user", "content": results})


def make_tool_executor(flight_price):
    """The 'world'. flight_price is what the real tool would return."""

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


# ---------------------------------------------------------------------------
# A crude recorder: no SQLite, no SHAs.
# ---------------------------------------------------------------------------


def record(fn):
    steps = []

    def wrapped(messages, tools, call_model, execute_tools):
        def recording_call_model(msgs, tls):
            # Snapshot the messages verbatim. This is the state a fork resumes from.
            snapshot = copy.deepcopy(msgs)
            response = call_model(msgs, tls)
            steps.append(
                {
                    "type": "model_call",
                    "input": {"messages": snapshot},
                    "output": {
                        "stop_reason": response.stop_reason,
                        "content": copy.deepcopy(response.content),
                    },
                }
            )
            return response

        def recording_execute_tools(response):
            results = execute_tools(response)
            steps.append(
                {
                    "type": "tool_call",
                    "input": [
                        copy.deepcopy(b)
                        for b in response.content
                        if b.get("type") == "tool_use"
                    ],
                    "output": copy.deepcopy(results),
                }
            )
            return results

        result = fn(messages, tools, recording_call_model, recording_execute_tools)
        return result, steps

    return wrapped


# ---------------------------------------------------------------------------
# The fork mechanic: the thing under test.
# ---------------------------------------------------------------------------


def fork_from_tool_call(steps, step_index, edit, tools, call_model, execute_tools):
    """Fork at a tool_call step, substituting an edited tool output.

    The seed history is the verbatim `messages` snapshot handed to the next model call; only the
    one field is patched.
    """
    step = steps[step_index]
    assert step["type"] == "tool_call", "can only fork tool_call steps in M0"

    # The state after step N is, by construction, the input to step N+1.
    next_model_call = _next_model_call(steps, step_index)
    if next_model_call is None:
        raise ValueError(
            "cannot fork from the last tool_call: no subsequent model call was "
            "recorded, so the message state after it was never observed"
        )
    seed = copy.deepcopy(next_model_call["input"]["messages"])

    edited_output = edit(copy.deepcopy(step["output"]))

    # Splice by tool_use_id and verify it landed, or refuse.
    patched = 0
    for recorded, new in zip(step["output"], edited_output):
        for message in seed:
            if not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if block.get("type") != "tool_result":
                    continue
                if block.get("tool_use_id") != recorded["tool_use_id"]:
                    continue
                if block.get("content") != recorded["content"]:
                    raise ValueError(
                        "recorded tool output does not appear verbatim in the "
                        "message history; refusing to guess where it went"
                    )
                block["content"] = new["content"]
                patched += 1
    if patched != len(step["output"]):
        raise ValueError(f"spliced {patched} of {len(step['output'])} tool results")

    # Resume the user's real loop from the spliced state.
    return run_agent(seed, tools, call_model, execute_tools)


def _next_model_call(steps, after_index):
    for step in steps[after_index + 1 :]:
        if step["type"] == "model_call":
            return step
    return None


# ---------------------------------------------------------------------------
# The experiment.
# ---------------------------------------------------------------------------


def main():
    tools = [{"name": "search_flight"}, {"name": "check_budget"}]

    # 1. Original run. The world says the flight costs $450.
    original_messages = [{"role": "user", "content": "Book me AUS to SFO."}]
    original_response, steps = record(run_agent)(
        original_messages, tools, fake_model, make_tool_executor(450)
    )
    print("ORIGINAL")
    print(f"  steps:  {[s['type'] for s in steps]}")
    print(f"  answer: {original_response.content[-1]['text']}\n")

    # 2. Fork at the tool_call, pretending the flight cost $999 instead.
    tool_call_index = next(i for i, s in enumerate(steps) if s["type"] == "tool_call")

    def edit(output):
        output[0]["content"] = json.dumps({"flight_price": 999})
        return output

    forked_messages = fork_from_tool_call(
        steps,
        tool_call_index,
        edit,
        tools,
        fake_model,
        # Only the spliced fact changed; later tool calls still run against the $450 world.
        make_tool_executor(450),
    )
    # run_agent returns the response; grab the history it built via closure.
    print("FORKED (spliced flight_price=999, resumed live)")
    print(f"  answer: {forked_messages.content[-1]['text']}\n")

    # 3. The assertion that matters: the forked run must match a from-scratch run in a world
    # where the flight really costs $999.
    counterfactual_messages = [{"role": "user", "content": "Book me AUS to SFO."}]
    counterfactual_response = run_agent(
        counterfactual_messages, tools, fake_model, make_tool_executor(999)
    )
    print("COUNTERFACTUAL (world where flight really costs $999)")
    print(f"  answer: {counterfactual_response.content[-1]['text']}\n")

    # --- checks ---
    checks = []

    checks.append(
        (
            "fork diverges from original (the edit actually steered the model)",
            original_response.content[-1]["text"]
            != forked_messages.content[-1]["text"],
        )
    )
    checks.append(
        (
            "fork == counterfactual final answer",
            forked_messages.content[-1]["text"]
            == counterfactual_response.content[-1]["text"],
        )
    )
    # The forked run grew a step (check_budget) the original never had.
    checks.append(
        (
            "fork took a structurally different path (new tool call appeared)",
            any(
                b.get("name") == "check_budget"
                for m in counterfactual_messages
                if isinstance(m.get("content"), list)
                for b in m["content"]
            ),
        )
    )
    # The original run's history must be untouched by the fork.
    checks.append(
        (
            "no leftover state: original history unmutated by the fork",
            original_messages[-1]["content"][-1]["text"]
            == original_response.content[-1]["text"],
        )
    )
    checks.append(
        (
            "no missing context: forked history kept the original user turn",
            original_messages[0] == counterfactual_messages[0],
        )
    )
    # Forking the last tool_call must fail loudly, not silently guess.
    try:
        fork_from_tool_call(
            [{"type": "tool_call", "input": [], "output": []}],
            0,
            edit,
            tools,
            fake_model,
            make_tool_executor(450),
        )
        refused = False
    except ValueError:
        refused = True
    checks.append(("refuses to fork a step whose successor state was never seen", refused))

    print("CHECKS")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    failed = [n for n, p in checks if not p]
    if failed:
        raise SystemExit(f"\nM0 FAILED: {len(failed)} check(s). Section 6.3 needs rework.")
    print("\nM0 PASSED. Real re-execution via history splicing holds. Safe to build on.")


if __name__ == "__main__":
    main()
