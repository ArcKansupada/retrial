"""Live-API validation. Costs real tokens. Opt in with: pytest -m live

Covers what a scripted model cannot:

  1. the serializer against real Pydantic `Message` objects
  2. real tool_use/tool_result blocks carrying `tool_use_id`
  3. thinking blocks surviving a round-trip through retrial's JSON
  4. that forking is live: two forks from one SHA can diverge

Cost: roughly $0.10-0.30 for the full file at effort=low.
"""

import json
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "examples"))

import live_booking_agent as live  # noqa: E402

from retrial import fork, record  # noqa: E402
from retrial.diff import diff, final_answer  # noqa: E402
from retrial.trajectory import trajectory  # noqa: E402

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    live.load_env()
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        pytest.skip("no credentials; set ANTHROPIC_API_KEY or create a .env file")
    return live.make_client()


@pytest.fixture
def agent(store):
    return record(session_name="live-booking", store=store)(live.run_agent.__wrapped__)


@pytest.fixture
def deps(client):
    return (live.TOOLS, live.make_call_model(client), live.execute_tools)


@pytest.fixture
def recorded(store, agent, deps):
    agent(
        [{"role": "user", "content": "Book me a flight from AUS to SFO."}],
        *deps,
    )
    return agent.last_session_id


def tool_step(store, session_id, name="search_flight"):
    for step in store.steps_for(session_id):
        if step["step_type"] == "tool_call" and any(
            b.get("name") == name for b in step["input"]
        ):
            return step
    raise AssertionError(f"no {name} tool_call in {session_id}")


# --- 1. the serializer against real SDK objects ----------------------------


def test_records_a_real_run(store, recorded):
    steps = store.steps_for(recorded)
    assert [s["step_type"] for s in steps][:2] == ["model_call", "tool_call"]
    assert store.get_session(recorded)["status"] == "complete"
    # Real usage came back and was captured, not silently dropped.
    assert steps[0]["tokens_used"] and steps[0]["tokens_used"] > 0


def test_a_real_message_object_round_trips_losslessly(store, recorded):
    """to_jsonable must not lose or mangle a Pydantic Message."""
    first = store.steps_for(recorded)[0]
    out = first["output"]
    assert out["type"] == "message"
    assert out["model"].startswith("claude-opus-4-8")
    assert out["stop_reason"] == "tool_use"
    assert isinstance(out["usage"]["input_tokens"], int)
    # And it survived the trip through SQLite as JSON.
    assert json.loads(json.dumps(out)) == out


def test_real_tool_blocks_carry_the_id_the_splice_needs(store, recorded):
    """fork.py matches recorded output back into history by tool_use_id."""
    step = tool_step(store, recorded)
    assert step["input"][0]["type"] == "tool_use"
    assert step["input"][0]["id"].startswith("toolu_")
    assert step["output"][0]["tool_use_id"] == step["input"][0]["id"]


def test_thinking_blocks_survive_the_round_trip(store, agent, client):
    """The hardest serializer path: thinking blocks and their signatures.

    Runs at effort=high so the model actually thinks. The API rejects an altered thinking block,
    so a second successful model call proves the round-trip.
    """
    agent(
        [{"role": "user", "content": "Book me a flight from AUS to SFO."}],
        live.TOOLS,
        live.make_call_model(client, effort="high"),
        live.execute_tools,
    )
    steps = store.steps_for(agent.last_session_id)

    model_calls = [s for s in steps if s["step_type"] == "model_call"]
    thinking = [
        b for s in model_calls for b in s["output"]["content"] if b["type"] == "thinking"
    ]
    assert thinking, "no thinking blocks at effort=high — this test proved nothing"
    assert all("signature" in b and b["signature"] for b in thinking)

    # Prove the echo happened rather than assuming it.
    assert len(model_calls) > 1, "run ended too early to echo thinking back"
    echoed = [
        b
        for message in model_calls[1]["input"]["messages"]
        if isinstance(message.get("content"), list)
        for b in message["content"]
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    assert echoed, "thinking blocks were not echoed into the next call's history"
    assert echoed[0]["signature"] == thinking[0]["signature"]


# --- 2. forking a real run -------------------------------------------------


def test_fork_with_an_edit_steers_a_real_model(store, recorded, agent, deps):
    """Substitute the fare, and the agent should decline to book."""
    step = tool_step(store, recorded)
    original_answer = final_answer(trajectory(store, recorded))

    edited = json.loads(step["output"][0]["content"])
    edited["fare_usd"] = 1200
    fork_id = fork(
        from_sha=step["sha"],
        edit={
            "op": "replace",
            "path": "/output/0/content",
            "value": json.dumps(edited),
        },
        agent=agent,
        store=store,
        agent_args=deps,
    )

    forked_answer = final_answer(trajectory(store, fork_id))
    assert forked_answer != original_answer
    # It must not have booked: the $1200 fare is over the stated $600 budget.
    tools_used = [
        b["name"]
        for s in store.steps_for(fork_id)
        if s["step_type"] == "tool_call"
        for b in s["input"]
    ]
    assert "book_flight" not in tools_used, (
        f"agent booked an over-budget fare; answer was: {forked_answer!r}"
    )


def test_diff_finds_the_divergence_on_a_real_pair(store, recorded, agent, deps):
    step = tool_step(store, recorded)
    edited = json.loads(step["output"][0]["content"])
    edited["fare_usd"] = 1200
    fork_id = fork(
        from_sha=step["sha"],
        edit={"op": "replace", "path": "/output/0/content", "value": json.dumps(edited)},
        agent=agent,
        store=store,
        agent_args=deps,
    )

    result = diff(store, recorded, fork_id)
    assert result["identical"] is False
    assert result["common_ancestor"] == recorded
    # The alignment must independently rediscover the recorded fork point.
    assert result["divergence"]["sha"] == store.get_session(fork_id)["parent_sha"]


# --- 3. the test only a real model can pass --------------------------------


def test_two_forks_from_one_sha_re_execute_independently(store, recorded, agent, deps):
    """Fork the same step twice with no edit. Each must genuinely re-run.

    Identical answers are fine; identical SHAs would mean nothing re-ran.
    """
    step = tool_step(store, recorded)

    runs = []
    for _ in range(2):
        fork_id = fork(
            from_sha=step["sha"], edit=None, agent=agent, store=store, agent_args=deps
        )
        runs.append(
            {
                "id": fork_id,
                "steps": store.steps_for(fork_id),
                "answer": final_answer(trajectory(store, fork_id)),
            }
        )

    a, b = runs
    assert a["id"] != b["id"]
    assert a["steps"] and b["steps"], "a probe recorded nothing — it never ran"

    # The real evidence is each fork producing its own model output.
    assert a["steps"][0]["sha"] != b["steps"][0]["sha"]
    assert a["steps"][0]["output"]["id"] != b["steps"][0]["output"]["id"], (
        "both forks report the same API message id — nothing was re-executed"
    )
    assert a["answer"] and b["answer"]


def test_a_pure_replay_fork_reaches_the_same_conclusion(store, recorded, agent, deps):
    """Forking with no edit should still book.

    The control for the edit test above.
    """
    step = tool_step(store, recorded)
    fork_id = fork(
        from_sha=step["sha"], edit=None, agent=agent, store=store, agent_args=deps
    )
    tools_used = [
        b["name"]
        for s in store.steps_for(fork_id)
        if s["step_type"] == "tool_call"
        for b in s["input"]
    ]
    assert "book_flight" in tools_used
