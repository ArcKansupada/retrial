"""Recording an async agent.

The async wrapper awaits the body and the interception points. The last test checks that
concurrent forks under asyncio.gather each record into their own session, which the ContextVar
handoff guarantees.
"""

import asyncio
import importlib
import json

import pytest
from conftest import TOOLS, fake_model, make_executor

from retrial import record

COUNT = 6


# -- async stand-ins for the conftest sync fixtures ---------------------------


async def async_model(messages, tools=None):
    return fake_model(messages, tools)


def make_async_executor(flight_price):
    sync = make_executor(flight_price)

    async def execute_tools(response):
        return sync(response)

    return execute_tools


async def async_agent(messages, tools, call_model, execute_tools):
    """Awaits both interception points - the async-through-and-through shape."""
    while True:
        response = await call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": await execute_tools(response)})


async def async_agent_sync_hooks(messages, tools, call_model, execute_tools):
    """Async agent, synchronous hooks - the hooks are called, not awaited."""
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


def make_exploding_model():
    """Succeeds once, then raises - so a crash happens with steps already recorded."""
    state = {"n": 0}

    async def model(messages, tools=None):
        state["n"] += 1
        if state["n"] >= 2:
            raise RuntimeError("model exploded")
        return fake_model(messages, tools)

    return model


# -- the async path records, and completes after the body runs ----------------


def test_calling_an_async_agent_returns_a_coroutine(store, opening):
    """The complement of the sync guarantee: an async agent stays awaitable."""
    agent = record(store=store)(async_agent)
    coro = agent(list(opening), TOOLS, async_model, make_async_executor(450))
    assert asyncio.iscoroutine(coro)
    asyncio.run(coro)  # drive it, so there is no un-awaited coroutine warning


def test_an_async_agent_records_its_steps_and_completes(store, opening):
    agent = record(store=store)(async_agent)
    result = asyncio.run(
        agent(list(opening), TOOLS, async_model, make_async_executor(450))
    )

    assert result.stop_reason == "end_turn"
    sessions = store.list_sessions()
    assert len(sessions) == 1
    # complete AND non-empty: the old bug stamped complete over zero steps.
    assert sessions[0]["status"] == "complete"
    steps = store.steps_for(sessions[0]["id"])
    assert [s["step_type"] for s in steps] == ["model_call", "tool_call", "model_call"]


def test_an_async_call_model_is_awaited_not_recorded_as_a_coroutine(store, opening):
    """A real response here proves the call was awaited."""
    agent = record(store=store)(async_agent)
    asyncio.run(agent(list(opening), TOOLS, async_model, make_async_executor(450)))

    first = store.steps_for(store.list_sessions()[0]["id"])[0]
    assert first["output"]["stop_reason"] == "tool_use"
    assert first["output"]["content"]  # the model's real content, not a coroutine


def test_an_async_agent_that_raises_partway_is_failed_with_steps_kept(store, opening):
    agent = record(store=store)(async_agent)
    with pytest.raises(RuntimeError, match="model exploded"):
        asyncio.run(
            agent(list(opening), TOOLS, make_exploding_model(), make_async_executor(450))
        )

    session = store.list_sessions()[0]
    assert session["status"] == "failed"
    assert store.steps_for(session["id"]), "steps before the crash were dropped"


def test_an_async_agent_with_sync_hooks_records(store, opening):
    """Mixed shape: async loop, synchronous call_model/execute_tools."""
    agent = record(store=store)(async_agent_sync_hooks)
    asyncio.run(agent(list(opening), TOOLS, fake_model, make_executor(450)))

    session = store.list_sessions()[0]
    assert session["status"] == "complete"
    assert store.steps_for(session["id"])


# -- phase 2: re-execution drives async agents --------------------------------


def test_fork_re_executes_an_async_agent(store, opening):
    """fork drives an async agent via asyncio.run, from sync code."""
    from retrial import fork

    agent = record(store=store)(async_agent)
    asyncio.run(agent(list(opening), TOOLS, async_model, make_async_executor(450)))
    root = store.list_sessions()[0]["id"]
    tool = next(s for s in store.steps_for(root) if s["step_type"] == "tool_call")

    fork_id = fork(
        from_sha=tool["sha"],
        edit={
            "op": "replace",
            "path": "/output/0/content",
            "value": json.dumps({"flight_price": 999}),
        },
        agent=agent,
        store=store,
        agent_args=(TOOLS, async_model, make_async_executor(999)),
    )

    forked = store.get_session(fork_id)
    assert forked["parent_session_id"] == root
    assert forked["status"] == "complete"
    assert store.steps_for(fork_id), "the async fork recorded no steps"


def test_fork_from_inside_a_running_loop_refuses(store, opening):
    """asyncio.run cannot nest, so fork refuses before creating a session row."""
    from retrial import fork
    from retrial.errors import IntegrationError

    agent = record(store=store)(async_agent)
    asyncio.run(agent(list(opening), TOOLS, async_model, make_async_executor(450)))
    root = store.list_sessions()[0]["id"]
    tool = next(s for s in store.steps_for(root) if s["step_type"] == "tool_call")

    async def fork_from_within_a_loop():
        return fork(
            from_sha=tool["sha"],
            agent=agent,
            store=store,
            agent_args=(TOOLS, async_model, make_async_executor(450)),
        )

    with pytest.raises(IntegrationError, match="running event loop"):
        asyncio.run(fork_from_within_a_loop())

    assert [s["id"] for s in store.list_sessions()] == [root], "a fork row leaked"


def test_rerun_drives_async_agents(store, opening):
    """rerun re-executes a recorded async run from ordinary sync code."""
    from retrial import rerun

    agent = record(store=store)(async_agent)
    asyncio.run(agent(list(opening), TOOLS, async_model, make_async_executor(450)))

    result = rerun(
        store,
        "output contains 'Booked'",
        agent=agent,
        agent_args=(TOOLS, async_model, make_async_executor(450)),
    )

    assert result["model_calls"] > 0
    assert not result["regressed"]  # same code and inputs, so nothing regressed


# -- the load-bearing one: concurrent forks keep separate sessions ------------


def test_concurrent_async_forks_do_not_cross_sessions(store, opening):
    """Each task sets the fork handoff; every session must get exactly its own steps."""
    record_module = importlib.import_module("retrial.record")
    agent = record(store=store)(async_agent)
    sessions = [store.create_session(name=f"fork-{i}") for i in range(COUNT)]

    async def run_all():
        async def one(i):
            record_module._pending.set(
                {"store": store, "session_id": sessions[i]}
            )
            # Yield so every task publishes its handoff before any consumes it.
            await asyncio.sleep(0)
            await agent(list(opening), TOOLS, async_model, make_async_executor(450))

        await asyncio.gather(*(one(i) for i in range(COUNT)))

    asyncio.run(run_all())

    for session_id in sessions:
        steps = store.steps_for(session_id)
        numbers = [s["step_number"] for s in steps]
        # A leak shows as uneven step counts or non-contiguous numbering.
        assert numbers == list(range(len(numbers))), "steps crossed between sessions"
        assert len(steps) == 3, f"{session_id} got {len(steps)} steps, not its own 3"
