from dataclasses import dataclass

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import record
from retrial.errors import IntegrationError


def test_records_the_whole_loop_with_a_sha_per_step(store, opening):
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    steps = store.steps_for(agent.last_session_id)
    assert [s["step_type"] for s in steps] == ["model_call", "tool_call", "model_call"]
    assert [s["step_number"] for s in steps] == [0, 1, 2]
    assert all(len(s["sha"]) == 64 for s in steps)
    assert len({s["sha"] for s in steps}) == 3
    assert store.get_session(agent.last_session_id)["status"] == "complete"


def test_model_call_input_is_the_history_verbatim_at_call_time(store, opening):
    """The snapshot must freeze the history as it was, not alias the live list."""
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    steps = store.steps_for(agent.last_session_id)
    first, second = [s for s in steps if s["step_type"] == "model_call"]

    assert len(first["input"]["messages"]) == 1
    assert first["input"]["messages"][0]["content"] == "Book me AUS to SFO."
    # By the second model call the snapshot shows the assistant turn and tool result.
    assert len(second["input"]["messages"]) == 3
    assert second["input"]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_state_after_a_tool_call_is_the_next_model_calls_input(store, opening):
    """The observation the entire fork engine rests on."""
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    steps = store.steps_for(agent.last_session_id)
    tool_call = steps[1]
    following = steps[2]

    recorded_result = tool_call["output"][0]
    history = following["input"]["messages"]
    blocks = [b for m in history if isinstance(m["content"], list) for b in m["content"]]
    assert recorded_result in blocks


def test_tool_call_records_the_invocation_and_the_result(store, opening):
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    tool_call = store.steps_for(agent.last_session_id)[1]
    assert [b["name"] for b in tool_call["input"]] == ["search_flight"]
    assert tool_call["output"][0]["tool_use_id"] == "toolu_01"
    assert tool_call["duration_ms"] is not None


def test_usage_is_captured_when_the_response_has_it(store, opening):
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    model_calls = [
        s for s in store.steps_for(agent.last_session_id) if s["step_type"] == "model_call"
    ]
    assert all(s["tokens_used"] == 15 for s in model_calls)


def test_missing_usage_is_not_an_error(store, opening):
    """Token accounting is a nicety; recording must not depend on it."""

    @dataclass
    class NoUsageResponse:
        stop_reason: str
        content: list

    def usageless_model(messages, tools=None):
        response = fake_model(messages, tools)
        return NoUsageResponse(response.stop_reason, response.content)

    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, usageless_model, make_executor(450))
    assert store.steps_for(agent.last_session_id)[0]["tokens_used"] is None


def test_a_crashed_run_keeps_its_steps_and_is_marked_failed(store, opening):
    """A broken run is the one you most want to inspect. Don't lose it."""

    def boom(response):
        raise RuntimeError("tool exploded")

    agent = record(session_name="booking", store=store)(raw_agent)
    with pytest.raises(RuntimeError, match="tool exploded"):
        agent(opening, TOOLS, fake_model, boom)

    assert store.get_session(agent.last_session_id)["status"] == "failed"
    assert len(store.steps_for(agent.last_session_id)) == 1


def test_two_runs_are_separate_sessions(store, opening):
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(list(opening), TOOLS, fake_model, make_executor(450))
    first = agent.last_session_id
    agent(list(opening), TOOLS, fake_model, make_executor(450))
    second = agent.last_session_id

    assert first != second
    assert len(store.list_sessions()) == 2
    # Same content in a different session => different SHA, by construction.
    assert store.steps_for(first)[0]["sha"] != store.steps_for(second)[0]["sha"]


def test_the_integration_contract_is_enforced_at_decoration_time(store):
    """Fail when the developer writes the code, not mid-run."""

    with pytest.raises(IntegrationError, match="'messages' parameter"):

        @record(store=store)
        def no_messages(tools, call_model, execute_tools):
            pass

    with pytest.raises(IntegrationError, match="'call_model' parameter"):

        @record(store=store)
        def no_model(messages, tools, execute_tools):
            pass


def test_custom_argument_names_are_supported(store, opening):
    @record(store=store, model_arg="llm", tools_arg="run_tools")
    def agent(messages, tools, llm, run_tools):
        return raw_agent(messages, tools, llm, run_tools)

    agent(opening, TOOLS, fake_model, make_executor(450))
    assert len(store.steps_for(agent.last_session_id)) == 3


def test_a_non_callable_interception_point_fails_at_the_boundary(store, opening):
    """@record wraps call_model/execute_tools before the body runs.

    So a `None` sentinel must be refused at the boundary.
    """
    agent = record(session_name="booking", store=store)(raw_agent)

    with pytest.raises(IntegrationError, match="not callable"):
        agent(opening, TOOLS, None, make_executor(450))

    with pytest.raises(IntegrationError, match="cannot be built lazily"):
        agent(opening, TOOLS, fake_model, None)


def test_a_rejected_call_does_not_strand_an_empty_session(store, opening):
    """Validation must happen before the store is touched, leaving no orphaned session."""
    agent = record(session_name="booking", store=store)(raw_agent)
    with pytest.raises(IntegrationError):
        agent(opening, TOOLS, None, make_executor(450))

    assert store.list_sessions() == []
    assert agent.last_session_id is None
