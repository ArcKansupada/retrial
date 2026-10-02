"""Bisect: localizing which step made a failure inevitable.

The first `search_flight` call times out; later calls succeed.

    fork from step 0  -> the tool is called again and succeeds -> recovered
    fork from step 1  -> the timeout is replayed              -> still broken

So the boundary is step 1.
"""

import json

import pytest

from retrial import record
from retrial.bisect import CheckError, bisect, forkable_steps, parse_check
from retrial.errors import RetrialError

# --- the flaky world -------------------------------------------------------


def flaky_model(messages, tools=None):
    last = messages[-1]
    if isinstance(last["content"], str):
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_01", "name": "search_flight",
                 "input": {"route": "AUS-SFO"}}
            ],
        }
    results = [b for b in last["content"] if b.get("type") == "tool_result"]
    payload = json.loads(results[-1]["content"])
    if "error" in payload:
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "I couldn't reach the airline."}],
        }
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": f"Booked for ${payload['flight_price']}."}],
    }


def make_flaky_executor():
    """Times out once, then works. The transient failure the original hit."""
    calls = {"n": 0}

    def execute_tools(response):
        out = []
        for block in response["content"]:
            if block.get("type") != "tool_use":
                continue
            calls["n"] += 1
            payload = (
                {"error": "timeout"} if calls["n"] == 1 else {"flight_price": 450}
            )
            out.append(
                {"type": "tool_result", "tool_use_id": block["id"],
                 "content": json.dumps(payload)}
            )
        return out

    return execute_tools


def loop(messages, tools, call_model, execute_tools):
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response["content"]})
        if response["stop_reason"] != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


@pytest.fixture
def broken_run(store, opening):
    agent = record(session_name="flaky", store=store)(loop)
    executor = make_flaky_executor()
    response = agent(opening, [], flaky_model, executor)
    assert response["content"][-1]["text"] == "I couldn't reach the airline."
    return agent, agent.last_session_id, executor


# --- parse_check -----------------------------------------------------------


@pytest.mark.parametrize(
    "expression, answer, expected",
    [
        ("output contains 'Booked'", "Booked for $450.", True),
        ("output contains 'Booked'", "I couldn't reach the airline.", False),
        ("output not contains 'error'", "all good", True),
        ("output not contains 'error'", "an error occurred", False),
        (r"output matches '\$[0-9]+'", "Booked for $450.", True),
        (r"output matches '\$[0-9]+'", "no price here", False),
        ('output contains "double quotes"', "has double quotes here", True),
        ("OUTPUT CONTAINS 'case'", "case insensitive keyword", True),
    ],
)
def test_parse_check(expression, answer, expected):
    assert parse_check(expression)(answer) is expected


def test_check_handles_a_missing_answer():
    assert parse_check("output contains 'x'")(None) is False
    assert parse_check("output not contains 'x'")(None) is True


@pytest.mark.parametrize(
    "expression",
    ["nonsense", "output equals 'x'", "output contains x", "output matches '['"],
)
def test_unparseable_checks_fail_loudly(expression):
    with pytest.raises(CheckError):
        parse_check(expression)


# --- forkable steps --------------------------------------------------------


def test_a_trailing_tool_call_is_not_a_bisect_candidate(store, opening):
    """Its successor state was never observed, so it cannot be resumed."""

    def crash(messages, tools, call_model, execute_tools):
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response["content"]})
        execute_tools(response)
        raise RuntimeError("boom")

    agent = record(session_name="crash", store=store)(crash)
    with pytest.raises(RuntimeError):
        agent(opening, [], flaky_model, make_flaky_executor())

    assert len(store.steps_for(agent.last_session_id)) == 2
    assert [s["step_type"] for s in forkable_steps(store, agent.last_session_id)] == [
        "model_call"
    ]


# --- bisect ----------------------------------------------------------------


def test_bisect_localizes_the_transient_failure(store, broken_run):
    agent, session_id, executor = broken_run

    result = bisect(
        store,
        session_id,
        "output contains 'Booked'",
        agent=agent,
        agent_args=([], flaky_model, executor),
    )

    culprit = result["culprit"]
    assert culprit["step_type"] == "tool_call"
    assert culprit["step_number"] == 1
    assert json.loads(culprit["output"][0]["content"]) == {"error": "timeout"}
    assert result["inherent"] is False
    assert result["unreproducible"] is False


def test_bisect_probes_are_real_re_executions_recorded_as_sessions(store, broken_run):
    """Every probe is recorded as its own session."""
    agent, session_id, executor = broken_run
    before = len(store.list_sessions())

    result = bisect(
        store, session_id, "output contains 'Booked'",
        agent=agent, agent_args=([], flaky_model, executor),
    )

    assert len(store.list_sessions()) == before + result["re_executions"]
    for probe in result["probes"]:
        session = store.get_session(probe["session_id"])
        assert session["parent_session_id"] == session_id
        assert store.steps_for(probe["session_id"])  # actually ran


def test_bisect_recovers_when_forked_from_the_start(store, broken_run):
    """The step-0 probe must pass, or there is no boundary to find."""
    agent, session_id, executor = broken_run

    result = bisect(
        store, session_id, "output contains 'Booked'",
        agent=agent, agent_args=([], flaky_model, executor),
    )
    step_zero = [p for p in result["probes"] if p["step_number"] == 0]
    assert step_zero and step_zero[0]["passed"] is True
    assert step_zero[0]["answer"] == "Booked for $450."


def test_bisect_is_logarithmic(store, broken_run):
    """3 candidates -> 2 probes."""
    agent, session_id, executor = broken_run

    result = bisect(
        store, session_id, "output contains 'Booked'",
        agent=agent, agent_args=([], flaky_model, executor),
    )
    assert len(result["candidates"]) == 3
    assert result["re_executions"] == 2


def test_bisect_reports_an_inherent_failure(store, opening):
    """A tool that always fails is not localizable to a step."""

    def always_broken(response):
        return [
            {"type": "tool_result", "tool_use_id": b["id"],
             "content": json.dumps({"error": "timeout"})}
            for b in response["content"] if b.get("type") == "tool_use"
        ]

    agent = record(session_name="doomed", store=store)(loop)
    agent(opening, [], flaky_model, always_broken)

    result = bisect(
        store, agent.last_session_id, "output contains 'Booked'",
        agent=agent, agent_args=([], flaky_model, always_broken),
    )
    assert result["inherent"] is True
    assert result["culprit"]["step_number"] == 0


def test_bisect_refuses_when_the_check_already_passes(store, opening):
    """Nothing to localize. Erroring beats reporting a meaningless culprit."""
    agent = record(session_name="fine", store=store)(loop)
    executor = make_flaky_executor()
    executor(  # burn the flaky first call so the run succeeds
        {"content": [{"type": "tool_use", "id": "warmup", "name": "search_flight"}]}
    )
    agent(opening, [], flaky_model, executor)

    with pytest.raises(RetrialError, match="already passes"):
        bisect(
            store, agent.last_session_id, "output contains 'Booked'",
            agent=agent, agent_args=([], flaky_model, executor),
        )


def test_bisect_accepts_a_callable_check(store, broken_run):
    agent, session_id, executor = broken_run

    def booked_it(answer):
        return "Booked" in (answer or "")

    result = bisect(
        store, session_id, booked_it,
        agent=agent, agent_args=([], flaky_model, executor),
    )
    assert result["culprit"]["step_number"] == 1
    assert result["check"] == "booked_it"


def test_samples_must_be_positive(store, broken_run):
    agent, session_id, executor = broken_run
    with pytest.raises(CheckError, match="at least 1"):
        bisect(
            store, session_id, "output contains 'Booked'",
            agent=agent, agent_args=([], flaky_model, executor), samples=0,
        )


def test_on_probe_streams_progress(store, broken_run):
    agent, session_id, executor = broken_run
    seen = []
    bisect(
        store, session_id, "output contains 'Booked'",
        agent=agent, agent_args=([], flaky_model, executor),
        on_probe=seen.append,
    )
    assert [p["step_number"] for p in seen] == [1, 0]
