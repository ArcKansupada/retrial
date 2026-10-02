"""Trajectory materialization and the diff engine."""

import json

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import fork, record
from retrial.diff import diff, final_answer, signature
from retrial.trajectory import trajectory

PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


@pytest.fixture
def original(store, opening):
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))
    return agent, agent.last_session_id


def tool_step(store, session_id):
    return next(s for s in store.steps_for(session_id) if s["step_type"] == "tool_call")


def make_fork(store, agent, session_id, edit=PRICE_999):
    return fork(
        from_sha=tool_step(store, session_id)["sha"],
        edit=edit,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )


# --- trajectory ------------------------------------------------------------


def test_a_root_trajectory_is_just_its_own_steps(store, original):
    _, session_id = original
    path = trajectory(store, session_id)
    assert [e["step_type"] for e in path] == ["model_call", "tool_call", "model_call"]
    assert all(e["origin"] == "live" for e in path)


def test_a_fork_trajectory_walks_up_the_parent_chain(store, original):
    """The fork stores 3 steps but its trajectory is 5; the prefix lives in the parent."""
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    assert len(store.steps_for(fork_id)) == 3
    path = trajectory(store, fork_id)
    assert len(path) == 5
    assert [e["origin"] for e in path] == [
        "replayed",
        "replayed",
        "live",
        "live",
        "live",
    ]


def test_the_replayed_prefix_is_the_parents_steps_verbatim(store, original):
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    parent = trajectory(store, session_id)
    forked = trajectory(store, fork_id)
    assert forked[0] == dict(parent[0], origin="replayed")


def test_the_forked_step_shows_what_the_fork_actually_saw(store, original):
    """The fork's view of the edited step, not what the parent recorded."""
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    forked_step = trajectory(store, fork_id)[1]
    assert forked_step["edited"] is True
    assert json.loads(forked_step["output"][0]["content"])["flight_price"] == 999
    # The parent still records what really happened.
    assert json.loads(trajectory(store, session_id)[1]["output"][0]["content"])[
        "flight_price"
    ] == 450


def test_a_callback_edits_effect_is_recovered_even_though_it_doesnt_round_trip(
    store, original
):
    """A callback's effect is read from the fork's first recorded model input."""
    agent, session_id = original

    def triple(step):
        payload = json.loads(step["output"][0]["content"])
        payload["flight_price"] *= 3  # 1350: no stored patch could tell us this
        step["output"][0]["content"] = json.dumps(payload)
        return step

    fork_id = make_fork(store, agent, session_id, edit=triple)
    forked_step = trajectory(store, fork_id)[1]
    assert json.loads(forked_step["output"][0]["content"])["flight_price"] == 1350
    assert forked_step["edit"]["type"] == "callback"


def test_a_fork_of_a_fork_materializes_the_whole_chain(store, original):
    agent, session_id = original
    first = make_fork(store, agent, session_id)
    second = fork(
        from_sha=tool_step(store, first)["sha"],  # the fork's own check_budget
        edit={
            "op": "replace",
            "path": "/output/0/content",
            "value": json.dumps({"approved": True}),
        },
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )
    path = trajectory(store, second)
    # root prefix + first fork's prefix + second fork's live steps
    assert [e["origin"] for e in path] == [
        "replayed",
        "replayed",
        "replayed",
        "replayed",
        "live",
    ]
    assert len({e["session_id"] for e in path}) == 3


# --- signatures ------------------------------------------------------------


def test_signature_folds_in_type_tool_name_and_output(store, original):
    _, session_id = original
    path = trajectory(store, session_id)
    assert signature(path[0]).startswith("model_call|tool_use|")
    assert signature(path[1]).startswith("tool_call|search_flight|")


def test_same_step_different_output_gets_a_different_signature(store, original):
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)
    parent_step = trajectory(store, session_id)[1]
    forked_step = trajectory(store, fork_id)[1]

    # Same step and tool, but the fork saw a different result.
    assert parent_step["sha"] == forked_step["sha"]
    assert signature(parent_step) != signature(forked_step)


# --- diff ------------------------------------------------------------------


def test_diff_finds_the_shared_prefix_and_divergence(store, original):
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    result = diff(store, session_id, fork_id)
    assert result["identical"] is False
    assert result["common_ancestor"] == session_id
    assert len(result["shared_prefix"]) == 1
    assert result["shared_prefix"][0]["step_type"] == "model_call"


def test_alignment_rediscovers_the_fork_point_on_its_own(store, original):
    """The divergence SHA must equal parent_sha, derived from signatures alone.

    Two independent mechanisms must agree.
    """
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    result = diff(store, session_id, fork_id)
    assert result["divergence"]["sha"] == store.get_session(fork_id)["parent_sha"]


def test_diff_surfaces_the_edit_that_caused_the_divergence(store, original):
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    result = diff(store, session_id, fork_id)
    assert result["divergence"]["edit"] == {"type": "patch", "patch": PRICE_999}


def test_diff_reports_the_final_answers(store, original):
    agent, session_id = original
    fork_id = make_fork(store, agent, session_id)

    result = diff(store, session_id, fork_id)
    assert result["final"]["a"] == "Booked for $450."
    assert result["final"]["b"] == "Over budget. Need approval first."


def test_a_session_is_identical_to_itself(store, original):
    _, session_id = original
    result = diff(store, session_id, session_id)
    assert result["identical"] is True
    assert result["divergence"] is None
    assert len(result["shared_prefix"]) == 3


def test_two_identical_independent_runs_diff_clean(store, opening):
    """Same inputs -> no divergence. Signatures must not include the session id."""
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(list(opening), TOOLS, fake_model, make_executor(450))
    a = agent.last_session_id
    agent(list(opening), TOOLS, fake_model, make_executor(450))
    b = agent.last_session_id

    result = diff(store, a, b)
    assert a != b
    assert result["identical"] is True
    assert result["common_ancestor"] is None


def test_independent_runs_that_differ_have_no_shared_prefix(store, opening):
    """The first tool result differs, so the shared prefix is exactly one step."""
    agent = record(session_name="booking", store=store)(raw_agent)
    agent(list(opening), TOOLS, fake_model, make_executor(450))
    a = agent.last_session_id
    agent(list(opening), TOOLS, fake_model, make_executor(999))
    b = agent.last_session_id

    result = diff(store, a, b)
    assert result["common_ancestor"] is None
    assert len(result["shared_prefix"]) == 1
    assert result["final"]["a"] != result["final"]["b"]


def test_sibling_forks_share_a_common_ancestor(store, original):
    agent, session_id = original
    cheap = make_fork(
        store,
        agent,
        session_id,
        edit={
            "op": "replace",
            "path": "/output/0/content",
            "value": json.dumps({"flight_price": 100}),
        },
    )
    pricey = make_fork(store, agent, session_id)

    result = diff(store, cheap, pricey)
    assert result["common_ancestor"] == session_id
    assert result["final"]["a"] == "Booked for $100."
    assert result["final"]["b"] == "Over budget. Need approval first."


def test_final_answer_is_none_when_nothing_was_said(store):
    assert final_answer([]) is None
